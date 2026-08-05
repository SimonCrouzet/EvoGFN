"""Run the benchmark suite, resuming whatever is already stored.

Safe to interrupt and safe to re-run. Every campaign is written the moment it
finishes, and a second invocation runs only what is missing -- so raising a
tier's seed count from 30 to 50 costs twenty campaigns per arm, not fifty.

    uv run python experiments/run_suite.py                  # everything
    uv run python experiments/run_suite.py --tier main      # headline only
    uv run python experiments/run_suite.py --seeds 50       # raise the count
    uv run python experiments/run_suite.py --report         # no runs, just read

Results land under ``results/`` as one JSONL file per task and method.

What the regret column is, and what it is not
---------------------------------------------

Regret here is against the **attainable** optimum -- what
[evogfn.benchmark.attainable][] audited a task's search space to contain --
rather than against the landscape's own. Regret against the landscape's optimum
carries a floor no method can clear, because the optimum need not sit inside the
space the protocol lets a method search. That floor is constant across arms, so
it contributes nothing a comparison can read and everything a reader can
misread.

Where the audit could not close the bracket, the interval is printed and the
regret is against its conservative end. A regret at or below zero therefore does
not mean an arm was perfect; it means the arm matched everything the audit could
construct, and the task has no demonstrated headroom left to separate it from a
better method with. Those arms are named as **solved**, and comparisons drawn on
them are marked vacuous rather than quietly reported -- a p-value against an arm
sitting on the ceiling is a statement about the ceiling.

What every arm is compared against
----------------------------------

The reference is `genetic`: the Ehrlich paper's own algorithm at its own
hyperparameters. Methods are compared **as published**, because a published
pipeline is what a lab actually chooses between. Pairing against
`genetic+search` instead -- a genetic algorithm handed the campaign's surrogate
-- would pair every headline number against something nobody proposed and no lab
runs.

`genetic+search` and `random+screen` stay in the table as **ablations** -- they
are the only thing separating "the surrogate won" from "the constructive sampler
won" -- but they are decomposition rows, not controls, and they are labelled as
such on every line they appear on. A reader who takes one for the yardstick is
reading exactly the comparison the reference change was made to stop.

Why proxy spend is printed beside the win
-----------------------------------------

Proxy calls are a *chosen* budget, not a constant of an architecture: the
GFlowNet's is ``steps x batch_size`` per round, the `genetic+search` ablation's is
``generations x population``. Let those differ by an order of magnitude and the
regret column is measuring compute rather than method. So the cost is printed
next to the win: a GFlowNet needing 10x the proxy calls is a real and publishable
cost of the method, and the column is where a reader finds it rather than
something to be inferred from a configuration file. One caveat, from
[evogfn.benchmark.methods][]: a sampler rebuilt at an anchor move restarts its own
accounting, so on a re-anchoring task the stored count covers the last anchor's
rounds rather than the campaign's, and reads as a floor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.methods import (
    BASELINES,
    OBJECTIVES,
    anchor_arms,
    flow_objectives,
    sensitivity,
    shipped_base,
    variant_arms,
)
from evogfn.benchmark.selection import _build_objective
from evogfn.benchmark.statistics import compare, seeds_needed
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import (
    MAIN,
    Purpose,
    Tier,
    anchor_study,
    budget_gradient,
    constraint_density_tier,
    objective_task,
    records_to_metric,
    replication,
    rounds_curve,
    run_anchor_study,
    run_tier,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from evogfn.benchmark.attainable import AttainableOptimum
    from evogfn.benchmark.store import RunRecord
    from evogfn.benchmark.tasks import Task

#: Seeds per tier. The main tiers differ because campaign cost differs by an
#: order of magnitude between L=4 and L=256, not because the claims differ.
MAIN_SEEDS = 100
LARGE_SPACE_SEEDS = 30
DIAGNOSTIC_SEEDS = 50

#: Everything except the flow objectives, which need a policy with a flow head
#: and are compared separately in the objectives diagnostic.
MAIN_METHODS = {**BASELINES, **OBJECTIVES}

#: Share of an arm's seeds sitting on the attainable optimum before the task is
#: called solved for that arm. Half is already fatal: a comparison against an arm
#: that hits the ceiling on half its seeds is measuring the ceiling on those, and
#: the p-value it reports is about the remainder.
VACUOUS_SHARE = 0.5

#: How close to the attainable optimum an arm has to sit to count as on it.
SOLVED_TOLERANCE = 1e-9

#: Named once because four places have to agree on it, and one of them is the
#: branch in `main` that sends this tier to `run_anchor_study` instead of
#: `run_tier`. A typo there does not raise -- it silently falls through to the
#: generic path, which runs the *cross* of the study's tasks and arms, and the
#: cross contains a campaign the study exists to keep out. A misspelling would
#: therefore cost compute and produce a duplicate row rather than an error.
ANCHOR_TIER = "anchor-study"

#: What the constructibility sweep runs. Spelled out rather than left to the
#: default, for two reasons that both silently produce a readable-looking table
#: full of nothing:
#:
#: * `genetic-gfn` is the only arm here that **breeds**, and
#:   ``unconstructible_fraction`` counts offspring a teacher proposed and the
#:   policy could not construct. Every other arm stores a share of nothing, which
#:   is zero -- so without this arm the curve the tier exists to draw is a column
#:   of zeros, and a column of zeros reads as "no gap" rather than "nothing bred".
#:   The default path would drop it: once a selection has been recorded,
#:   `methods_for` replaces *every* untuned GFlowNet arm with the chosen one.
#: * The other three are what the share has to be read against: `gfn-tb` masks at
#:   every step and so can only reach the constructible part in the first place,
#:   `genetic-feasible` pays for the same set by rejection, and `genetic` ignores
#:   the constraint entirely and is the reference the tier pairs against.
DENSITY_ARMS = ("genetic", "genetic-feasible", "gfn-tb", "genetic-gfn")


def _anchor_tasks() -> tuple[Task, ...]:
    """The distinct tasks of the anchor study, in cell order.

    Derived from the cells rather than listed here: the study names both of its
    tasks, and a second list would be right only until one of them moved.
    Deduplicated because both tasks carry more than one cell, and a tier
    repeating a task runs it twice.
    """
    seen: dict[str, Task] = {}
    for cell in anchor_study():
        seen.setdefault(cell.task.name, cell.task)
    return tuple(seen.values())


def tiers(main_seeds: int, diagnostic_seeds: int) -> list[Tier]:
    """The suite, split by what each tier is for.

    Args:
        main_seeds: Seeds for the headline tiers.
        diagnostic_seeds: Seeds for the diagnostics.

    Returns:
        Tiers in the order they should run: cheap and decisive first, so an
        interrupted night still yields something readable.
    """
    cheap = tuple(t for t in MAIN if t.name != "large-space")
    expensive = tuple(t for t in MAIN if t.name == "large-space")
    return [
        Tier("objectives", (objective_task(),), tuple(range(diagnostic_seeds)), Purpose.DIAGNOSTIC),
        # Shares the objectives task deliberately: same landscape, same
        # protocol, same seeds, so a setting's effect and an objective's are
        # measured against each other rather than across two configurations.
        Tier("sensitivity", (objective_task(),), tuple(range(diagnostic_seeds)), Purpose.SELECTION),
        # `Purpose.SELECTION` and not `DIAGNOSTIC`, which is the distinction the
        # enum exists to hold: both rungs are off by default, so what this tier
        # returns *chooses the configuration the method ships* rather than
        # describing how methods behave. Called a diagnostic it would be eligible
        # to appear in the results table as a mechanism finding while the same
        # campaigns had already fixed our own configuration -- a choice reported
        # as a result. On the diagnostic landscape, which no headline task uses,
        # so the choice is not being made on the test set.
        #
        # On `objective_task` for the reason the ladder is built by lookup: its
        # base rung is the stored `(objectives, gfn-tb)` cell, so the control the
        # three rungs are read against is the identical configuration by
        # construction, and its first `diagnostic_seeds` are already banked.
        #
        # Seeded like the headline rather than like a diagnostic, because the
        # rung that justifies the fourth arm is an *interaction*:
        # `+terminal+anchor` earns its compute only by beating what the two
        # single rungs predict by addition, and a difference of differences
        # carries about twice the standard error of either main effect. At the
        # diagnostic count that rung comes back inconclusive against its own
        # prediction, which is the one answer this tier must not return.
        Tier("variant-ladder", (objective_task(),), tuple(range(main_seeds)), Purpose.SELECTION),
        Tier("main", cheap, tuple(range(main_seeds)), Purpose.BENCHMARK),
        # Same arms and the same seed count as `main`, because it answers a
        # question about `main`: every constrained comparison there rests on one
        # draw from the generator, and a hundred seeds vary the wild type and the
        # surrogate's initialisation without ever varying the instance. A
        # lower-powered replicate could not distinguish "the ordering broke" from
        # "we ran out of seeds", which is the one thing it exists to decide.
        Tier("replication", replication(), tuple(range(main_seeds)), Purpose.BENCHMARK),
        Tier("rounds-curve", rounds_curve(), tuple(range(diagnostic_seeds)), Purpose.DIAGNOSTIC),
        Tier(
            "budget-gradient", budget_gradient(), tuple(range(diagnostic_seeds)), Purpose.DIAGNOSTIC
        ),
        # The sweep's own helper rather than a `Tier` assembled here: its task
        # list and its standing are properties of the sweep, and restating either
        # at the call site is how the two drift apart.
        #
        # Diagnostic seeds, and the same count as `objectives`, because the rung
        # at `DIAGNOSTIC_DENSITY` **is** the objectives task -- so at this count
        # the point the curve passes through is the very measurement the
        # objectives table reports, rather than a second estimate of it at a
        # different power that a reader would have to reconcile. Nothing here
        # argues for more: four of the five rungs declare no attainable optimum,
        # so their regret column is empty by design and this family is read on
        # the constructibility columns. Seeds bought for a regret comparison
        # would be buying precision on a column the tier does not report.
        constraint_density_tier(range(diagnostic_seeds)),
        # A tier for *reporting* that is **run** cell by cell -- see the branch
        # in `main`. The cross of these two tasks with `anchor_arms` is six
        # campaigns and the study is five: the cross also contains a rebuilt
        # policy on the task whose anchor never moves, and since nothing is ever
        # rebuilt there that is the carried arm's campaign under a second name.
        # Running the cross would pay twice for one experiment and put two
        # identically-valued rows in the table with nothing to say which was
        # which. Listing the tier anyway is what gets the study into `--report`,
        # where the unrun cell simply has no records and prints no row.
        #
        # Diagnostic seeds, and again the objectives count, because the
        # moved-and-carried cell *is* `(objectives, gfn-tb)`: it is the control
        # the other four cells are read against, and it is already stored at this
        # count. Asking for another would leave the control powered differently
        # from everything it controls.
        Tier(ANCHOR_TIER, _anchor_tasks(), tuple(range(diagnostic_seeds)), Purpose.DIAGNOSTIC),
        # Last, and on fewer seeds for the same reason: a campaign at L=256
        # costs far more than one on the cheap tiers.
        Tier("large-space", expensive, tuple(range(LARGE_SPACE_SEEDS)), Purpose.BENCHMARK),
    ]


#: Where the selection phase records the configuration it chose.
CHOICE_FILE = Path("results/selected.json")


def selected_gflownet() -> dict[str, object]:
    """The GFlowNet arm the selection phase chose, if it has run.

    Reading the decision rather than re-deriving it matters: re-deriving would
    silently pick a different arm the moment a seed count or an arm list moved,
    and the table would then report a configuration no selection ever chose.

    Returns:
        A single-entry mapping, or an empty one when no selection has been
        recorded -- in which case the caller falls back to the untuned defaults
        and says so, rather than quietly benchmarking them as though they had
        been chosen.
    """
    if not CHOICE_FILE.exists():
        return {}
    choice = json.loads(CHOICE_FILE.read_text())
    # Every axis the selection moved has to be here, including the ones whose
    # value may legitimately be null. Absent and null are different claims: null
    # says this objective reads no such knob, absent says the file predates the
    # stage that chose it -- and rebuilding the arm from a file missing an axis
    # would silently run the default on it and report the result as selected.
    missing = [
        key
        for key in ("objective", "arm", "beta", "steps", "lam", "mix", "hidden_dim")
        if key not in choice
    ]
    if missing:
        # Loudly, rather than falling back to the untuned defaults. A file
        # written by a selection that stopped partway describes a configuration
        # no rule ever chose, and running the defaults instead would produce a
        # headline table that silently disagrees with the selection page.
        raise ValueError(
            f"{CHOICE_FILE} is missing {', '.join(missing)}, so the selection it "
            f"records is unfinished; run experiments/select_configuration.py to "
            f"completion, or delete the file to benchmark the untuned defaults"
        )
    arm = str(choice["arm"])
    return {
        arm: _build_objective(
            str(choice["objective"]),
            float(choice["beta"]),
            int(choice["steps"]),
            lam=None if choice["lam"] is None else float(choice["lam"]),
            mix=None if choice["mix"] is None else float(choice["mix"]),
            hidden_dim=int(choice["hidden_dim"]),
        )
    }


def methods_for(tier: Tier) -> dict[str, object]:
    """Which methodologies a tier runs.

    The objectives diagnostic is GFlowNet-only, since a classical baseline has
    no training objective to vary; the sensitivity tier is narrower still, being
    one GFlowNet with one setting moved at a time; everything else compares
    methods.

    Every branch here sits **above** the selection lookup, and that placement is
    the point rather than an accident of ordering. Below it, a recorded selection
    replaces the untuned GFlowNet arms wholesale -- which is right for a tier
    comparing methods and fatal for the three tiers whose arms are defined
    *relative to* an untuned one. A ladder whose base rung had been swapped for
    the selected arm would report every rung as the difference between two
    configurations, and an anchor study missing `gfn-tb` would lose the cell its
    other four are read against.
    """
    fixed: dict[str, Callable[[], dict[str, object]]] = {
        "objectives": lambda: {**OBJECTIVES, **flow_objectives()},
        "sensitivity": lambda: dict(sensitivity()),
        # The four-arm ladder, not the baseline set: no arm in `BASELINES` has a
        # policy, so neither rung is even definable for one, and a table mixing
        # them would answer "does a GFlowNet beat a GA" -- which `main` already
        # answers -- in place of "does this rung add anything".
        "variant-ladder": lambda: dict(variant_arms()),
        # By name out of the shipped table rather than rebuilt, so these are the
        # same objects, and therefore the same store cells, that every other tier
        # runs. A name that stopped existing raises here instead of quietly
        # sweeping a smaller set.
        "constraint-density": lambda: {name: MAIN_METHODS[name] for name in DENSITY_ARMS},
        ANCHOR_TIER: lambda: dict(anchor_arms()),
    }
    if build := fixed.get(tier.name):
        return build()
    chosen = selected_gflownet()
    if not chosen:
        return dict(MAIN_METHODS)
    # The selected arm replaces the untuned GFlowNet arms rather than joining
    # them: keeping both would put two configurations of the same method in one
    # table, and the better of the two would be the one the selection was run to
    # avoid reporting.
    return {**BASELINES, **chosen}


def attainable_for(task: Task) -> AttainableOptimum | None:
    """What this task's search space was audited to contain.

    Args:
        task: The task being reported on.

    Returns:
        The audited optimum, or ``None`` for a task carrying no declaration or
        measuring more than one objective -- where the landscape's optimum is an
        ideal point and the gap to it is not a regret.
    """
    landscape = task.landscape()
    optimum = landscape.optimum
    if optimum is None or landscape.n_objectives != 1:
        return None
    return task.attainable_optimum(float(np.max(optimum)))


def _attainable_line(attainable: AttainableOptimum | None) -> str:
    """State the attainable optimum so a bound cannot be read as a measurement."""
    if attainable is None:
        return "  attainable: not audited, so no regret is reported for this task"
    if attainable.exact is not None:
        value = f"{attainable.exact:.4f} exact"
    else:
        value = f"[{attainable.lower:.4f}, {attainable.upper:.4f}]"
    floor = attainable.regret_floor
    gap = f"{floor[0]:.4f}" if np.isclose(*floor) else f"[{floor[0]:.4f}, {floor[1]:.4f}]"
    return (
        f"  attainable {value} of a nominal {attainable.nominal:.4f} "
        f"at {attainable.budget} cumulative mutations; regret against the nominal "
        f"would carry a floor of {gap}\n    ({attainable.method})"
    )


def _at_optimum(records: Mapping[int, RunRecord], attainable: AttainableOptimum | None) -> float:
    """Share of an arm's seeds sitting on the attainable optimum.

    Args:
        records: The arm's stored records, by seed.
        attainable: What the task can reach, or ``None`` when unaudited.

    Returns:
        A fraction in ``[0, 1]``, or ``nan`` when there is nothing to compare
        against. ``nan`` rather than zero: "no audit" is a different statement
        from "no seed reached it".
    """
    if attainable is None or not records:
        return float("nan")
    best = np.asarray([record.best for record in records.values()], dtype=np.float64)
    usable = best[np.isfinite(best)]
    if not usable.size:
        return float("nan")
    return float(np.mean(usable >= attainable.lower - SOLVED_TOLERANCE))


#: What each tier's arms are compared against. A tier that does not contain the
#: default reference gets its own, because `report` skips the paired section
#: silently when the reference is absent -- which reads as "nothing separated
#: these arms" when what happened is that nothing was tested.
REFERENCES = {
    # The shipped configuration, so each swept value is read as a change from
    # what the headline rows were produced at.
    "sensitivity": "steps-300",
    # Trajectory balance, the objective the others are alternatives to.
    "objectives": "gfn-tb",
    # The ladder's own base rung, which is what each rung is one step above.
    # `genetic` is not in this tier at all, so without an entry here the whole
    # paired section would be replaced by one line saying there was no reference
    # -- three rungs run and nothing compared.
    #
    # Resolved from the ladder itself rather than named, because the ladder is
    # now built on whatever configuration the selection recorded. A literal here
    # would name an arm the tier had stopped running the moment a selection
    # moved, and `reference_for` returns None for a name the tier does not hold
    # -- so the failure is a tier that quietly compares nothing, which is what
    # this entry exists to prevent.
    "variant-ladder": lambda: shipped_base().name,
    # The policy-carrying arm, so the pair that gets printed on the re-anchored
    # task is rebuilt-against-carried: the amortisation cell, and the only cell
    # in this study that a paired test can reach. Against the default reference
    # the study's own axis would never appear in a comparison, because `genetic`
    # has no learned state and so sits on neither side of it.
    #
    # The other axis -- moved against fixed -- is across two *tasks*, and
    # `report` pairs only within one. It is read off the two tables rather than
    # tested here.
    ANCHOR_TIER: "gfn-tb",
}

#: The Ehrlich paper's own algorithm, at its own hyperparameters. The reference
#: has to be a *published pipeline*, because a pipeline is what a lab chooses
#: between; pairing against `genetic+search` instead would pair every headline
#: number against a hybrid we invented, and no reviewer has to accept a win over
#: that.
DEFAULT_REFERENCE = "genetic"

#: Arms that decompose a published pipeline rather than being one, mapped to the
#: pipeline they decompose. They answer "was it the surrogate or the sampler?",
#: which is a real question and the first one a reviewer asks -- but it is an
#: attribution question, and its answer belongs in a decomposition row. Naming
#: them in the table is what keeps a reader from reading one as the yardstick:
#: drop this and `genetic+search` looks like just another baseline that lost.
ABLATIONS = {
    "genetic+search": "genetic",
    "random+screen": "random",
}


def reference_for(tier: Tier, methods: dict[str, object]) -> str | None:
    """Which arm a tier's comparisons are drawn against.

    Args:
        tier: The tier being reported on.
        methods: The arms it runs, so an absent reference is caught here rather
            than becoming a missing section further down.

    Returns:
        The reference arm's name, or ``None`` when the tier has no arm to serve
        as one -- said explicitly so the report can name the omission.
    """
    chosen = REFERENCES.get(tier.name, DEFAULT_REFERENCE)
    # A callable entry names an arm that is not fixed at import -- the ladder's
    # base rung follows the recorded selection -- and is resolved here so the
    # table below never has to know which kind it got.
    if callable(chosen):
        chosen = chosen()
    return chosen if chosen in methods else None


def report(store: ResultStore, tier: Tier, reference: str | None = None) -> str:
    """Read the store and compare every arm against one, paired across seeds.

    Regret is read straight from the records, where it is already against the
    attainable optimum -- see `evogfn.benchmark.suite._scores`. What this adds is
    the context that makes it readable: the interval the task can reach, how many
    of an arm's seeds are sitting on it, what each arm spent to get there, which
    rows are ablations rather than published pipelines, and a refusal to present a
    paired comparison drawn on a solved task as though it ranked anything.

    Args:
        store: Where results live.
        tier: The tier to report on.
        reference: Arm to compare against. Defaults to whatever
            `reference_for` picks for this tier -- `DEFAULT_REFERENCE`, a
            published pipeline, outside the diagnostics. ``None`` where the tier
            has no arm that can serve, which is reported rather than passed over.

    Returns:
        A multi-line report.
    """
    lines = []
    names = list(methods_for(tier))
    if reference is None:
        reference = reference_for(tier, methods_for(tier))
    for task in tier.tasks:
        attainable = attainable_for(task)
        lines.append(f"\n{task!r}")
        lines.append(_attainable_line(attainable))
        held = {name: store.usable(task.name, name) for name in names}
        seeds = [s for s in tier.seeds if all(s in held[n] for n in names if held[n])]
        rows, solved = _arm_rows(held, names, tier.seeds, attainable)
        lines.extend(rows)
        for name in sorted(solved):
            lines.append(
                f"  SOLVED  {name} sits on the attainable optimum "
                f"{attainable.lower:.4f} on {_at_optimum(held[name], attainable):.0%} of its "  # type: ignore[union-attr]
                f"seeds; this task cannot rank it against anything better, and a "
                f"comparison naming it is not a comparison of methods"
            )

        lines.extend(_paired(held, names, seeds, reference, solved))
    return "\n".join(lines)


def _arm_rows(
    held: Mapping[str, Mapping[int, RunRecord]],
    names: list[str],
    seeds: Sequence[int],
    attainable: AttainableOptimum | None,
) -> tuple[list[str], set[str]]:
    """One line per arm, and which arms turned out to be sitting on the ceiling.

    Split out of `report` rather than inlined: `report` was already at the branch
    limit, and the alternative to a helper is a table nobody may add a column to.

    The `proxy` column is the point of the split. Proxy spend is a budget someone
    chose -- ``steps x batch_size`` for the GFlowNet, ``generations x population``
    for the `genetic+search` ablation -- so an arm that wins on regret while
    spending an order of magnitude more surrogate evaluations has won on compute,
    and printing the two side by side is the only place a reader would see that.

    Args:
        held: Stored records by arm.
        names: The arms, in report order.
        seeds: The tier's seeds, fixing the order metrics are pulled in.
        attainable: What the task can reach, or ``None`` when unaudited.

    Returns:
        The report lines, and the arms whose share of seeds on the attainable
        optimum is at or above `VACUOUS_SHARE` -- returned rather than recomputed
        by the caller, since recomputing it is how the table and the comparison
        below it drift into disagreeing about which arms are solved.
    """
    lines, solved = [], set()
    for name in names:
        records = held[name]
        if not records:
            continue
        regret = records_to_metric(records, seeds, "regret")
        feasible = records_to_metric(records, seeds, "feasible_fraction")
        spread = records_to_metric(records, seeds, "diversity")
        spent = records_to_metric(records, seeds, "oracle_calls")
        proxy = records_to_metric(records, seeds, "proxy_calls")
        # Every mean on this row is taken over completed campaigns only --
        # `records_to_metric` drops the rest -- so the count of seeds that failed
        # has to appear beside them or the row reports a subset while `n` reads
        # like a full one. An arm that exhausted on *every* seed is the case this
        # is really for: it used to be absent from the table altogether, and an
        # empty cell is an absence a reader fills in as they please. It gets a
        # line of its own, because there is no mean to print on it and a row of
        # `nan`s would read as a broken table rather than as a result.
        exhausted = sum(1 for seed in seeds if (r := records.get(seed)) is not None and r.exhausted)
        if not len(regret):
            lines.append(
                f"  {name:<18} exhausted on all {exhausted} of its stored seeds; it could not "
                f"propose designs it had not already measured, so there is no number here"
            )
            continue
        failed = f"  exhausted={exhausted}" if exhausted else ""
        error = regret.std(ddof=1) / len(regret) ** 0.5 if len(regret) > 1 else 0.0
        share = _at_optimum(records, attainable)
        if share >= VACUOUS_SHARE:
            solved.add(name)
        # On the row itself, not in a footnote: a decomposition row sitting
        # unmarked among published pipelines is read as one of them.
        mark = f"  [ablation of {ABLATIONS[name]}]" if name in ABLATIONS else ""
        lines.append(
            f"  {name:<18} regret {regret.mean():>7.3f} +/- {error:<6.3f} "
            f"at-opt {share:>5.2f}  feas {feasible.mean():>5.3f}  "
            f"div {spread.mean():>5.2f}  spent {spent.mean():>6.0f}  "
            f"proxy {proxy.mean():>7.0f}  n={len(regret)}{failed}{mark}"
        )
    return lines, solved


def _paired(
    held: Mapping[str, Mapping[int, RunRecord]],
    names: list[str],
    seeds: list[int],
    reference: str | None,
    solved: set[str],
) -> list[str]:
    """Compare every arm against the reference, paired across shared seeds.

    Args:
        held: Stored records by arm.
        names: The arms, in report order.
        seeds: Seeds every arm has, so the comparison is genuinely paired.
        reference: The arm to compare against, or ``None`` if the tier has none.
        solved: Arms already sitting on the attainable optimum.

    Returns:
        Report lines, including a line naming the omission when there is
        nothing to compare against -- an absent section reads as "nothing
        separated these arms", which is a different statement from "nothing
        was tested" -- and a line marking every ablation, whose comparison
        against a published pipeline is an attribution and not a ranking.
    """
    if reference is None:
        return ["  no reference arm in this tier, so nothing is paired"]
    base = held.get(reference)
    if not base or not seeds:
        return [f"  reference {reference} has no usable seeds here, so nothing is paired"]
    lines = [f"  paired vs {reference} (positive favours the first):"]
    for name in names:
        if name == reference or not held[name]:
            continue
        mine = records_to_metric(held[name], seeds, "regret")
        theirs = records_to_metric(base, seeds, "regret")
        if len(mine) != len(theirs) or len(mine) < 2:  # noqa: PLR2004
            # Said rather than skipped. The lengths differ when one of the two
            # arms exhausted on a seed the other completed, and a pairing that
            # dropped it from one side only would compare two different seed
            # sets; an omitted line, on the other hand, reads as "these did not
            # separate", which is the opposite of what happened.
            lines.append(
                f"    {name} vs {reference}: not paired -- {len(mine)} and {len(theirs)} "
                f"completed campaigns over {len(seeds)} shared seeds"
            )
            continue
        outcome = compare(name, mine, theirs, higher_is_better=False)
        lines.append(f"    {outcome!r}")
        # Marked here as well as in the table above: read on its own, an
        # ablation's line is a p-value against a published pipeline, and that is
        # a claim about methods rather than the attribution claim it supports.
        if decomposes := ABLATIONS.get(name):
            lines.append(
                f"        attribution: decomposes {decomposes}; separates the "
                f"surrogate's contribution from the sampler's, and ranks no "
                f"method a lab could run"
            )
        # Said before the p-value is read rather than after: an arm on the
        # ceiling makes the difference a measurement of the ceiling, and
        # significance there is significance about the task.
        if vacuous := solved.intersection({name, reference}):
            lines.append(
                f"        vacuous: {', '.join(sorted(vacuous))} already reached "
                f"everything this task was audited to contain"
            )
        elif not outcome.significant and (needed := seeds_needed(outcome)):
            lines.append(f"        inconclusive; ~{needed} seeds would resolve this")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run or report on the suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", action="append", help="Run only these tiers.")
    parser.add_argument(
        "--task",
        action="append",
        help="Run only these tasks. Sharding by task is race-free -- the store "
        "keeps one file per task and method -- so a process per task uses the "
        "cores far better than threads do, most of the work being serial Python.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Run only these arms. The classical baselines do not depend on which "
        "GFlowNet configuration the selection phase picks, so they can be banked "
        "while it is still running.",
    )
    parser.add_argument("--seeds", type=int, default=MAIN_SEEDS, help="Seeds for main tiers.")
    parser.add_argument(
        "--seed-from",
        type=int,
        default=0,
        help="First seed this process runs. Sharding by task alone leaves a tier "
        "with few tasks unable to fill the cores, and one arm's hundred seeds "
        "then run serially while the rest of the machine idles. Every campaign "
        "is seeded from its own seed rather than from process order, so a "
        "sharded run and a serial one produce identical records.",
    )
    parser.add_argument(
        "--seed-to",
        type=int,
        default=None,
        help="One past the last seed this process runs. Defaults to the tier's own count.",
    )
    parser.add_argument(
        "--diagnostic-seeds", type=int, default=DIAGNOSTIC_SEEDS, help="Seeds for diagnostics."
    )
    parser.add_argument("--results", default="results", help="Where to store results.")
    parser.add_argument("--report", action="store_true", help="Report without running.")
    args = parser.parse_args(argv)

    # Before any tensor work: a multithreaded matmul sums in thread-completion
    # order, and a few hundred gradient steps turn that into a different design.
    configure_determinism()
    if not args.report and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    store = ResultStore(Path(args.results))
    selected = tiers(args.seeds, args.diagnostic_seeds)
    if args.tier:
        selected = [t for t in selected if t.name in set(args.tier)]
    wanted_methods = set(args.method) if args.method else set()
    if args.task:
        wanted = set(args.task)
        selected = [
            Tier(
                t.name,
                tuple(task for task in t.tasks if task.name in wanted),
                t.seeds,
                t.purpose,
            )
            for t in selected
        ]
        selected = [t for t in selected if t.tasks]
    if not selected:
        print(f"nothing matched tier={args.tier} task={args.task}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    for tier in selected:
        if not args.report:
            arms = methods_for(tier)
            if args.method:
                arms = {k: v for k, v in arms.items() if k in wanted_methods}
                if not arms:
                    _flush(f"{tier.name}: no arm matched --method, skipping")
                    continue
            # Sliced for running only. The report below reads the tier's full
            # seed set, so a shard says what the store holds rather than what
            # this process happened to be given -- a shard reporting its own
            # slice as the tier would read as a complete tier at 33 seeds.
            stop = len(tier.seeds) if args.seed_to is None else args.seed_to
            mine = tier.seeds[args.seed_from : stop]
            if not mine:
                _flush(f"{tier.name}: no seed in [{args.seed_from}, {stop}), skipping")
                continue
            running = Tier(tier.name, tier.tasks, mine, tier.purpose)
            ran = _run(running, arms, store, by_method=bool(args.method))
            if ran is None:
                continue
            _flush(f"{tier.name}: ran {ran} campaigns")
        _flush(report(store, tier))
    _flush(f"\ntotal {time.perf_counter() - started:.0f}s")
    _flush(store.summarise())
    return 0


def _run(
    running: Tier, arms: dict[str, object], store: ResultStore, *, by_method: bool
) -> int | None:
    """Run one tier's outstanding campaigns, however that tier is shaped.

    A tier is normally the cross of its tasks with its arms, and `run_tier` runs
    that cross. The anchor study is the exception, and is the reason this
    function exists rather than a call: its five cells are a *subset* of the
    cross, so running the cross would add a rebuilt policy on the task whose
    anchor never moves -- which, since nothing is ever rebuilt there, is the
    carried arm's campaign stored a second time under another name.

    Args:
        running: The tier, already sliced to the seeds this process owns.
        arms: What to run, for the tiers that are a cross.
        store: Where results go.
        by_method: Whether the caller passed ``--method``.

    Returns:
        Campaigns run, or ``None`` where the tier was skipped and has already
        said so.
    """
    if running.name != ANCHOR_TIER:
        return run_tier(running, arms, store, report=_flush)  # type: ignore[arg-type]
    if by_method:
        # Refused rather than applied. The study is a list of ``(task, arm)``
        # cells and not an arm set, so an arm filter would drop whole cells
        # silently -- and the cells it would drop are the ones whose absence
        # turns "carrying the policy helps" into a column with nothing paired
        # against it. Run this tier without ``--method``.
        _flush(f"{running.name}: runs named cells, so --method cannot select within it")
        return None
    return run_anchor_study(store, running.seeds, report=_flush)


def _flush(message: str) -> None:
    """Print immediately, so an overnight run can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
