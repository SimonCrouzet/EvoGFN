"""Run the benchmark suite, resuming whatever is already stored.

Safe to interrupt and safe to re-run. Every campaign is written the moment it
finishes, and a second invocation runs only what is missing -- so raising a
tier's seed count from 30 to 50 costs twenty campaigns per arm, not fifty.

    uv run python experiments/run_suite.py                  # everything
    uv run python experiments/run_suite.py --tier main      # headline only
    uv run python experiments/run_suite.py --seeds 50       # raise the count
    uv run python experiments/run_suite.py --report         # no runs, just read
    uv run python experiments/run_suite.py --promote ARM    # ship a ladder rung

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
``generations x population``. That closed form describes an arm which only samples
from its own policy, and it is **wrong for an arm that breeds**: a genetic
teacher's proposals are themselves proxy evaluations, which the formula does not
count -- ``gfn-subtb@b0.1-s300-l0.9-h64`` records exactly ``3 x 300 x 64 =
57,600`` while ``genetic-gfn@b0.5-s300-m0.25-h64`` at the same steps, width and
round count records 70,350, about 22% more. So the column is **read off the
record** rather than derived from a configuration file; reconstructing it from
the settings gets those arms wrong by more than the arms differ.
Let those budgets differ by an order of magnitude and the
regret column is measuring compute rather than method. So the cost is printed
next to the win: a GFlowNet needing 10x the proxy calls is a real and publishable
cost of the method, and the column is where a reader finds it rather than
something to be inferred from a configuration file. One caveat, from
[evogfn.benchmark.methods][]: a sampler rebuilt at an anchor move restarts its own
accounting, so on a re-anchoring task the stored count covers the last anchor's
rounds rather than the campaign's, and reads as a floor.

Why a supervised row can say it never fitted
--------------------------------------------

A method that trains on its own measurements is only that method once the
training has happened. MLDE screens at random until it holds enough usable
assays, and
[MLDE.observe][evogfn.algorithms.baselines.mlde.MLDE.observe] discards an
infeasible one -- there is no fitness to regress on -- so on a landscape with a
transition constraint the screening plates buy far fewer training examples than
they cost, and where the infeasible share is large the handover never happens at
all. Such a campaign spends its whole budget, fills every plate and reports a
regret that is arithmetically indistinguishable from a fitted one's.

So the report says it outright. `RunRecord.fitted` is stored per seed, the arm's
row carries `unfit=n`, an arm that never fitted gets a `NEVER FITTED` line of its
own, and every paired comparison naming one is marked as a difference against a
random screen rather than against a supervised baseline. Reading such a row as a
tuned method that lost fairly is the one conclusion the column exists to prevent.

Why the replicated tasks are read one instance at a time, and refuse otherwise
-------------------------------------------------------------------------------

The replication tier runs the same protocol shape on several draws from the
instance generator. Its per-task tables are already per instance; what a reader
does with them is the risk. Averaging six tasks x 100 seeds into one figure of
600 does not look like a mistake -- it looks like *more* evidence, at a standard
error several times narrower than the design supports, because instance x seed
pairs are not independent observations. A seed varies the wild type and the
surrogate's initialisation *within* a draw; nothing in that pooled array varies
the draw.

So the report adds a section that takes the **draw** as the unit -- each arm's
per-draw effect, the interval across draws at ``n = draws``, and a sign test on
how many draws agreed -- and `pooled_metric` **raises** on any task set holding
two draws of one shape rather than being merely absent. Absent is a convention,
and a convention is what the next person needing a single number breaks. The
measured factor by which pooling would have understated the error is printed on
every arm's line, so the refusal comes with its own arithmetic.

How many draws that section should have is not settled here. It is measured by
``experiments/variance_pilot.py``, whose rule is fixed before its numbers; what
this report can already say without any measurement is that three draws cannot
reach significance under a sign test at any seed count, and it says so.

How a ladder rung reaches the headline table
---------------------------------------------

`variant-ladder` crosses two GFlowNet mechanisms on the shipped configuration and
decides which to ship. It runs on the diagnostic landscape under
`Purpose.SELECTION`, and its rungs reach `main` only through a **separate,
explicit step**: ``--promote``, which reads the ladder's own stored campaigns,
prints every rung against the base, refuses a rung the ladder did not support,
and writes `PROMOTION_FILE`. `methods_for` reads that file and nothing else.

The ordering is the whole point. If the rungs sat in `main` and the best were
picked afterwards, the configuration would have been chosen on the tasks that
carry the claims -- tuning on the test set, which is what the selection phase
exists to prevent. Once promoted, the rungs that lost appear as decomposition
rows exactly where `genetic+search` does, labelled `[ablation of ...]` on the row
and carrying an attribution line beside every p-value, so a reader cannot take
one for a method a lab would run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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
from evogfn.benchmark.statistics import (
    compare,
    decompose,
    seeds_needed,
    unanimity_floor,
    unanimity_p,
)
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import (
    MAIN,
    Purpose,
    Tier,
    anchor_study,
    budget_gradient,
    constraint_density_tier,
    fit_status,
    objective_task,
    records_to_metric,
    replicate_instance,
    replication,
    rounds_curve,
    run_anchor_study,
    run_tier,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from evogfn.benchmark.attainable import AttainableOptimum
    from evogfn.benchmark.statistics import PairedComparison
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

#: The GFlowNet mechanism ladder, named for the same reason `ANCHOR_TIER` is: the
#: promotion step below reads *this tier and no other*, which is what stops a
#: configuration being chosen on a task that carries claims. A literal misspelt
#: at either end would silently promote against an empty record set.
LADDER_TIER = "variant-ladder"

#: The tier whose tasks differ from one another only in the landscape draw.
#: Everything in the instance-analysis path keys off this name, and the tier's
#: tasks are matched through
#: [replicate_instance][evogfn.benchmark.suite.replicate_instance] rather than by
#: a pattern written here.
REPLICATION_TIER = "replication"

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
#:
#: The MLDE pair is here for the same reason the sweep exists at all. A dense
#: constraint starves a supervised arm's training set -- an infeasible assay
#: carries no fitness to regress on -- so `mlde` can spend its whole budget
#: screening at random and still be tabled under a supervised method's name, and
#: `mlde+earlyfit` exists to say whether the method or the fit was the problem.
#: That pair is currently a single observation on the one headline task that
#: constrains feasibility, with nothing to read it against; density is the axis it
#: is a function of, and this tier is the axis. Both, never one: `mlde+earlyfit`
#: alone would be a control on a tier that does not hold the thing it controls,
#: and the row a reader would then quote is the adapted arm's -- which must never
#: be read as MLDE. `mlde-over-budget` stays out, its extra plate being a second
#: axis this tier does not vary.
DENSITY_ARMS = (
    "genetic",
    "genetic-feasible",
    "gfn-tb",
    "genetic-gfn",
    "mlde",
    "mlde+earlyfit",
)

#: Tiers whose **axis is the budget**, and from which `OVER_BUDGET_ARMS` are
#: therefore removed.
#:
#: ``rounds-curve`` splits one fixed total across different numbers of rounds;
#: ``budget-gradient`` sweeps the total from a wet-lab plate to the
#: machine-learning convention. Both read a difference between their tasks as
#: the effect of the budget, which requires every arm to have spent the same one.
#: An arm that adds a plate does not: on ``rounds-curve`` it breaks the fixed
#: total the whole curve is defined by, and on ``budget-gradient`` it sits one
#: plate to the right of the point it is plotted at -- which at the ``8x12``
#: rung is a 96-assay arm on a 96-assay task, a doubling, and at the ``10x1000``
#: rung is a 1% perturbation. So the same arm would be a different distortion at
#: every rung, in the direction that flatters it, and the curve would read as a
#: property of the methods.
#:
#: It stays in ``main``, ``replication`` and ``large-space``, where the budget is
#: held fixed and the extra plate is the point: those tiers report it *because*
#: it does not fit, so that the headline does not rest on a comparator we had
#: compressed to a quarter of its published training set.
BUDGET_AXIS_TIERS = ("rounds-curve", "budget-gradient")

#: Arms that deliberately spend more than their task's protocol allots. Named
#: rather than detected: ``extra_rounds`` is a parameter of the arm's own
#: closure in [evogfn.benchmark.methods][] and reading it back out here would be
#: reaching into another module's internals to re-derive a fact that module
#: already states.
OVER_BUDGET_ARMS = ("mlde-over-budget",)


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


#: Where the promotion step records which rung of the variant ladder ships.
#:
#: A **separate file from** `CHOICE_FILE`, and a separate step, because the two
#: decisions are made at different times against different evidence: the
#: selection chooses hyperparameters from its own screen, and the promotion
#: chooses a mechanism from a ladder that stands on whatever the selection
#: chose. Writing both into one file would make the second look like a field of
#: the first, which is exactly the reading that turns the promotion into an
#: automatic consequence of the ladder rather than a decision somebody took.
PROMOTION_FILE = Path("results/promoted.json")

#: Axes a promotion record must carry. Absent is not the same as null here for
#: the reason it is not in `CHOICE_FILE`: a file written by a promotion that
#: stopped partway describes a configuration no rule ever chose.
_PROMOTION_AXES = ("arm", "base", "tier", "task")


def promoted_rung() -> str | None:
    """The ladder rung the promotion step chose, if it has run.

    Returns:
        The rung's arm name, or ``None`` when nothing has been promoted -- in
        which case the headline tiers run exactly what they ran before the
        ladder existed, which is the state the ladder must not be able to change
        on its own.

    Raises:
        ValueError: If the record is unfinished, or names a rung the ladder no
            longer builds, or was taken against a base the selection has since
            moved off. All three describe an arm nobody runs, and falling back
            to the untuned path instead would put a configuration in the
            headline table that no promotion ever chose.
    """
    if not PROMOTION_FILE.exists():
        return None
    record = json.loads(PROMOTION_FILE.read_text())
    if missing := [key for key in _PROMOTION_AXES if key not in record]:
        raise ValueError(
            f"{PROMOTION_FILE} is missing {', '.join(missing)}, so the promotion it "
            f"records is unfinished; re-run experiments/run_suite.py --promote, or "
            f"delete the file to leave the ladder unpromoted"
        )
    arm, base = str(record["arm"]), str(record["base"])
    rungs = variant_arms()
    if arm not in rungs:
        raise ValueError(
            f"{PROMOTION_FILE} promotes {arm!r}, which is not a rung of the current "
            f"ladder ({', '.join(rungs)}); the ladder has moved under the promotion, "
            f"so the recorded decision is about an arm nothing runs"
        )
    if base != shipped_base().name:
        raise ValueError(
            f"{PROMOTION_FILE} promotes a rung measured against base {base!r}, but the "
            f"ladder now stands on {shipped_base().name!r}; the promotion rests on a "
            f"comparison against a configuration this project no longer ships"
        )
    return arm


def promoted_arms() -> dict[str, object]:
    """The ladder's arms, once a promotion has put them in the headline path.

    Empty until the promotion step runs, and that emptiness is the whole design.
    The rungs are decided on the diagnostic landscape and *then* promoted; if
    they simply appeared in `main` and the best were picked afterwards, the
    configuration would have been chosen on the tasks that carry the claims --
    tuning on the test set, which is the failure the entire selection phase
    exists to prevent. Making promotion a file somebody writes, rather than a
    consequence of the tier existing, is what keeps the two in that order.

    Returns:
        Every rung by name, or an empty mapping when nothing is promoted.
    """
    return dict(variant_arms()) if promoted_rung() is not None else {}


def promoted_ablations() -> dict[str, str]:
    """The rungs that are decomposition rows, mapped to the rung that ships.

    Every rung except the promoted one. They are exactly what `genetic+search`
    is to `genetic`: real campaigns, honestly run, that answer *which part of
    the method does the work* rather than *which method a lab should run* -- and
    an unmarked one sitting among published pipelines is read as a pipeline.
    ``+wide`` is in here too, and it is the one that would be misread hardest:
    it is a capacity control whose only purpose is to make ``+anchor``
    attributable, and a reader taking it for a method would take a deliberately
    over-sized policy for something anyone proposed.

    Returns:
        Arm name to the promoted arm it decomposes, or an empty mapping.
    """
    promoted = promoted_rung()
    if promoted is None:
        return {}
    return {name: promoted for name in variant_arms() if name != promoted}


def promote(store: ResultStore, rung: str, *, report: Callable[[str], None] = print) -> int:
    """Record which rung of the ladder ships, after the ladder has resolved.

    The separate, explicit step. It reads **only** the ladder tier's own task --
    the diagnostic landscape no headline task uses -- so a promotion cannot be
    argued from a benchmark result even by a caller who wanted to. It runs
    nothing: every comparison it prints comes from campaigns the ladder already
    banked, so the decision is taken on evidence that existed before anyone
    named a winner.

    The rung is **named by the caller and never derived**. Deriving the best
    rung here would make promotion an automatic consequence of running the
    ladder, which is the thing this whole path exists to be instead of. What is
    checked is that the named rung is one the evidence permits:

    * a rung other than the base must have beaten the base, paired across shared
      seeds, with the interval excluding zero;
    * the base itself may be promoted -- "neither mechanism earns its compute" is
      a real outcome of a ladder -- but only when no other rung beat it, since a
      decision contradicting the tier it was taken from is not a decision.

    Args:
        store: Where the ladder's campaigns live.
        rung: The arm name to promote.
        report: Where the evidence goes.

    Returns:
        A process exit code: zero when the promotion was recorded, non-zero when
        it was refused. Refused rather than warned about, because a promotion is
        read back by `methods_for` and would otherwise fix the headline table's
        configuration on evidence nobody checked.
    """
    rungs = variant_arms()
    base = shipped_base().name
    if rung not in rungs:
        report(f"  {rung!r} is not a rung of the ladder ({', '.join(rungs)})")
        return 2
    task = objective_task()
    held = {name: store.usable(task.name, name) for name in rungs}
    seeds = sorted(set.intersection(*(set(records) for records in held.values())) or set())
    if len(seeds) < 2:  # noqa: PLR2004 - a paired comparison needs two seeds
        report(
            f"  the ladder holds {len(seeds)} seed(s) every rung completed on {task.name}; "
            f"there is nothing to promote from"
        )
        return 4

    outcomes = {
        name: compare(
            name,
            records_to_metric(held[name], seeds, "regret"),
            records_to_metric(held[base], seeds, "regret"),
            higher_is_better=False,
        )
        for name in rungs
        if name != base
    }
    report(f"  ladder on {task.name}, {len(seeds)} shared seeds, paired against {base}:")
    for outcome in outcomes.values():
        report(f"    {outcome!r}")

    beat_the_base = sorted(
        name for name, outcome in outcomes.items() if outcome.significant and outcome.mean > 0.0
    )
    if rung == base and beat_the_base:
        report(
            f"  REFUSED: promoting the base {base!r} would ship the configuration that "
            f"{', '.join(beat_the_base)} beat on this ladder; a promotion contradicting "
            f"the tier it was taken from is not a decision"
        )
        return 5
    if rung != base and rung not in beat_the_base:
        report(
            f"  REFUSED: {rung!r} did not beat {base!r} with an interval excluding zero "
            f"({outcomes[rung]!r}); promoting it would put a rung in the headline table "
            f"on evidence the ladder did not produce"
        )
        return 5

    PROMOTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROMOTION_FILE.write_text(
        json.dumps(
            {
                "arm": rung,
                "base": base,
                "tier": LADDER_TIER,
                "task": task.name,
                "seeds": len(seeds),
                # The comparisons the decision rests on, stored beside it. A
                # promotion whose evidence lives only in a terminal scrollback
                # is one nobody can audit later, and this file is what
                # `methods_for` reads to decide what the headline table runs.
                "evidence": [repr(outcome) for outcome in outcomes.values()],
                "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
        + "\n"
    )
    others = ", ".join(sorted(promoted_ablations()))
    report(
        f"  promoted {rung!r} into the headline path; {others} now appear there as "
        f"decomposition rows, labelled on every line as ablations of {rung!r}"
    )
    return 0


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
        #
        # Above the promotion lookup as well as above the selection one, and for
        # the mirror of the same reason: the ladder is where a rung is *decided*,
        # so it must go on running all five arms and comparing them against the
        # base whatever has since been promoted. A promoted ladder that had
        # dropped its losing rungs here would leave the decision unauditable the
        # moment it was taken.
        LADDER_TIER: lambda: dict(variant_arms()),
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
    # The selected arm replaces the untuned GFlowNet arms rather than joining
    # them: keeping both would put two configurations of the same method in one
    # table, and the better of the two would be the one the selection was run to
    # avoid reporting.
    arms: dict[str, object] = dict(MAIN_METHODS) if not chosen else {**BASELINES, **chosen}
    # The promoted ladder replaces the single selected arm with all five rungs:
    # the winner as the method, the rest as decomposition rows. It sits on the
    # default path rather than in a tier list of its own, so the promoted rungs
    # appear wherever `genetic+search` already does -- there is then no second
    # rule about where a decomposition row lives, and the shipped configuration
    # cannot differ between two tiers that both claim to report it.
    #
    # It is not free: five GFlowNet arms in place of one is four extra campaigns
    # per task per seed, which at the stored ~255s per campaign is hundreds of
    # core-hours across the default-path tiers. That cost is paid only by
    # somebody who both promotes and then runs; until then the rows report as
    # unrun, which is the visible state rather than a surprise.
    if promoted := promoted_arms():
        arms = {**BASELINES, **promoted}
    if tier.name in BUDGET_AXIS_TIERS:
        # `mlde-over-budget` runs one plate beyond its task's protocol, which is
        # the point of the arm and is exactly what a tier whose axis *is* the
        # budget cannot hold constant. See `BUDGET_AXIS_TIERS`.
        for name in OVER_BUDGET_ARMS:
            arms.pop(name, None)
    return arms


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
    LADDER_TIER: lambda: shipped_base().name,
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
#:
#: **Every** rung of the ``genetic+`` ladder, not the top one alone. A ladder
#: marked only at its last rung is the worst of both readings: `genetic+screen`
#: and `genetic+distinct` then sit unmarked in a table whose other decomposition
#: rows *are* marked, so the absence of a mark reads as a positive claim that they
#: are pipelines somebody published -- and neither is. ``+screen`` is a genetic
#: algorithm handed a deep ensemble, which appears in no paper this suite is read
#: against, and ``+distinct`` is a plate rule of the harness's rather than of any
#: method. `mlde+earlyfit` carries a ``+`` too and is deliberately **not** here:
#: it replaces a parameter of a published pipeline rather than adding a rung to
#: one, so it decomposes nothing, and the attribution sentence beside these rows
#: would be false of it. What it needs is its own scope note, which it does not
#: yet have.
ABLATIONS = {
    "genetic+screen": "genetic",
    "genetic+search": "genetic",
    "genetic+distinct": "genetic",
    "random+screen": "random",
}

#: What each family of decomposition row separates, keyed by the pipeline it
#: decomposes. Two families, because they answer different questions and a row
#: carrying the wrong one is a row explaining itself wrongly: the classical
#: ablations split a surrogate's contribution from a sampler's, while a promoted
#: ladder rung splits one GFlowNet mechanism from another.
_ATTRIBUTIONS = {
    "surrogate": (
        "separates the surrogate's contribution from the sampler's, and ranks "
        "no method a lab could run"
    ),
    "mechanism": (
        "separates one mechanism of the shipped configuration from the rest; it "
        "was decided on the diagnostic landscape and is a decomposition of the "
        "promoted arm, not a method a lab would run"
    ),
}


def ablations() -> dict[str, str]:
    """Every decomposition row in the table, mapped to what it decomposes.

    The static pair plus whatever the promotion added. Resolved through a
    function rather than left as a module constant so that the labelling cannot
    be forgotten for the promoted rungs: `_arm_rows` and `_paired` both call
    this, so a rung that reaches the table reaches it marked. A mapping extended
    at promotion time by whoever remembered to would be a convention, and this
    is the one place a convention fails -- an unmarked decomposition row among
    published pipelines reads as a pipeline that lost.

    Returns:
        Arm name to the pipeline or promoted arm it decomposes.
    """
    return {**ABLATIONS, **promoted_ablations()}


def _attribution(name: str) -> str:
    """Which question a decomposition row answers, in the row's own terms."""
    kind = "mechanism" if name in promoted_ablations() else "surrogate"
    return _ATTRIBUTIONS[kind]


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
    rows are ablations rather than published pipelines, **which supervised rows
    never got as far as fitting a model**, and a refusal to present a paired
    comparison drawn on a solved task as though it ranked anything.

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
        rows, solved, unfitted = _arm_rows(held, names, tier.seeds, attainable)
        lines.extend(rows)
        for name in sorted(solved):
            lines.append(
                f"  SOLVED  {name} sits on the attainable optimum "
                f"{attainable.lower:.4f} on {_at_optimum(held[name], attainable):.0%} of its "  # type: ignore[union-attr]
                f"seeds; this task cannot rank it against anything better, and a "
                f"comparison naming it is not a comparison of methods"
            )
        # Printed as its own line rather than as a column, for the reason the
        # exhausted-on-every-seed line is: this is not a worse number, it is a
        # statement that the row is not the method its name claims.
        for name in sorted(unfitted):
            reported, never = fit_status(held[name], tier.seeds)
            lines.append(
                f"  NEVER FITTED  {name} finished {never} of its {reported} stored campaigns "
                f"without ever fitting its model, so on those seeds it screened at random for "
                f"the whole budget; this row is a random baseline under a supervised method's "
                f"name and must not be read as a tuned baseline that lost"
            )

        lines.extend(_paired(held, names, seeds, reference, solved=solved, unfitted=unfitted))
    # After every per-task table, never instead of them. The tables above are
    # already per instance; what this adds is the only reading of them that
    # treats the *instance* as the unit of replication, and a line saying that
    # the pooled alternative is refused rather than merely not printed.
    lines.extend(_instance_section(store, tier, names, reference))
    return "\n".join(lines)


def instance_families(tasks: Sequence[Task]) -> dict[str, list[Task]]:
    """Group replicate tasks by the thing they are replicates *of*.

    Args:
        tasks: A tier's tasks.

    Returns:
        Protocol shape to its landscape draws, in task order, keeping only
        shapes with more than one draw. A shape with one draw is not replicated,
        and reporting it here would put a single instance under a heading that
        claims instances were varied.
    """
    families: dict[str, list[Task]] = {}
    for task in tasks:
        split = replicate_instance(task.name)
        if split is not None:
            families.setdefault(split[0], []).append(task)
    return {shape: drawn for shape, drawn in families.items() if len(drawn) > 1}


@dataclass(frozen=True, slots=True)
class InstanceEffects:
    """One arm against the reference, with the landscape draw as the unit.

    Attributes:
        arm: The arm under test.
        reference: What it was paired against, within each draw.
        shape: The protocol shape whose draws these are.
        per_draw: One paired comparison per draw, each taken on the seeds that
            draw holds. Paired within the instance and never across it: two
            campaigns on different draws share no wild type, no surrogate
            initialisation and no optimum, so a difference between them is the
            between-instance term wearing a paired comparison's clothes.
        differences: The per-seed paired differences those comparisons were
            taken from, one array per draw. Kept because the variance
            decomposition needs the raw values and recovering them from a
            summary is how a printed factor comes to describe a different
            calculation from the one above it.
        across: The per-draw mean effects treated as the observations they are,
            so ``n`` is the number of **draws**. This is the whole of the
            difference between the analysis the framing commits to and the one
            it calls pseudo-replication.
    """

    arm: str
    reference: str
    shape: str
    per_draw: tuple[PairedComparison, ...]
    differences: tuple[np.ndarray, ...]
    across: PairedComparison

    @property
    def agreeing(self) -> int:
        """Draws on which the effect took the majority direction."""
        return max(self.across.wins, self.across.n - self.across.wins)

    @property
    def sign_test(self) -> float:
        """Two-sided sign-test p for that agreement, with draws as the unit.

        The one across-instance statistic that assumes nothing about how the
        effect is distributed over draws, which at three or four draws is the
        only honest thing to quote: the ``t`` interval beside it rests on a
        normality assumption three numbers cannot support.
        """
        return unanimity_p(self.across.n, self.agreeing)

    def __repr__(self) -> str:
        """The across-draw effect, how many draws agreed, and what that is worth."""
        return (
            f"{self.arm} vs {self.reference} across {self.across.n} draws of "
            f"{self.shape}: {self.across.mean:+.3f} "
            f"[{self.across.low:+.3f}, {self.across.high:+.3f}]  "
            f"same direction on {self.agreeing}/{self.across.n} "
            f"(sign test p={self.sign_test:.3f})"
        )


def instance_effects(  # noqa: PLR0913, PLR0917 - a per-instance effect names both arms and every draw
    store: ResultStore,
    tasks: Sequence[Task],
    shape: str,
    arm: str,
    reference: str,
    seeds: Sequence[int],
) -> InstanceEffects | None:
    """One arm's effect, per landscape draw and then across them.

    Args:
        store: Where records live.
        tasks: The draws of one protocol shape.
        shape: That shape's name, for the printed line.
        arm: The arm under test.
        reference: What it is paired against within each draw.
        seeds: Seeds to read.

    Returns:
        The effects, or ``None`` when fewer than two draws hold a pairable
        campaign -- with one draw there is no across-instance analysis to do,
        and returning something that looked like one would be the pooled reading
        under another name.
    """
    per_draw, differences = [], []
    for task in tasks:
        mine = store.usable(task.name, arm)
        theirs = store.usable(task.name, reference)
        shared = [seed for seed in seeds if seed in mine and seed in theirs]
        first = records_to_metric(mine, shared, "regret")
        second = records_to_metric(theirs, shared, "regret")
        if first.size != second.size or first.size < 2:  # noqa: PLR2004
            continue
        difference = second - first
        if not np.isfinite(difference).all():
            continue
        per_draw.append(compare(task.name, first, second, higher_is_better=False))
        differences.append(difference)
    if len(per_draw) < 2:  # noqa: PLR2004 - two draws are the fewest that vary the instance
        return None
    means = np.asarray([outcome.mean for outcome in per_draw], dtype=np.float64)
    return InstanceEffects(
        arm=arm,
        reference=reference,
        shape=shape,
        per_draw=tuple(per_draw),
        differences=tuple(differences),
        # Against a vector of zeros, which is a one-sample interval on the draw
        # means built out of the module's own paired machinery rather than a
        # second statistic that could disagree with it.
        across=compare(f"{arm} across draws", means, np.zeros_like(means)),
    )


def pooled_metric(
    store: ResultStore,
    tasks: Sequence[Task],
    arm: str,
    seeds: Sequence[int],
    metric: str = "regret",
) -> np.ndarray:
    """One arm's per-seed metric across several tasks -- and a refusal for replicates.

    This function exists to be the door somebody walks through when they need a
    single number for the replication tier, and to shut in that one case. The
    convention "analyse per instance" is broken by whoever next wants one figure
    for a slide, and pooling six tasks x 100 seeds into a mean of 600 does not
    look like a mistake from the outside -- it looks like *more* evidence, with a
    standard error several times narrower than the design supports. That is why
    the refusal is here rather than in a comment: an absent function would simply
    be re-written as ``np.concatenate`` at the call site.

    Args:
        store: Where records live.
        tasks: The tasks to pull from.
        arm: The arm to pull.
        seeds: Seeds to read.
        metric: The record field.

    Returns:
        The concatenated per-seed values, for tasks that are genuinely different
        experiments.

    Raises:
        ValueError: If two of the tasks are landscape draws of the same protocol
            shape. Their seeds are not exchangeable -- a seed varies the wild
            type and the surrogate's initialisation *within* a draw, and nothing
            in the pooled array varies the draw -- so instance x seed pairs are
            not independent observations and a mean over them reports a
            precision the tier does not have. `instance_effects` is what to use
            instead.
    """
    families = instance_families(tasks)
    if families:
        offenders = {shape: [task.name for task in drawn] for shape, drawn in families.items()}
        raise ValueError(
            f"refusing to pool across landscape draws of the same protocol shape: "
            f"{offenders}. Instance x seed pairs are not independent observations, so a "
            f"mean over them understates its own standard error and reads as more "
            f"evidence rather than less; use instance_effects(), which takes the draw as "
            f"the unit of replication"
        )
    return np.concatenate(
        [records_to_metric(store.usable(task.name, arm), seeds, metric) for task in tasks]
    )


def _instance_section(
    store: ResultStore,
    tier: Tier,
    names: Sequence[str],
    reference: str | None,
) -> list[str]:
    """The per-instance analysis, and the refusal of the pooled one.

    Printed for any tier that holds more than one draw of the same protocol
    shape, keyed off the task names rather than off the tier's, so a tier
    assembled elsewhere out of replicate tasks gets the same treatment -- the
    error is a property of the data, not of what the tier was called.

    Args:
        store: Where records live.
        tier: The tier being reported on.
        names: Its arms, in report order.
        reference: What they are compared against.

    Returns:
        Report lines, or none when the tier replicates nothing.
    """
    families = instance_families(tier.tasks)
    if not families or reference is None:
        return []
    floor = unanimity_floor()
    lines = ["\n  --- per instance, with the landscape draw as the unit ---"]
    for shape, drawn in sorted(families.items()):
        lines.append(f"  {shape}: {len(drawn)} draws")
        if len(drawn) < floor:
            # Said once per shape and before any number, because it is not a
            # caveat on the numbers below -- it is a statement that the strongest
            # verdict they can produce is not evidence.
            lines.append(
                f"    UNDERPOWERED BY DESIGN  {len(drawn)} draws; a unanimous verdict over "
                f"{len(drawn)} reaches only sign-test p={unanimity_p(len(drawn), len(drawn)):.3f}, "
                f"so no seed count makes this ordering significant with the instance as the "
                f"unit. {floor} draws is the floor; experiments/variance_pilot.py is what "
                f"decides whether more are needed"
            )
        for name in names:
            if name == reference:
                continue
            effects = instance_effects(store, drawn, shape, name, reference, tier.seeds)
            if effects is None:
                continue
            lines.append(f"    {effects!r}")
            lines.extend(f"        {outcome!r}" for outcome in effects.per_draw)
            lines.append(f"        {_pseudo_replication_line(effects)}")
    lines.append(
        "  no pooled figure is printed, and pooled_metric() refuses to compute one: "
        "instance x seed pairs are not independent observations, so a mean over them "
        "would carry a standard error narrower than this design supports and would read "
        "as more evidence rather than less"
    )
    return lines


def _pseudo_replication_line(effects: InstanceEffects) -> str:
    """What the pooled reading of this arm would have claimed, as a factor.

    Measured rather than asserted, and from the *point* estimate of the instance
    term rather than a conservative bound on it: a number quoted to argue
    against pooling has to be the one that is hardest to dismiss, and inflating
    the instance term would be the self-serving direction here.

    At the seed count these draws were actually run at, so the factor describes
    the pooled figure somebody could compute from this very store rather than
    one from a design nobody ran.
    """
    components = decompose(effects.arm, list(effects.differences), conservative=False)
    factor = components.design_effect(seeds=components.seeds)
    return (
        f"instance term {components.between:.4f} against seed term "
        f"{components.within:.4f}; pooling these draws at {components.seeds:g} seeds "
        f"would understate the standard error {factor:.2f}x"
    )


def _arm_rows(
    held: Mapping[str, Mapping[int, RunRecord]],
    names: list[str],
    seeds: Sequence[int],
    attainable: AttainableOptimum | None,
) -> tuple[list[str], set[str], set[str]]:
    """One line per arm, which sat on the ceiling, and which never fitted a model.

    Split out of `report` rather than inlined: `report` was already at the branch
    limit, and the alternative to a helper is a table nobody may add a column to.

    The `proxy` column is the point of the split. Proxy spend is a budget someone
    chose -- ``steps x batch_size`` for the GFlowNet, ``generations x population``
    for the `genetic+search` ablation -- so an arm that wins on regret while
    spending an order of magnitude more surrogate evaluations has won on compute,
    and printing the two side by side is the only place a reader would see that.

    The `unfit` column is the second thing a number alone cannot say. A
    supervised arm whose model never fitted spends its budget, fills its plates
    and reports a regret indistinguishable from a fitted one's -- see
    [RunRecord.fitted][evogfn.benchmark.store.RunRecord.fitted] -- so without
    this the row is a random baseline printed under a supervised method's name,
    and the reading it invites is that the method was tried and beaten.

    Args:
        held: Stored records by arm.
        names: The arms, in report order.
        seeds: The tier's seeds, fixing the order metrics are pulled in.
        attainable: What the task can reach, or ``None`` when unaudited.

    Returns:
        The report lines; the arms whose share of seeds on the attainable
        optimum is at or above `VACUOUS_SHARE`; and the arms that finished at
        least one campaign without fitting. Both sets are returned rather than
        recomputed by the caller, since recomputing is how the table and the
        comparison below it drift into disagreeing about which arms are which.
    """
    lines, solved, unfitted = [], set(), set()
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
        # Counted over every stored seed, including the exhausted ones, and
        # before the early return below: an arm that exhausted everywhere without
        # ever fitting has no mean to print and is the case where "it never
        # fitted" is the entire finding.
        _, never = fit_status(records, seeds)
        if never:
            unfitted.add(name)
        if not len(regret):
            lines.append(
                f"  {name:<18} exhausted on all {exhausted} of its stored seeds; it could not "
                f"propose designs it had not already measured, so there is no number here"
            )
            continue
        failed = f"  exhausted={exhausted}" if exhausted else ""
        # On the row, not only in the line beneath it. A reader scanning the
        # regret column sees every other qualifier here, and an unmarked row is
        # the one that gets quoted.
        unfit = f"  unfit={never}" if never else ""
        error = regret.std(ddof=1) / len(regret) ** 0.5 if len(regret) > 1 else 0.0
        share = _at_optimum(records, attainable)
        if share >= VACUOUS_SHARE:
            solved.add(name)
        # On the row itself, not in a footnote: a decomposition row sitting
        # unmarked among published pipelines is read as one of them. Resolved
        # through `ablations` so a promoted ladder rung is marked by the same
        # code that marks `genetic+search`, rather than by a second list.
        decomposes = ablations().get(name)
        mark = f"  [ablation of {decomposes}]" if decomposes else ""
        lines.append(
            f"  {name:<18} regret {regret.mean():>7.3f} +/- {error:<6.3f} "
            f"at-opt {share:>5.2f}  feas {feasible.mean():>5.3f}  "
            f"div {spread.mean():>5.2f}  spent {spent.mean():>6.0f}  "
            f"proxy {proxy.mean():>7.0f}  n={len(regret)}{failed}{unfit}{mark}"
        )
    return lines, solved, unfitted


def _paired(  # noqa: PLR0913 - each argument is one qualifier a p-value needs
    held: Mapping[str, Mapping[int, RunRecord]],
    names: list[str],
    seeds: list[int],
    reference: str | None,
    *,
    solved: set[str],
    unfitted: set[str],
) -> list[str]:
    """Compare every arm against the reference, paired across shared seeds.

    Args:
        held: Stored records by arm.
        names: The arms, in report order.
        seeds: Seeds every arm has, so the comparison is genuinely paired.
        reference: The arm to compare against, or ``None`` if the tier has none.
        solved: Arms already sitting on the attainable optimum.
        unfitted: Arms that finished at least one campaign without fitting the
            model they are named for.

    Returns:
        Report lines, including a line naming the omission when there is
        nothing to compare against -- an absent section reads as "nothing
        separated these arms", which is a different statement from "nothing
        was tested" -- a line marking every ablation, whose comparison against a
        published pipeline is an attribution and not a ranking, and a line
        marking every comparison that names an arm which never fitted, whose
        p-value is about a random screen rather than about a supervised method.
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
        if decomposes := ablations().get(name):
            lines.append(f"        attribution: decomposes {decomposes}; {_attribution(name)}")
        # Said before the p-value is read, like the vacuous marker below and for
        # the same reason: a difference against an arm that never fitted is a
        # difference against random screening, and reporting it unmarked is how
        # "we beat MLDE" gets written about a row where MLDE never ran.
        if unfit := unfitted.intersection({name, reference}):
            lines.append(
                f"        not a method comparison: {', '.join(sorted(unfit))} never fitted "
                f"a model on some or all of these seeds, so this difference is against a "
                f"random screen wearing a supervised method's name"
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
    parser.add_argument(
        "--promote",
        metavar="RUNG",
        help="Record which rung of the variant ladder ships, and stop. Runs "
        "nothing and reports nothing else: it reads the ladder's stored "
        "campaigns on the diagnostic landscape, prints every rung against the "
        "base, and writes the decision. Naming the rung is the point -- deriving "
        "the winner here would make promotion an automatic consequence of the "
        "ladder having run, which is what a separate step exists instead of.",
    )
    args = parser.parse_args(argv)

    # Before any tensor work: a multithreaded matmul sums in thread-completion
    # order, and a few hundred gradient steps turn that into a different design.
    configure_determinism()
    if not args.report and not args.promote and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    store = ResultStore(Path(args.results))
    if args.promote:
        # Before any tier is selected, and returning outright. A promotion that
        # shared an invocation with a run would fix the headline configuration
        # and then immediately benchmark it, which is the one ordering the
        # separate step exists to make impossible.
        return promote(store, args.promote, report=_flush)
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
