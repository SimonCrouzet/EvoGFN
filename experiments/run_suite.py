"""Run the benchmark suite, resuming whatever is already stored.

Safe to interrupt and safe to re-run. Every campaign is written the moment it
finishes, and a second invocation runs only what is missing -- so raising a
tier's seed count from 30 to 50 costs twenty campaigns per arm, not fifty.

    uv run python experiments/run_suite.py                  # everything
    uv run python experiments/run_suite.py --tier main      # headline only
    uv run python experiments/run_suite.py --seeds 50       # raise the count
    uv run python experiments/run_suite.py --report         # no runs, just read

Results land under ``results/`` as one JSONL file per task and method.

How to read the report
----------------------

**Regret** is against the **attainable** optimum -- what
[evogfn.benchmark.attainable][] audited a task's search space to contain -- not
against the landscape's own, which carries a floor no method can clear. Where
the audit could not close the bracket, the interval is printed and the regret is
against its conservative end, so a regret at or below zero means the arm matched
everything the audit could construct rather than that it was perfect. Those arms
are marked **solved** and comparisons drawn on them are marked vacuous.

**The reference is `genetic`**, the Ehrlich paper's own algorithm at its own
hyperparameters; methods are compared as published. `genetic+search`,
`random+screen`, `genetic+screen` and `genetic+distinct` are decomposition rows
rather than baselines, and carry `[ablation of ...]` wherever they appear.

**Proxy calls** are read off the record rather than derived from the
configuration: the closed form ``steps x batch_size`` is wrong for an arm that
breeds, since a genetic teacher's proposals are themselves proxy evaluations
(``genetic-gfn`` records about 22% more than the formula predicts at the same
steps, width and round count). One caveat from [evogfn.benchmark.methods][]: a
sampler rebuilt at an anchor move restarts its own accounting, so on a
re-anchoring task the stored count covers the last anchor's rounds and reads as
a floor.

**`unfit=n` and `NEVER FITTED`** mark an arm that never reached its supervised
stage. [MLDE.observe][evogfn.algorithms.baselines.mlde.MLDE.observe] discards an
infeasible assay, having no fitness to regress on, so under a transition
constraint the screening plates can buy too few training examples for the
handover to happen at all. Such a campaign still spends its budget and reports a
regret indistinguishable from a fitted one's, so it is a random screen under a
supervised method's name.

**The replication tier is read one draw at a time.** Instance x seed pairs are
not independent observations, so `pooled_metric` **raises** on a task set holding
two draws of one shape rather than returning a mean. The report gives each arm's
per-draw effect, an interval at ``n = draws``, a sign test, and the factor by
which pooling would have understated the error. Three draws cannot reach
significance under a sign test at any seed count.

**The five mechanism rungs run in `main` and nothing is selected from them** --
the shipped arm as the base, plus ``+terminal``, ``+anchor``,
``+terminal+anchor`` and ``+wide``
([variant_arms][evogfn.benchmark.methods.variant_arms]). They are paired against
the shipped base in a section of their own.

Four of those rungs are **reproduced at report time and nothing is stored for
them**: ``+terminal`` defers the feasibility rule to the stop action, and
`gb1-anchor` and `trpb-anchor` have no transition matrix to defer, so the two
environments describe the identical graph there. Reproduced rows print in italic
with a legend naming the arm they repeat. Which rows those are is derived from
the environment, so a task that gains a transition matrix is measured instead
with no edit anywhere.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
    objective_family,
    reproduced_rungs,
    sensitivity,
    shipped_base,
    variant_arms,
)
from evogfn.benchmark.scoring import (
    SCORING_RULE,
    scored_metric,
    scoring_note,
    worst_case_from_attainable,
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
    objective_family_tier,
    objective_task,
    records_to_metric,
    replicate_instance,
    replication,
    rounds_curve,
    run_anchor_study,
    run_tier,
    width2_tier,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from evogfn.benchmark.attainable import AttainableOptimum
    from evogfn.benchmark.scoring import WorstCase
    from evogfn.benchmark.statistics import PairedComparison
    from evogfn.benchmark.store import RunRecord
    from evogfn.benchmark.tasks import Task

#: Seeds per tier. The main tiers differ because campaign cost differs by an
#: order of magnitude between L=4 and L=256, not because the claims differ.
#: How often the shard supervisor wakes to reap finished children.
_SHARD_POLL_SECONDS = 2.0

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

#: The headline tier, and the one tier that carries the mechanism rungs. Named
#: for the same reason `ANCHOR_TIER` is: `methods_for` branches on it, and a
#: misspelling does not raise -- it falls through to the baseline set, so the
#: five-rung study would silently become a one-arm row and the table would look
#: exactly as it did before anybody decided to report the mechanisms.
MAIN_TIER = "main"

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
        # The five mechanism rungs run here rather than in a tier of their own on
        # the diagnostic landscape. A separate tier was what a *selection* needed
        # -- decide off the test set, then promote -- and no rung is selected any
        # more, so what a second tier would now buy is the same five arms measured
        # where no claim is made, at 500 campaigns, and read across landscapes to
        # the tasks that do carry the claims. That last step is the inference the
        # reward-exponent reversal already cost this project once.
        Tier(MAIN_TIER, cheap, tuple(range(main_seeds)), Purpose.BENCHMARK),
        # The published pipelines and the selected arm, at `main`'s seed count --
        # but **not** the mechanism rungs, which stay on `main` alone. This tier
        # asks whether the *ordering* survives another draw from the instance
        # generator, and a mechanism's size is not part of that ordering; adding
        # the four rungs here would cost 2,400 campaigns to replicate a
        # decomposition rather than a claim.
        #
        # Same seed count as `main`, because it answers a question about `main`:
        # every constrained comparison there rests on one draw from the generator,
        # and a hundred seeds vary the wild type and the surrogate's
        # initialisation without ever varying the instance. A lower-powered
        # replicate could not distinguish "the ordering broke" from "we ran out of
        # seeds", which is the one thing it exists to decide.
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
        # The second predicate family in the headline comparison, not only in
        # the diagnostic support study: `feasibility` and `protocol-alde`
        # re-run under a width-2 contact predicate in place of their own
        # Ehrlich-adjacency rule. Main-tier seed count, since this asks whether
        # the headline finding survives a different constraint shape on the
        # same landscapes, not a diagnostic axis.
        width2_tier(range(main_seeds)),
        # The objective comparison carried from the diagnostic landscape onto the
        # constrained tasks, plus a learned-P_B rung. Main-tier seed count,
        # because which objective wins where construction is constrained is a
        # headline question; `gfn-tb`/`gfn-subtb` reuse their headline cells, so
        # only the four arms new here cost a campaign.
        objective_family_tier(range(main_seeds)),
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


def mechanism_ablations() -> dict[str, str]:
    """The rungs that are decomposition rows, mapped to the arm they decompose.

    Every rung but the base, and the base is the shipped arm -- so this is the
    four mechanism rows of the headline table, each pointing at the
    configuration it is one step away from. They are exactly what
    `genetic+search` is to `genetic`: real campaigns, honestly run, that answer
    *which part of the method does the work* rather than *which method a lab
    should run*, and an unmarked one sitting among published pipelines is read
    as a pipeline. ``+wide`` is the one that would be misread hardest -- it is a
    capacity control whose only purpose is to make ``+anchor`` attributable, so
    a reader taking it for a method would take a deliberately over-sized policy
    for something somebody proposed.

    Derived from `variant_arms` rather than written out, on the same reasoning
    that put the ladder on the recorded selection: the rung names carry the
    shipped arm's name as a prefix, so a literal list here would stop matching
    the table the day a selection moved, and the rows it stopped matching would
    go unmarked rather than raise.

    Returns:
        Arm name to the shipped arm it decomposes.
    """
    base = shipped_base().name
    return {name: base for name in (*variant_arms(), *_reinit_rung()) if name != base}


#: The rung that switches amortisation off, and the one rung `variant_arms` does
#: not build.
#:
#: ``carry_policy`` off discards the trained weights at every anchor move and
#: rebuilds from the campaign's own stream -- at identical gradient steps, an
#: identical proxy spend and, at the opening anchor, identical weights. The arm
#: therefore differs from the shipped one in exactly one thing: whether what was
#: learned at the previous parent survives the move.
#:
#: It is built here rather than in
#: [variant_arms][evogfn.benchmark.methods.variant_arms] because it is not a
#: mechanism added on top of the shipped arm; it is the shipped arm's own
#: mechanism switched off. The flag has existed in
#: [gflownet][evogfn.benchmark.methods.gflownet] since the method was written,
#: is named there as "what *amortisation* means here", and has never been run.
REINIT_RUNG = "+reinit"


def _reinit_rung() -> dict[str, object]:
    """The shipped arm with amortisation removed.

    Returns:
        One arm, by name.
    """
    base = shipped_base()
    return {f"{base.name}{REINIT_RUNG}": base.rung(carry_policy=False)}


def _reproduced_on(task: Task) -> dict[str, str]:
    """Rungs this task cannot measure, mapped to the arm whose campaign they are.

    One function for both halves of the decision -- what `_run` does not run and
    what `report` reproduces -- because the two halves must agree exactly. A
    reproduction of a row that *was* run would print a copy beside a measurement;
    a row skipped and not reproduced would simply be missing, and an empty cell in
    a headline table is an absence a reader fills in as they like.

    Args:
        task: The task being run or reported on.

    Returns:
        Rung name to the arm it repeats, empty on any task where the mechanism has
        something to act on. Derived from the environment by
        [reproduced_rungs][evogfn.benchmark.methods.reproduced_rungs], so a task
        that gains a transition matrix starts being measured with no edit here.
    """
    reproduced = dict(reproduced_rungs(task=task))
    if not task.reanchor:
        # Nothing can survive a move the task never makes. With a fixed anchor
        # the policy is built once and `carry_policy` never fires, so this rung
        # is the base arm under a second name -- the same relation `+terminal`
        # has to the base where nothing constrains construction, and it gets the
        # same treatment: reproduced at report time, never stored.
        base = shipped_base().name
        reproduced[f"{base}{REINIT_RUNG}"] = base
    return reproduced


def _reproduce(
    store: ResultStore, task: Task, held: dict[str, Mapping[int, RunRecord]]
) -> dict[str, str]:
    """Fill the rows this task cannot measure from the campaigns they repeat.

    Reproduced here and not stored, which is the decision this arrangement turns
    on. A stored copy is indistinguishable from a measurement the moment it is
    written -- same cell shape, same fingerprint, same seeds -- and every reader of
    the store would count it as an independent campaign. Reproducing at report
    time keeps the store a record of what was measured and puts the copy where the
    claim is made, on the line a reader takes the number from.

    Args:
        store: Where records live, consulted to catch a contradiction.
        task: The task being reported on.
        held: Records by arm, modified in place so the reproduced rows are read
            by every section below exactly as a measured one would be. Its keys
            are the arms this tier reports, and a rung outside them is not
            reproduced: there is no row for it to fill.

    Returns:
        Rung name to the arm it reproduces, for the marker and the legend. A rung
        whose source has no records is not reproduced: an empty row copied from an
        empty row says nothing, and the arm simply prints nothing as it would
        have anyway.

    Raises:
        ValueError: If the store already holds campaigns for a rung this task
            declares a reproduction. The two claims cannot both be true, and
            neither silent resolution is acceptable: preferring the store prints
            an independent-looking row the suite believes could not exist, and
            preferring the copy discards a measurement somebody paid for. Either
            the task gained a transition matrix -- in which case the rung is now
            measurable and the records are the right answer -- or the records were
            produced against a task that has since lost one.
    """
    # Only rungs this table actually reports. A tier that does not run them has
    # no row to fill, and filling one anyway would put a legend under a table
    # holding nothing the legend describes.
    copies = {rung: source for rung, source in _reproduced_on(task).items() if rung in held}
    for rung, source in sorted(copies.items()):
        stored = store.usable(task.name, rung)
        if stored:
            why = (
                "this task never moves its anchor, so there is no move for a carried "
                "policy to survive and the two arms are one campaign"
                if rung.endswith(REINIT_RUNG)
                else "nothing on this task constrains construction, so the two arms "
                "build the identical graph"
            )
            raise ValueError(
                f"{task.name}/{rung}: the store holds {len(stored)} campaigns for a row this "
                f"suite reproduces from {source}, because {why}. Both claims cannot hold: "
                f"either the task changed and the rung is now measurable, in which case it "
                f"must be run and this reproduction dropped, or those records describe a task "
                f"that no longer exists and must be deleted"
            )
        if held.get(source):
            held[rung] = held[source]
    return {rung: source for rung, source in copies.items() if held.get(rung)}


def _mechanism_rungs(arms: Mapping[str, object]) -> dict[str, object]:
    """The rungs to add to a table that already holds the shipped arm.

    Args:
        arms: What the tier runs before the rungs are added.

    Returns:
        Every rung except the base, since the base **is** the arm the tier
        already runs. Adding it again under the ladder's own object would put
        one configuration in the table twice if the two builders ever disagreed
        about its name, and would spend a second hundred campaigns proving they
        agree if they did not.

    Raises:
        ValueError: If the base rung is not already among `arms`. That means the
            selection's builder and the ladder's have drifted apart on the arm's
            name, and the silent outcome is the bad one: two GFlowNet
            configurations in a headline table that the selection phase exists to
            leave holding exactly one.
    """
    base = shipped_base().name
    if base not in arms:
        raise ValueError(
            f"the ladder stands on {base!r}, which is not in the tier's arms "
            f"({', '.join(sorted(arms))}); the selection and the ladder disagree about "
            f"what this project ships, and reporting both would put two configurations "
            f"of one method in the headline table"
        )
    return {
        **{name: arm for name, arm in variant_arms().items() if name != base},
        **_reinit_rung(),
    }


#: Arms held back from every tier while a defect in them is being diagnosed.
#:
#: On `protocol-alde`, which has a live transition matrix, both terminal rungs
#: returned regret 0.8750 with a standard error of exactly zero over 31 seeds,
#: and returned it *identically to each other* -- which cannot be right, since
#: they differ by anchor conditioning and both mechanisms are active there. A
#: constant with no variance is what a sampler that never searches produces, so
#: the number is treated as a defect until something says otherwise rather than
#: as evidence that the mechanism fails.
#:
#: Suspending here rather than deleting the arms keeps the ladder's shape and
#: the stored campaigns intact, so the moment the diagnosis lands this is one
#: line to revert and the tier resumes where it stopped.
SUSPENDED_ARMS = frozenset(
    {
        "gfn-subtb@b0.1-s300-l0.9-h64+terminal",
        "gfn-subtb@b0.1-s300-l0.9-h64+terminal+anchor",
    }
)


def methods_for(tier: Tier) -> dict[str, object]:
    """Which methodologies a tier runs.

    The objectives diagnostic is GFlowNet-only, since a classical baseline has
    no training objective to vary; the sensitivity tier is narrower still, being
    one GFlowNet with one setting moved at a time; everything else compares
    methods.

    Every branch here sits **above** the selection lookup, and that placement is
    the point rather than an accident of ordering. Below it, a recorded selection
    replaces the untuned GFlowNet arms wholesale -- which is right for a tier
    comparing methods and fatal for the two tiers whose arms are defined
    *relative to* an untuned one: an anchor study missing `gfn-tb` would lose the
    cell its other four are read against.
    """
    fixed: dict[str, Callable[[], dict[str, object]]] = {
        "objectives": lambda: {**OBJECTIVES, **flow_objectives()},
        "objective-family": objective_family,
        "sensitivity": lambda: dict(sensitivity()),
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
    # The mechanism study, in the headline tier and nowhere else. Four rungs
    # rather than five, because the fifth is the arm already above: the base rung
    # *is* the selected arm, resolved from the same file by the same builder, so
    # adding it a second time would be one configuration under two names --
    # `_mechanism_rungs` raises rather than letting that pass silently.
    #
    # Appended rather than merged in place, so the four decomposition rows sit
    # after the pipelines they decompose instead of interleaving with them, and
    # the base keeps the position it had when it was the tier's only GFlowNet.
    #
    # It is not free: four extra arms across five tasks at a hundred seeds would
    # be 2,000 campaigns, of which 400 are not run -- see below -- leaving 1,600,
    # which at the recorded ~255s per campaign is on the order of 113 core-hours.
    # The headline seed count is what the `+terminal+anchor` rung needs rather
    # than a number inherited: it earns its place only by beating what the two
    # single rungs predict by addition, and a difference of differences carries
    # about twice the standard error of either main effect -- at the diagnostic
    # count it comes back inconclusive against its own prediction, which is the
    # one answer this study must not return.
    #
    # Two properties of these rungs are task-dependent, and both are handled where
    # the task is known rather than by a caveat in a table's footnotes:
    #
    # * `+terminal` is a **no-op without a transition matrix**. With no adjacency
    #   to defer, `TerminalFeasibilityEnvironment` and `MutationEnvironment`
    #   describe the identical graph, so on `gb1-anchor` and `trpb-anchor` the
    #   `+terminal` and `+terminal+anchor` rows would be the base and `+anchor`
    #   recomputed under other names -- 400 campaigns, and two pairs of identical
    #   rows a reader could quote as "we tested the mechanism here and it made no
    #   difference", which is a different and false claim from "there was nothing
    #   here to test". They are therefore **not run**: `reproduced_rungs` names
    #   them from the environment, `_run` omits them, and `report` reproduces the
    #   row in italic with a legend saying whose campaign it is.
    # * `+wide` is sized by `matched_capacity`, which counts parameters at the
    #   shape of the task it will run on -- 101 at 32 positions, 103 at 64, 88 at
    #   four sites, each a little over the conditioned arm rather than under. A
    #   single width would be right on one shape only: 101 carries 1.63% *fewer*
    #   parameters than `+anchor` on the 64-position tasks, and under-resourcing
    #   the control is the direction that manufactures the effect it exists to
    #   rule out. The resolved width and the achieved residual are written onto
    #   every record the arm produces.
    if tier.name == MAIN_TIER:
        arms = {**arms, **_mechanism_rungs(arms)}
    if tier.name in BUDGET_AXIS_TIERS:
        # `mlde-over-budget` runs one plate beyond its task's protocol, which is
        # the point of the arm and is exactly what a tier whose axis *is* the
        # budget cannot hold constant. See `BUDGET_AXIS_TIERS`.
        for name in OVER_BUDGET_ARMS:
            arms.pop(name, None)
    # Applied last, and to every tier: a suspended arm is one whose stored
    # numbers are not believed, so it must not reach a table by any route while
    # that is true. See `SUSPENDED_ARMS`.
    for name in SUSPENDED_ARMS:
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
#: method. `mlde+earlyfit` is here too, on the same reasoning rather than in
#: spite of it: what makes a row a decomposition is that it isolates one thing
#: about the pipeline beside it, and reading `mlde` against it separates "MLDE
#: loses because it suits a constrained space badly" from "MLDE loses because it
#: never fitted". That it reaches that by replacing a parameter rather than
#: adding a component changes which sentence explains it, not whether it needs
#: one -- and an unmarked row named `mlde+...` in a table of published pipelines
#: is the single likeliest thing here to be quoted as MLDE.
ABLATIONS = {
    "genetic+screen": "genetic",
    "genetic+search": "genetic",
    "genetic+distinct": "genetic",
    "random+screen": "random",
    "mlde+earlyfit": "mlde",
}

#: What each family of decomposition row separates, keyed by the pipeline it
#: decomposes. Three families, because they answer different questions and a row
#: carrying the wrong one is a row explaining itself wrongly: the classical
#: ablations split a surrogate's contribution from a sampler's, while a mechanism
#: rung splits one mechanism of the shipped GFlowNet from another.
_ATTRIBUTIONS = {
    "surrogate": (
        "separates the surrogate's contribution from the sampler's, and ranks "
        "no method a lab could run"
    ),
    # It says *nothing is chosen from these rows* in as many words, because the
    # obvious misreading of a five-row mechanism block in a headline table is
    # that the best row is the method. It is not: the base is what ships, the
    # other four say what each mechanism does on this task, and a rung that wins
    # here is a finding about the mechanism rather than a new configuration.
    "mechanism": (
        "separates one mechanism of the shipped configuration from the rest, "
        "measured on the task that carries the claim; nothing is selected from "
        "these rows and none of them is a method a lab would run"
    ),
    "handover": (
        "separates whether the published pipeline is unsuited to this landscape "
        "from whether it never reached its model at all; it is our adaptation, "
        "run at a training size no paper specifies, and no row of it may be "
        "quoted as the published method"
    ),
}

#: Decomposition rows whose question is the handover rather than the surrogate.
#: Kept as a set beside `ABLATIONS` rather than folded into it, because the two
#: mappings answer different questions -- one says what a row decomposes, the
#: other says which sentence explains it -- and a row that reached the table
#: with the wrong sentence would be explaining itself wrongly, which is worse
#: than an unexplained row.
_HANDOVER_ROWS = frozenset({"mlde+earlyfit"})


def ablations() -> dict[str, str]:
    """Every decomposition row in the table, mapped to what it decomposes.

    The classical rungs plus the GFlowNet's. Resolved through a function rather
    than left as a module constant because the second half of it is named after
    whatever configuration the selection recorded: `_arm_rows` and `_paired` both
    call this, so a rung that reaches the table reaches it marked, where a
    constant would have to be edited by whoever remembered to. That is the one
    place a convention fails -- an unmarked decomposition row among published
    pipelines reads as a pipeline that lost.

    Returns:
        Arm name to the pipeline or shipped arm it decomposes.
    """
    return {**ABLATIONS, **mechanism_ablations()}


def _attribution(name: str) -> str:
    """Which question a decomposition row answers, in the row's own terms."""
    if name in _HANDOVER_ROWS:
        return _ATTRIBUTIONS["handover"]
    kind = "mechanism" if name in mechanism_ablations() else "surrogate"
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
    return chosen if chosen in methods else None


def report(store: ResultStore, tier: Tier, reference: str | None = None) -> str:
    """Read the store and compare every arm against one, paired across seeds.

    Regret is read straight from the records, where it is already against the
    attainable optimum -- see `evogfn.benchmark.suite._scores`. What this adds is
    the context that makes it readable: the interval the task can reach, how many
    of an arm's seeds are sitting on it, what each arm spent to get there, which
    rows are ablations rather than published pipelines, **which supervised rows
    never got as far as fitting a model**, a second paired block that reads the
    mechanism rungs against the shipped arm rather than against `genetic`, and a
    refusal to present a paired comparison drawn on a solved task as though it
    ranked anything.

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
        held: dict[str, Mapping[int, RunRecord]] = {
            name: store.usable(task.name, name) for name in names
        }
        # Before the seed intersection and before any row is built, so a
        # reproduced row is read by every section below exactly as a measured one
        # is -- and is *marked* in every one of them.
        copies = _reproduce(store, task, held)
        seeds = [s for s in tier.seeds if all(s in held[n] for n in names if held[n])]
        rows, solved, unfitted = _arm_rows(held, names, tier.seeds, attainable, reproduced=copies)
        # Above the rows it applies to, and only where it applied: a mean taken
        # under the convention and one taken without it print identically, so a
        # table that states the rule unconditionally teaches a reader to skip it.
        if any("scored at worst" in row or "unscorable" in row for row in rows):
            lines.append(SCORING_RULE)
        lines.extend(rows)
        lines.extend(_reproduction_legend(copies))
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

        lines.extend(
            _paired(
                held,
                names,
                seeds,
                reference,
                solved=solved,
                unfitted=unfitted,
                reproduced=copies,
                worst_case=worst_case_from_attainable(attainable.lower if attainable else None),
            )
        )
        lines.extend(
            _mechanism_pairs(
                held,
                names,
                seeds,
                solved=solved,
                unfitted=unfitted,
                reproduced=copies,
                worst_case=worst_case_from_attainable(attainable.lower if attainable else None),
            )
        )
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


def _reproduction_legend(reproduced: Mapping[str, str]) -> list[str]:
    """The legend the italic rows above cannot be read without.

    Printed under the table it belongs to rather than once per report, because a
    reader quoting a row reads the block around it and not the top of a file. It
    is the whole of the difference between "we tested the mechanism here and it
    made no difference" -- which the identical numbers invite and which is false --
    and "there was nothing here to test", which is what happened.

    Args:
        reproduced: Rung name to the arm it reproduces, empty on a task that
            measured everything.

    Returns:
        The legend lines, or none.
    """
    if not reproduced:
        return []
    pairs = ", ".join(f"{rung} = {source}" for rung, source in sorted(reproduced.items()))
    return [
        f"  *italic* rows are REPRODUCTIONS, not measurements ({pairs}). Nothing on this task "
        f"constrains the order substitutions are made in, so deferring feasibility to the stop "
        f"action defers nothing: TerminalFeasibilityEnvironment and MutationEnvironment describe "
        f"the identical graph here and the rung is its neighbour's campaign under another name. "
        f"It was not run and nothing is stored for it; the numbers above are copied, and the "
        f"agreement between the two rows is arithmetic rather than a finding about the mechanism"
    ]


def _arm_rows(
    held: Mapping[str, Mapping[int, RunRecord]],
    names: list[str],
    seeds: Sequence[int],
    attainable: AttainableOptimum | None,
    *,
    reproduced: Mapping[str, str] | None = None,
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

    A **reproduced** row is the third. Where a rung's campaign on this task is
    another arm's campaign under a second name, the row is not a measurement at
    all -- see `_reproduce` -- and it prints in italic, marked with the arm it
    copies. The numbers are real; what is not real is their independence, and two
    identical rows read as agreement between two experiments unless the line
    itself says otherwise.

    Args:
        held: Stored records by arm.
        names: The arms, in report order.
        seeds: The tier's seeds, fixing the order metrics are pulled in.
        attainable: What the task can reach, or ``None`` when unaudited.
        reproduced: Rung name to the arm it copies, for the rows this task could
            not measure.

    Returns:
        The report lines; the arms whose share of seeds on the attainable
        optimum is at or above `VACUOUS_SHARE`; and the arms that finished at
        least one campaign without fitting. Both sets are returned rather than
        recomputed by the caller, since recomputing is how the table and the
        comparison below it drift into disagreeing about which arms are which.
    """
    copies = reproduced or {}
    lines, solved, unfitted = [], set(), set()
    for name in names:
        records = held[name]
        if not records:
            continue
        # Scored rather than dropped: a campaign whose every proposal was
        # infeasible finished, spent its plate and measured nothing, so its
        # regret is `+inf` and one such seed takes the whole row to infinity.
        # See [evogfn.benchmark.scoring][] for why the alternative -- dropping
        # those seeds -- both flatters the arm that failed most often and breaks
        # the pairing below.
        regret, scored, unscorable = scored_metric(
            records,
            seeds,
            "regret",
            worst_case_from_attainable(attainable.lower if attainable else None),
        )
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
        # through `ablations` so a mechanism rung is marked by the same code that
        # marks `genetic+search`, rather than by a second list.
        decomposes = ablations().get(name)
        mark = f"  [ablation of {decomposes}]" if decomposes else ""
        # Italic, in the only way a fixed-width text report has one: asterisks
        # around the name, which is the plain-text convention and which the legend
        # under the table names explicitly rather than leaving to be inferred. The
        # marker beside it carries the arm, because "this is a copy" and "a copy of
        # what" are two different things a reader needs and only one of them fits
        # in an emphasis.
        copied = copies.get(name)
        label = f"*{name}*" if copied else name
        repeat = f"  [reproduced from {copied}]" if copied else ""
        lines.append(
            f"  {label:<18} regret {regret.mean():>7.3f} +/- {error:<6.3f} "
            f"at-opt {share:>5.2f}  feas {feasible.mean():>5.3f}  "
            f"div {spread.mean():>5.2f}  spent {spent.mean():>6.0f}  "
            f"proxy {proxy.mean():>7.0f}  n={len(regret)}{failed}{unfit}{mark}{repeat}"
            f"{scoring_note(scored, unscorable)}"
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
    reproduced: Mapping[str, str] | None = None,
    worst_case: WorstCase | None = None,
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
        reproduced: Rung name to the arm it copies. A copy is **not** paired: its
            comparison against any third arm is the comparison its source already
            printed, to the last digit, and a second p-value with a second name on
            it is one piece of evidence read as two. Against the source itself the
            statistic is not merely redundant but undefined -- every paired
            difference is exactly zero, so the standard error is zero -- and a
            printed ``nan`` there would look like a failed test rather than a
            comparison that never had anything to compare.
        worst_case: How to score a seed whose campaign measured nothing, so that
            an unmeasured seed neither makes every difference against it ``-inf``
            nor gets dropped from one side of a pair. ``None`` scores nothing,
            which is right only where the task was never audited.

    Returns:
        Report lines, including a line naming the omission when there is
        nothing to compare against -- an absent section reads as "nothing
        separated these arms", which is a different statement from "nothing
        was tested" -- a line marking every ablation, whose comparison against a
        published pipeline is an attribution and not a ranking, a line marking
        every comparison that names an arm which never fitted, whose p-value is
        about a random screen rather than about a supervised method, and a line
        for each reproduced row saying why it is not paired.
    """
    copies = reproduced or {}
    if reference is None:
        return ["  no reference arm in this tier, so nothing is paired"]
    base = held.get(reference)
    if not base or not seeds:
        return [f"  reference {reference} has no usable seeds here, so nothing is paired"]
    lines = [f"  paired vs {reference} (positive favours the first):"]
    for name in names:
        if name == reference or not held[name]:
            continue
        if source := copies.get(name):
            # Said rather than skipped, on the reasoning every other refusal in
            # this report carries: an omitted line reads as "these did not
            # separate", which is the opposite of what happened. Two sentences,
            # because the copy stands in two different relations to a reference --
            # redundant with it against a third arm, degenerate against its own
            # source -- and one wording would be wrong about one of them.
            because = (
                f"it is {source}'s own campaign here, so every paired difference is exactly "
                f"zero by construction and the statistic has no standard error"
                if source == reference
                else f"it reproduces {source}, so this comparison is {source}'s, already "
                f"printed under the name of the arm that measured it"
            )
            lines.append(f"    {name} vs {reference}: not paired -- {because}")
            continue
        # Scored, not dropped, and by the same rule the rows above use: an
        # unmeasured seed left in raw would make every difference against it
        # `-inf`, and dropped from one side only would pair two different seed
        # sets. See [evogfn.benchmark.scoring][].
        reader = worst_case or worst_case_from_attainable(None)
        mine, _, _ = scored_metric(held[name], seeds, "regret", reader)
        theirs, _, _ = scored_metric(base, seeds, "regret", reader)
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


def _mechanism_pairs(  # noqa: PLR0913 - each argument is one qualifier a p-value needs
    held: Mapping[str, Mapping[int, RunRecord]],
    names: list[str],
    seeds: list[int],
    *,
    solved: set[str],
    unfitted: set[str],
    reproduced: Mapping[str, str] | None = None,
    worst_case: WorstCase | None = None,
) -> list[str]:
    """The mechanism rungs, paired against the shipped arm rather than `genetic`.

    A second block and not a replacement for the first. The table's reference is
    a published pipeline because that is what a lab chooses between, and every
    arm here -- rungs included -- has to be readable against it. But a rung's
    difference from `genetic` is not what the rung measures: ``+anchor`` is one
    step from the shipped configuration and nothing else, so the number that
    isolates the mechanism is the paired difference against *that*. Without this
    block the rung's row carries an attribution line saying it decomposes the
    shipped arm, sitting directly beneath a p-value taken against a different
    one, which is a row explaining itself wrongly.

    It cannot be had by subtracting the two `genetic`-referenced lines either:
    those are two paired differences over the same seeds, and their difference
    has a standard error neither of them states.

    Args:
        held: Stored records by arm.
        names: The arms, in report order.
        seeds: Seeds every arm has, so the comparison is genuinely paired.
        solved: Arms already sitting on the attainable optimum.
        unfitted: Arms that finished a campaign without fitting.
        reproduced: Rung name to the arm it copies. A reproduced rung is one whose
            mechanism has nothing to act on here, so it is exactly the rung this
            block cannot say anything about -- and the block says that, in place of
            a difference that is zero by construction.
        worst_case: Passed through to
            [_paired][run_suite._paired]; the rungs are the arms this most often
            matters for, since a mechanism that proposes nothing feasible is
            exactly what it is there to report.

    Returns:
        Report lines, or none where this tier holds no rungs -- which is every
        tier but `main`, and is why the block is keyed off the arms present
        rather than off the tier's name.
    """
    copies = reproduced or {}
    base = shipped_base().name
    decomposed = mechanism_ablations()
    rungs = [name for name in names if name == base or name in decomposed]
    if base not in rungs or len(rungs) < 2:  # noqa: PLR2004 - a study needs the base and one rung
        return []
    heading = "\n  --- mechanisms, each one step from the shipped arm ---"
    measured = [name for name in rungs if name != base and name not in copies and held.get(name)]
    if not measured:
        # Said rather than left as a heading over nothing. `_paired` skips an arm
        # with no records, which is right in the main table and wrong here: a
        # block whose only content is its own header reads as four mechanisms
        # that separated nothing, when what happened is that none of them ran.
        # A block holding *only* reproductions is the same failure with numbers
        # in it, and worse for being readable.
        return [
            heading,
            f"  no rung has run on this task, so nothing decomposes {base} here; this is "
            f"an unrun study rather than four mechanisms that made no difference",
        ]
    return [
        heading,
        *_paired(
            held,
            rungs,
            seeds,
            base,
            solved=solved,
            unfitted=unfitted,
            reproduced=copies,
            worst_case=worst_case,
        ),
    ]


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Run this many shards as subprocesses. Determinism pins each "
        "process to one thread, so the unit of parallelism has to be the "
        "process; and the store keeps one file per task and method, so shards "
        "never contend. Sharding lives here rather than in a shell script "
        "because the roster -- which tasks, which arms -- is then the tier "
        "declarations, which are typed and tested, instead of a list nothing "
        "checks.",
    )
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

    if args.workers > 1 and not args.report:
        return _run_sharded(selected, args)

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
        # The second departure from the cross, and unlike the anchor study it is
        # a property of the (task, arm) pair rather than of the tier: `+terminal`
        # is a real mechanism on a task that constrains construction and its
        # neighbour's own campaign on one that does not. Passed as a rule rather
        # than resolved here, so the same function decides what is skipped and
        # what the report reproduces.
        return run_tier(running, arms, store, report=_flush, omit=_reproduced_on)  # type: ignore[arg-type]
    if by_method:
        # Refused rather than applied. The study is a list of ``(task, arm)``
        # cells and not an arm set, so an arm filter would drop whole cells
        # silently -- and the cells it would drop are the ones whose absence
        # turns "carrying the policy helps" into a column with nothing paired
        # against it. Run this tier without ``--method``.
        _flush(f"{running.name}: runs named cells, so --method cannot select within it")
        return None
    return run_anchor_study(store, running.seeds, report=_flush)


def _shard_bounds(count: int, workers: int) -> list[tuple[int, int]]:
    """Contiguous seed ranges covering ``count`` seeds, at most ``workers`` of them.

    Contiguous rather than strided so a shard's log reads as a seed range, and
    so a shard that dies leaves an obvious hole rather than a comb.

    Args:
        count: Seeds in the tier.
        workers: Most shards to cut it into.

    Returns:
        ``(low, high)`` pairs, half-open, covering ``range(count)`` exactly.
    """
    workers = max(1, min(workers, count))
    step, spare = divmod(count, workers)
    bounds, low = [], 0
    for shard in range(workers):
        high = low + step + (1 if shard < spare else 0)
        bounds.append((low, high))
        low = high
    return bounds


def _run_sharded(selected: list[Tier], args: argparse.Namespace) -> int:
    """Re-invoke this script as one subprocess per (task, seed range).

    Each child runs with ``--workers 1``, so the recursion is one level deep by
    construction. Every campaign is seeded from its own seed rather than from
    process order, so a sharded run and a serial one produce identical records
    -- which is what makes this a scheduling decision and not an experimental
    one.

    Args:
        selected: Tiers already filtered by the caller's arguments.
        args: The parsed command line, reused verbatim for the children.

    Returns:
        A process exit code: 0 when every shard succeeded, 1 otherwise.
    """
    jobs: list[list[str]] = []
    for tier in selected:
        for task in tier.tasks:
            for low, high in _shard_bounds(len(tier.seeds), args.workers):
                if low == high:
                    continue
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--tier",
                    tier.name,
                    "--task",
                    task.name,
                    "--seeds",
                    str(args.seeds),
                    "--diagnostic-seeds",
                    str(args.diagnostic_seeds),
                    "--seed-from",
                    str(low),
                    "--seed-to",
                    str(high),
                    "--results",
                    args.results,
                ]
                for name in args.method or []:
                    command += ["--method", name]
                jobs.append(command)

    _flush(f"{len(jobs)} shards at {args.workers} workers")
    failures = 0
    running: list[subprocess.Popen[bytes]] = []
    pending = list(jobs)
    while pending or running:
        while pending and len(running) < args.workers:
            running.append(subprocess.Popen(pending.pop(0)))  # noqa: S603 - our own argv
        done = [process for process in running if process.poll() is not None]
        for process in done:
            running.remove(process)
            failures += process.returncode != 0
        if not done:
            time.sleep(_SHARD_POLL_SECONDS)
    if failures:
        _flush(f"{failures} of {len(jobs)} shards failed")
    return 1 if failures else 0


def _flush(message: str) -> None:
    """Print immediately, so an overnight run can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
