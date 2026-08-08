"""How many landscape draws the replication tier needs, measured rather than assumed.

`evogfn.benchmark.suite.REPLICATION_SEEDS` is ``(3, 5, 7)``. Three draws, chosen
a priori, justified in the constant's own comment as "enough to see an ordering
break". Nothing has ever checked that, and two things are already known about it
without running anything:

* **Three draws cannot clear the sign-test floor at any seed count.** The
  per-instance analysis reports how many draws the ordering held on, and the
  only test that reads that column without assuming a distribution is the sign
  test. Its strongest possible verdict on three draws is unanimity at two-sided
  ``p = 0.25``. Running a thousand seeds per draw does not move that number,
  because seeds are not the unit.
* **The instance term does not shrink with seeds.** The standard error of a mean
  effect over ``m`` draws at ``n`` seeds is
  ``sqrt(sigma_b^2 / m + sigma_w^2 / (m n))``. Only the second term reads ``n``.
  Past the point where ``sigma_w^2 / n`` is small against ``sigma_b^2``, seeds
  are free precision on the wrong quantity.

What this script measures is ``sigma_b`` against ``sigma_w``, and what it
produces is a draw count from a rule fixed before the numbers arrived.

Why a pilot is shaped the opposite way to the tier it sizes
-----------------------------------------------------------

The tier runs few draws at many seeds. The pilot runs **many draws at few
seeds**, and the inversion is the whole point: ``sigma_w`` is estimated on
``sum(n_i - 1)`` degrees of freedom and is well determined almost immediately,
while ``sigma_b`` is estimated on ``m - 1`` and is the term the design turns on.
At the tier's own three draws that is two degrees of freedom, where the 95%
upper bound on a variance is nineteen times its estimate -- which is to say the
tier as it stands cannot size itself, and could not do so if every draw ran a
thousand seeds.

The rule, stated before any draw was run
-----------------------------------------

> **The replication design runs the smallest number of landscape draws ``m``
> satisfying both clauses, and no draw count is declared when that exceeds the
> cap:**
>
> 1. *precision*: ``m >= ((z + z_power) / delta*)^2 * (sigma_b^2 + sigma_w^2 / n)``
>    -- the across-instance interval on the mean effect resolves an ordering of
>    size `ORDERING_MARGIN` with 80% power, at the seed count the tier runs, with
>    the **instance** as the unit of replication;
> 2. *the sign floor*: ``m >= unanimity_floor(0.05)`` -- enough draws that a
>    unanimous per-instance verdict is significant at all. This clause reads no
>    variance and no effect, so no measurement can argue it away.
>
> ``sigma_b`` is the 95% **upper** bound, not the point estimate. A thinner pilot
> widens that bound, which asks for more draws: **underpowering fails toward a
> larger design**, which costs compute rather than shrinking a reported claim.
> This is the same polarity `evogfn.benchmark.saturation` is built on and it is
> the property that makes the number safe rather than merely argued.
>
> When the two clauses ask for more than `DRAW_CAP`, **no draw count is
> declared**. What is reported in its place is the residual: the smallest
> ordering a design at the cap can resolve. That is the honest statement, and it
> is more useful than a knee anyway -- set it beside the margin a headline claims
> over its baseline and the exposure is quantified rather than asserted.

What the trigger cannot see
---------------------------

`evogfn.benchmark.statistics.draws_needed` takes a
[VarianceComponents][evogfn.benchmark.statistics.VarianceComponents], which
deliberately carries no mean. A rule that could see which way the effect went
could be re-run until the answer was agreeable; this one cannot, and the
signature is the guarantee rather than a promise made here.

`ORDERING_MARGIN` is the other half of that. It is declared below, before the
numbers, and it is *not* read off the observed effects -- a margin chosen after
the variance is known is a margin chosen to produce an affordable answer.

Where this runs, and where it must not
---------------------------------------

On `replication`'s own tasks at draws the tier does not use, which is the only
place a between-instance variance is even defined. That means it runs on
landscapes of the headline family, so the reading has to be stated: this
measures the *dispersion* of an effect across draws and reports no effect. No
number from here belongs in a results table, in a caption, or in a claim -- the
boundary `evogfn.benchmark.selection` draws around its own screen. Sizing a
design from a pilot is what a pilot is for.

A script rather than a tier, for the reason
``experiments/proxy_saturation.py`` gives: a measurement that fixes our own
experimental design does not belong in the same ``--report`` output as the
headline tables, and nothing under ``experiments/`` sits inside the store's
dependency closure.

    uv run python experiments/variance_pilot.py --report      # read, no runs
    uv run python experiments/variance_pilot.py               # run the pilot
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.methods import BASELINES, shipped_base
from evogfn.benchmark.statistics import (
    VarianceComponents,
    compare,
    decompose,
    draws_needed,
    pool_components,
    unanimity_floor,
    unanimity_p,
)
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import (
    REPLICATION_SEEDS,
    Purpose,
    Tier,
    records_to_metric,
    replicate_instance,
    replication,
    run_tier,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evogfn.benchmark.methods import Methodology
    from evogfn.benchmark.tasks import Task

#: The smallest ordering the replication tier must be able to confirm, in regret.
#:
#: Declared here, before any draw was run, and deliberately not read off the
#: observed effects: a margin fixed after the variance is known is a margin
#: fixed to produce an affordable answer. Three independent grounds, which
#: happen to agree:
#:
#: * it is the size of the smallest gap between two *published pipelines* on the
#:   replicated tasks -- the constrained genetic algorithm sits about 0.05 regret
#:   from the unconstrained one there, and an ordering the tier cannot confirm at
#:   that size is an ordering the tier cannot confirm at all;
#: * it is about 7% of those tasks' regret scale, which sits where
#:   `evogfn.benchmark.saturation`'s own margin sits relative to the diagnostic
#:   landscape's;
#: * it is several times the precision the headline table is *printed* at -- the
#:   reported ``regret +- error`` carries a standard error near 0.01 on these
#:   arms -- so a margin below it would be asking the design for precision the
#:   table does not display.
#:
#: Tightening this asks for more draws and is safe; loosening it asks for fewer,
#: which is why it is a constant and not a command-line default that drifts.
ORDERING_MARGIN = 0.05

#: Seeds per draw in the design being sized. The tier's own count, so the
#: within-instance term enters the rule at the value it will actually have.
#: Read from the design rather than passed, because a pilot that sized the tier
#: at a seed count the tier does not run would be sizing a different experiment.
DESIGN_SEEDS = 100

#: Seeds per draw in the **pilot**, which is a different number for a stated
#: reason. The pilot spends its budget on draws, because ``sigma_w`` is already
#: well determined at ten seeds per instance across fifteen instances -- 135
#: degrees of freedom -- while ``sigma_b`` has only fourteen however many seeds
#: are run. Ten is the smallest count at which a single instance's own paired
#: comparison is still readable, which the printed table needs.
PILOT_SEEDS = 10

#: Landscape draws the pilot takes. The tier's three are in by construction
#: rather than by a literal repeating them, so the pilot covers the instances the
#: design already holds records for; the twelve above are new, contiguous, and
#: far from any seed the suite uses, so a pilot task can never collide with a
#: headline or replicate task's store key.
#:
#: Fifteen rather than more: the bound on ``sigma_b^2`` improves as
#: ``df / chi2_0.05(df)``, which falls steeply to about fourteen degrees of
#: freedom and slowly after -- 1.55x at fourteen against 1.36x at thirty, bought
#: for twice the compute.
PILOT_DRAWS: tuple[int, ...] = (*REPLICATION_SEEDS, *range(101, 113))

#: The most draws the design may ask for. Declared in advance so that "run more
#: instances" cannot become an open-ended search for a design that clears the
#: margin.
#:
#: Priced from the store rather than picked: one replicate draw is two protocol
#: shapes at 100 seeds over the tier's arms, which the stored headline tasks put
#: at about 22 core-hours, of which the GFlowNet arm alone is 14. Twelve draws is
#: therefore ~260 core-hours -- around a day at twelve concurrent processes, and
#: more than the entire suite has ever cost. Past that the replication tier costs
#: more than everything it replicates, which is not a design anyone runs.
DRAW_CAP = 12

#: Significance the unanimous per-instance verdict has to reach. Fixes the sign
#: floor, and the floor is the clause of the rule that no variance can argue
#: away.
SIGN_ALPHA = 0.05

#: What every arm is compared against here, matching the headline tables'
#: reference. The components being estimated are of a *paired difference*, so
#: they are a property of the pair and not of an arm, and a pilot that paired
#: against something the tier does not pair against would size the design for
#: comparisons nobody reports.
REFERENCE = "genetic"


def pilot_tasks(draws: Sequence[int] = PILOT_DRAWS) -> tuple[Task, ...]:
    """The replication tier's own tasks, at the pilot's draws.

    Built by `replication` rather than assembled here, which is the difference
    between measuring the design and measuring a lookalike of it. Every
    parameter but the generator's seed -- length, alphabet, motif structure,
    density, protocol, radius, anchor rule, attainability declaration -- comes
    from the function the tier itself calls.

    Args:
        draws: Generator seeds.

    Returns:
        One task per (shape, draw).
    """
    return replication(draws)


def pilot_arms() -> dict[str, Methodology]:
    """The arms the replication tier runs.

    The full baseline set plus whatever configuration is recorded as shipped,
    rather than a cheap subset. Sizing a design from a subset of the comparisons
    it has to serve is the assumption this whole script exists to replace: the
    variance components are a property of the *pair*, and a pooled figure taken
    over three cheap arms would be sized for three comparisons the paper does not
    lead with.

    Returns:
        Methodology by arm name.
    """
    base = shipped_base()
    return {**BASELINES, base.name: base.arm}


def instance_differences(
    store: ResultStore,
    tasks: Sequence[Task],
    arm: str,
    *,
    reference: str = REFERENCE,
    seeds: Sequence[int],
) -> list[np.ndarray]:
    """Per-seed paired differences against the reference, one array per instance.

    Paired within the instance and never across it. Two campaigns on different
    landscape draws share no wild type, no surrogate initialisation and no
    optimum, so a difference taken between them is not a paired difference at
    all -- it is the between-instance term the decomposition is trying to
    estimate, smuggled in as if it were noise.

    Args:
        store: Where records live.
        tasks: The instances, one task each.
        arm: The arm under test.
        reference: What it is paired against.
        seeds: Seeds to read, in order.

    Returns:
        One array per instance holding at least two shared seeds, oriented so a
        positive value means the arm beat the reference. Instances with fewer
        are dropped rather than padded, and the caller reports the count.
    """
    groups = []
    for task in tasks:
        mine = store.usable(task.name, arm)
        theirs = store.usable(task.name, reference)
        shared = [seed for seed in seeds if seed in mine and seed in theirs]
        # Regret, and lower is better, so the reference's regret minus the arm's
        # is positive when the arm won -- the orientation `compare` produces for
        # `higher_is_better=False`, restated here because `decompose` takes raw
        # differences and cannot know which way they point.
        first = records_to_metric(mine, shared, "regret")
        second = records_to_metric(theirs, shared, "regret")
        if first.size != second.size or first.size < 2:  # noqa: PLR2004
            continue
        difference = second - first
        if not np.isfinite(difference).all():
            continue
        groups.append(difference)
    return groups


@dataclass(frozen=True, slots=True)
class DrawPlan:
    """What the rule asks for, and what it reports when nobody can afford that.

    Attributes:
        components: The decomposition the count was taken from.
        margin: The ordering the design must resolve.
        seeds: Seeds per draw the design will run.
        floor: The sign-test floor, which no variance can move.
        cap: The declared maximum number of draws.
        required: What the rule asks for before the cap is applied. Kept rather
            than clamped away: a requirement nobody can afford is a finding, and
            a clamp would file it under "we chose the cap".
    """

    components: VarianceComponents
    margin: float
    seeds: float
    floor: int
    cap: int
    required: int

    @property
    def draws(self) -> int | None:
        """The declared draw count, or ``None`` when the cap binds."""
        return self.required if self.required <= self.cap else None

    @property
    def resolvable(self) -> float:
        """The smallest ordering a design at the cap can resolve, at 80% power.

        The residual, and the number to quote when no draw count was declared.
        A design that cannot confirm an ordering of this size cannot confirm one,
        and saying so beside the margin a headline claims is the whole of what an
        unaffordable requirement has to offer.
        """
        # Inverting the precision clause at `m = cap`: the count scales as
        # `(k * tau / delta)^2`, so the margin a given count reaches scales as
        # `1 / sqrt(count)` from the margin it was sized at.
        return self.margin * math.sqrt(self.required / self.cap)

    @property
    def reason(self) -> str:
        """Why this count, in a form that can be pasted into a caption."""
        shared = (
            f"between-instance sd {self.components.between:.4f} "
            f"(95% upper bound) against within-instance {self.components.within:.4f}, "
            f"ICC {self.components.intraclass_correlation:.3f}"
        )
        if self.draws is None:
            return (
                f"no draw count declared: resolving a {self.margin:g} ordering with the "
                f"instance as the unit needs {self.required} draws at {self.seeds:g} seeds, "
                f"above the declared cap of {self.cap}. At the cap the design resolves an "
                f"ordering of {self.resolvable:.4f} and no smaller one -- that bound is what "
                f"the tier reports, not a confirmation. From {shared}"
            )
        binding = "the sign floor" if self.required == self.floor else "precision"
        return (
            f"{self.required} landscape draws at {self.seeds:g} seeds each, set by {binding}: "
            f"resolves a {self.margin:g} ordering at 80% power with the instance as the unit "
            f"of replication, and a unanimous verdict over {self.required} draws reaches "
            f"sign-test p={unanimity_p(self.required, self.required):.4f}. From {shared}"
        )

    def __repr__(self) -> str:
        """Name the count and its standing."""
        where = (
            "uncapped requirement " + str(self.required) if self.draws is None else str(self.draws)
        )
        return f"draws={where} (margin {self.margin:g}, cap {self.cap})"


def plan(
    components: VarianceComponents,
    *,
    margin: float = ORDERING_MARGIN,
    seeds: float = DESIGN_SEEDS,
    cap: int = DRAW_CAP,
    alpha: float = SIGN_ALPHA,
) -> DrawPlan:
    """Apply the pre-declared rule to a measured decomposition.

    Args:
        components: What the pilot measured, normally conservative.
        margin: The smallest ordering the design must resolve.
        seeds: Seeds per draw in the design being sized.
        cap: The declared maximum.
        alpha: Significance the unanimous per-instance verdict must reach.

    Returns:
        The plan, carrying the uncapped requirement so an unaffordable answer
        stays visible.

    Raises:
        ValueError: If the margin exceeds `ORDERING_MARGIN`. The bound was fixed
            before any draw was run and may be tightened, never loosened -- a
            wider margin asks for fewer draws, which is the one edit that could
            turn any pilot into a design that clears.
    """
    if margin > ORDERING_MARGIN:
        raise ValueError(
            f"margin {margin} is looser than the declared {ORDERING_MARGIN}; the "
            f"bound was fixed before any draw was run and may be tightened, never "
            f"loosened, since a wider one asks for fewer draws"
        )
    floor = unanimity_floor(alpha)
    return DrawPlan(
        components=components,
        margin=margin,
        seeds=seeds,
        floor=floor,
        cap=cap,
        required=draws_needed(components, margin=margin, seeds=seeds, floor=floor),
    )


def measure(
    store: ResultStore,
    tasks: Sequence[Task],
    arms: Sequence[str],
    *,
    seeds: Sequence[int],
    reference: str = REFERENCE,
) -> dict[str, list[np.ndarray]]:
    """Every arm's per-instance paired differences against the reference.

    Returns the differences rather than the decomposition, because two
    decompositions are taken from them -- the conservative one the rule is
    applied to and the point estimate the pooling factor is quoted from -- and
    reading the store twice is how those two come to disagree about which
    instances were in.

    Args:
        store: Where records live.
        tasks: The pilot's tasks, of one protocol shape.
        arms: Arm names; the reference is skipped.
        seeds: Seeds to read.
        reference: What each arm is paired against.

    Returns:
        One list of per-instance arrays per arm, in the order given.
    """
    return {
        arm: instance_differences(store, tasks, arm, reference=reference, seeds=seeds)
        for arm in arms
        if arm != reference
    }


def describe(
    store: ResultStore,
    shape: str,
    tasks: Sequence[Task],
    arms: Sequence[str],
    *,
    seeds: Sequence[int],
) -> str:
    """Lay out one protocol shape: every arm's components, and the rule's verdict.

    The whole table is printed whatever the verdict, for the reason
    ``experiments/proxy_saturation.py`` prints its whole ladder: a pilot that
    found a large instance term and one that found none produce the same table
    and differ only in the last lines, which is what stops the reader from
    taking the verdict on trust.

    Args:
        store: Where records live.
        shape: The protocol shape, for the caption.
        tasks: Its instances.
        arms: Arm names.
        seeds: Seeds to read.

    Returns:
        The block, ready to print.
    """
    lines = [f"\n--- variance pilot: {shape} ({len(tasks)} draws x {len(seeds)} seeds) ---"]
    differences = measure(store, tasks, arms, seeds=seeds)
    bounded: dict[str, VarianceComponents] = {}
    estimated: dict[str, VarianceComponents] = {}
    for arm, groups in differences.items():
        if len(groups) < 2:  # noqa: PLR2004 - a between-instance variance needs two instances
            lines.append(
                f"  {arm:<34}  {len(groups)} of {len(tasks)} draws hold a pairable "
                f"campaign; two are needed before an instance term exists"
            )
            continue
        bounded[arm] = decompose(f"{arm} vs {REFERENCE}", groups)
        estimated[arm] = decompose(f"{arm} vs {REFERENCE}", groups, conservative=False)
        means = np.array([group.mean() for group in groups], dtype=np.float64)
        agreeing = int(max((means > 0).sum(), (means < 0).sum()))
        lines.append(f"  {bounded[arm]!r}")
        lines.append(
            f"      per draw {np.array2string(means, precision=3, floatmode='fixed')}  "
            f"same direction on {agreeing}/{len(means)} "
            f"(sign test p={unanimity_p(len(means), agreeing):.3f})"
        )

    if not bounded:
        lines.append("  nothing pairable is stored, so no decomposition can be read yet")
        return "\n".join(lines)

    label = f"{shape} pooled over {len(bounded)} arms"
    pooled = pool_components(label, list(bounded.values()))
    decision = plan(pooled)
    lines.append(f"  {pooled!r}")
    lines.append(
        f"  margin: the design must resolve an ordering of {ORDERING_MARGIN:g} regret; "
        f"sign floor {decision.floor} draws; cap {decision.cap}"
    )
    lines.append(f"  RULE: {decision.reason}")
    honest = pool_components(label, list(estimated.values()))
    lines.append(f"  {_pooling_note(honest, decision)}")
    lines.append(f"  {_seed_note(honest)}")
    return "\n".join(lines)


def _pooling_note(estimated: VarianceComponents, decision: DrawPlan) -> str:
    """What the pooled instance x seed reading would have claimed instead.

    Printed beside the design rather than left to the report, because this is the
    number that says *how bad* the pooled reading is on these landscapes. A
    factor near one would mean the framing's objection to pooling was formal; a
    factor of four means the pooled interval is a quarter of its true width and
    would read as far more evidence than the tier holds.

    Taken from the **point estimate** and not from the conservative bound the
    rule is sized on, and the two are not interchangeable here. Inflating
    ``sigma_b`` is the safe direction for choosing a draw count and the
    self-serving one for this figure: it would make the pooled reading look
    worse than it is, and a number quoted to argue against pooling has to be the
    one that is hardest to dismiss.
    """
    factor = estimated.design_effect(seeds=decision.seeds)
    return (
        f"pooling: treating instance x seed pairs as independent at {decision.seeds:g} seeds "
        f"would understate the standard error {factor:.2f}x, so a pooled interval would be "
        f"{1 / factor:.0%} of its true width -- and would look like more evidence, not less"
    )


def _seed_note(estimated: VarianceComponents) -> str:
    """Where seeds stop buying anything a replication design needs.

    From the point estimate, like the pooling factor and for the mirror-image
    reason: an inflated ``sigma_b`` would put this knee lower and so would say
    "run fewer seeds", which is the direction that costs precision rather than
    compute. The conservative bound is right for choosing a draw count and wrong
    for choosing a seed count, and the two are separated here rather than left
    to whoever reads the line.
    """
    between = estimated.between
    if between <= 0.0:
        return (
            "seeds: no instance term was resolved, so seeds are still the binding unit "
            "and the design is limited by the sign floor rather than by precision"
        )
    # The seed count past which the within-instance term is a tenth of the
    # instance term, so further seeds move the per-draw variance by under 5%.
    knee = math.ceil(10.0 * (estimated.within / between) ** 2)
    return (
        f"seeds: past ~{knee} per draw the within-instance term is a tenth of the "
        f"instance term, so seeds beyond that buy precision on the quantity a "
        f"replication design is not limited by"
    )


def _by_shape(tasks: Sequence[Task]) -> dict[str, list[Task]]:
    """Group the pilot's tasks by protocol shape.

    Read back through `replicate_instance` rather than parsed here, so a rename
    of the naming scheme cannot leave this grouping silently returning one
    family per task -- which would make every instance its own shape and every
    decomposition undefined.
    """
    families: dict[str, list[Task]] = {}
    for task in tasks:
        split = replicate_instance(task.name)
        if split is None:
            continue
        families.setdefault(split[0], []).append(task)
    return families


def _paired_within(store: ResultStore, task: Task, arm: str, seeds: Sequence[int]) -> str:
    """One instance's own comparison, so a printed draw can be read on its own."""
    mine = store.usable(task.name, arm)
    theirs = store.usable(task.name, REFERENCE)
    shared = [seed for seed in seeds if seed in mine and seed in theirs]
    first = records_to_metric(mine, shared, "regret")
    second = records_to_metric(theirs, shared, "regret")
    if first.size != second.size or first.size < 2:  # noqa: PLR2004
        return f"    {task.name}: {first.size} and {second.size} completed, not paired"
    return f"    {compare(task.name, first, second, higher_is_better=False)!r}"


def main(argv: list[str] | None = None) -> int:
    """Run the pilot, or read what is stored, and apply the rule to it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Read the store without running.")
    parser.add_argument("--results", default="results", help="Where results live.")
    parser.add_argument(
        "--seeds",
        type=int,
        default=PILOT_SEEDS,
        help="Seeds per draw. Low on purpose: the pilot spends its budget on "
        "draws, since the within-instance term is well determined almost at "
        "once and the between-instance term is not.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Restrict to these arms. The full set is the default because the "
        "components are a property of a pair, and a design sized on a subset is "
        "sized for comparisons the paper does not lead with.",
    )
    parser.add_argument(
        "--per-draw",
        action="store_true",
        help="Also print each draw's own paired comparison, which is what an "
        "ordering flip is a claim about.",
    )
    args = parser.parse_args(argv)

    configure_determinism()
    if not args.report and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    store = ResultStore(Path(args.results))
    arms = pilot_arms()
    if args.method:
        unknown = [name for name in args.method if name not in arms]
        if unknown:
            print(f"no pilot arm named {sorted(unknown)}", file=sys.stderr)
            return 2
        # The reference is kept whatever the filter says: every quantity here is
        # a difference against it, so dropping it leaves nothing to decompose.
        arms = {name: arms[name] for name in (*args.method, REFERENCE) if name in arms}

    seeds = tuple(range(args.seeds))
    tasks = pilot_tasks()
    started = time.perf_counter()
    if not args.report:
        run_tier(
            # `Purpose.SELECTION`, not `DIAGNOSTIC`: this fixes our own
            # experimental design rather than describing how methods behave, and
            # a tier that chooses the design is the one thing that must never be
            # eligible to appear in a results table.
            Tier("variance-pilot", tuple(tasks), seeds, Purpose.SELECTION),
            arms,
            store,
            report=_flush,
        )

    for shape, family in _by_shape(tasks).items():
        _flush(describe(store, shape, family, list(arms), seeds=seeds))
        if args.per_draw:
            for arm in arms:
                if arm == REFERENCE:
                    continue
                _flush(f"  {arm} vs {REFERENCE}, per draw:")
                for task in family:
                    _flush(_paired_within(store, task, arm, seeds))
    _flush(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


def _flush(message: str) -> None:
    """Print immediately, so a long pilot can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
