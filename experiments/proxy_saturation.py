"""Measure the proxy budget at which each pipeline's own returns saturate.

Proxy spend is a column the results table prints, and until now it printed a
number nobody measured. A GFlowNet round costs ``steps x TRAINING_BATCH``
surrogate evaluations; the ``genetic+search`` ablation's inner loop costs
``generations x population``. Both were chosen by us. The decision this script
implements is that each pipeline gets the budget at which *its own* returns
saturate, that the point is measured, and that the number is reported -- not
hard-equalised across pipelines, and not inherited.

The existing evidence supports no saturation claim at all. The joint screen in
[evogfn.benchmark.selection][] found ``steps`` to be the only axis that moved
regret, and it improved monotonically to 450, which was the top of that grid.
There is no measured knee anywhere in the store, so the proxy column as it
stands is a number we picked.

What runs
---------

**The ``steps`` ladder**, at the shipped configuration of each screened
objective, on a doubling grid that contains the shipped 300 and reaches two
rungs below it and three above. Two rungs below because the question is "how
little is enough" and an answer of 150 halves a reported column; three above
because a curve pinned at the top of its own grid cannot say whether it
flattened.

Both objectives get the ladder, and the second one is not decoration. Stage A
compared six objectives at one shared step count, which is fair only if 300
sits in the same place on every objective's return curve. Refitting the
screen's own step means says it does not: ``gfn-subtb`` looks flat above 150
while ``genetic-gfn`` is still improving at 450. If those knees really are a
factor of four apart then stage A compared a saturated arm against a starved
one, and "sub-trajectory balance was selected" becomes "sub-trajectory balance
was selected at a budget that suited it". That is the outcome to find here
rather than at review.

**The ``generations`` ladder** for ``genetic+search``, whose polarity is the
opposite one and therefore the more dangerous. An unsaturated GFlowNet curve
understates our own method; an unsaturated comparator curve manufactures a win.
There is also live reason to expect this curve is not monotone at all -- on
three of the four headline tasks in the store ``genetic+search`` is *worse* than
plain ``genetic``, so optimising against a surrogate fitted to a few hundred
assays can and apparently does hurt. The rule handles a worsening curve
correctly without amendment, but the reading is proxy over-optimisation rather
than saturation, and the printed lines are what distinguish them.

Where the decision lives, and where it must not
-----------------------------------------------

The rule is in [evogfn.benchmark.saturation][], written down before a single
rung was run, and it is an *equivalence* bound: saturation is declared only when
the 95% upper bound on the gain from doubling falls under the margin. Too few
seeds widens that interval and the rule refuses -- underpowering fails toward
"not saturated", which costs compute rather than shrinking a reported column.

Everything here runs on ``objective_task()``, the diagnostic landscape no
headline task uses, so the budget is not being chosen on the test set. The one
exception is the twin precondition below, which runs on ``feasibility`` because
that is where the records it must reproduce live; it decides nothing and its
numbers are never reported.

Why this is a script and not a tier of `run_suite`
--------------------------------------------------

The precedent is ``experiments/select_configuration.py``: a selection-phase
measurement that runs its own arms on the diagnostic task under
`Purpose.SELECTION` without appearing in the suite. Adding a tier would put a
curve that fixes our own configuration into the same ``--report`` output as the
headline tables, which is the confusion `Purpose` exists to prevent. It would
also cost more than it looks: the store's dependency closure covers
``benchmark.methods``, ``benchmark.selection``, ``benchmark.suite`` and
``loop.campaign``, so editing any of them restamps all 15,971 stored records --
about 185 core-hours. Nothing under ``experiments/`` is in the closure, and the
`Configuration` API already expresses every rung of the ``steps`` ladder, so
that ladder needs no source change whatsoever.

    uv run python experiments/proxy_saturation.py --report       # read, no runs
    uv run python experiments/proxy_saturation.py --ladder steps
    uv run python experiments/proxy_saturation.py --ladder generations
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evogfn.algorithms.inner_loop import ProxyOptimising
from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.methods import (
    DEFAULT_POOL,
    Arm,
    _anchor_seed,
    _campaign,
    _genetic,
    _parts,
)
from evogfn.benchmark.saturation import (
    INDIFFERENCE_MARGIN,
    SATURATION_SEEDS,
    Saturation,
    Verdict,
    pooled_spread,
    required_seeds,
    saturation,
)
from evogfn.benchmark.selection import Configuration, screen_arms
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import MAIN, Purpose, Tier, objective_task, run_task, run_tier
from evogfn.surrogate.proxy import ProxyLandscape

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from evogfn.algorithms.base import Sampler
    from evogfn.benchmark.methods import Methodology
    from evogfn.benchmark.store import RunRecord
    from evogfn.benchmark.tasks import Task
    from evogfn.env.mutation import MutationEnvironment
    from evogfn.loop.campaign import Campaign

#: Gradient steps per round: the GFlowNet's proxy budget, and the axis the joint
#: screen found to be the only one that moved regret.
#:
#: Ratio spacing at factor two, anchored on the shipped 300 rather than on round
#: decimal numbers, so the reported configuration is *on* the grid instead of
#: beside it. Doubling is the largest ratio that still localises a knee to
#: within a factor of two; anything finer shrinks the per-rung effect below what
#: an affordable seed count can bound, which is the failure this whole design
#: exists to avoid.
#:
#: 450 -- the old screen's top rung -- is deliberately not here. It is 1.5x the
#: shipped value, off the doubling ladder, and joining the old grid is not worth
#: breaking the spacing for.
#:
#: **2400 is a declared ceiling, not "as far as we got".** It is fixed before
#: any rung was run, and `evogfn.benchmark.saturation` states what is reported
#: if the curve is still moving there: a cost ceiling with a measured residual,
#: rather than a saturation point. No rung is added in response to that outcome.
STEPS: tuple[int, ...] = (75, 150, 300, 600, 1200, 2400)

#: Inner generations per round for ``genetic+search``, on the same doubling
#: ladder anchored on the shipped 50 (12 and 25 round 12.5 and 25 to integers).
#:
#: Unlike the ``steps`` ladder there is no free rung here: ``genetic+search``
#: has no records on the diagnostic task at all, since that tier is
#: GFlowNet-only. All six rungs run from scratch, and they run unconditionally
#: rather than under the sequential stopping rule, because the whole ladder is
#: about an hour of wall clock.
GENERATIONS: tuple[int, ...] = (12, 25, 50, 100, 200, 400)

#: The shipped configuration of each objective that gets a ladder, read from
#: ``results/selected.json``'s confirmed arms. Everything except ``steps`` is
#: pinned here, so the curve is a curve in the knob and not in the whole space.
LADDER: dict[str, dict[str, float | int]] = {
    # The shipped arm. Its 300-step rung *is* the stored
    # `gfn-subtb@b0.1-s300-l0.9-h64`, which the confirmation already paid 100
    # seeds for; it is reused rather than re-declared, and only topped up.
    "gfn-subtb": {"beta": 0.1, "lam": 0.9, "hidden_dim": 64},
    # Not shipped -- `methods_for` replaces the untuned GFlowNet arms with the
    # selected one -- so this curve is not a reported proxy column. It is the
    # best of the five `genetic-gfn` finalists, and it is here because the
    # objective comparison assumed a shared budget.
    "genetic-gfn": {"beta": 0.5, "mix": 0.25, "hidden_dim": 64},
}

#: Seeds for the ``generations`` ladder. Higher than the GFlowNet ladders'
#: because it is affordable outright -- the whole ladder is ~14 core-hours -- and
#: there is no reason to measure a comparator at lower resolution than our own
#: method.
GENERATION_SEEDS = 500

#: The task whose stored ``genetic+search`` records the script-local twin has to
#: reproduce, and the seeds it reproduces them on. Ten campaigns, about three
#: minutes, and bit-for-bit equality on a stored record is a stronger guarantee
#: than reading two definitions side by side.
TWIN_TASK = "feasibility"
TWIN_SEEDS = 10

#: The shipped inner-loop budget the twin is verified at. Kept as its own name
#: rather than read from ``DEFAULT_GENERATIONS``: the point of the check is that
#: the twin agrees with what the *store* holds, and reading the constant the
#: twin also reads would make the check agree with itself.
SHIPPED_GENERATIONS = 50

#: Separates the ``genetic+search`` twin from the shipped arm in the store. The
#: shipped arm's key is ``genetic+search``; every rung here carries ``@g{N}``, so
#: no rung can ever be read as the shipped arm and the shipped arm's records are
#: never written over.
_TWIN_PREFIX = "genetic+search"


def steps_rungs(objective: str) -> tuple[Configuration, ...]:
    """The ``steps`` ladder for one objective, as configurations.

    No source change is needed for any of this: `Configuration` already
    expresses every point of the grid and `screen_arms` already builds them, so
    the ladder is a script-local tuple outside the store's dependency closure.

    Args:
        objective: A key of `LADDER`.

    Returns:
        One configuration per rung, ascending.

    Raises:
        KeyError: If the objective has no pinned configuration here.
    """
    pinned = LADDER[objective]
    return tuple(
        Configuration(
            objective=objective,
            beta=float(pinned["beta"]),
            steps=steps,
            hidden_dim=int(pinned["hidden_dim"]),
            lam=None if "lam" not in pinned else float(pinned["lam"]),
            mix=None if "mix" not in pinned else float(pinned["mix"]),
        )
        for steps in STEPS
    )


def steps_arms(objective: str) -> dict[str, Methodology]:
    """The ``steps`` ladder for one objective, as runnable arms by name."""
    return screen_arms(steps_rungs(objective))


def genetic_search(generations: int) -> Arm:
    """``genetic+search`` at a chosen inner-loop budget: a script-local twin.

    ``ProxyOptimising``'s ``generations`` is fixed at construction inside
    `classical`'s ``make()`` and `classical` does not expose it, so the knob
    does not exist in the source. Adding it there is the single-definition
    answer and it is the right one *if the answer turns out to be a different
    value* -- but ``DEFAULT_GENERATIONS`` lives in ``algorithms/inner_loop.py``,
    which is inside ``benchmark.methods``'s import closure, so changing that one
    integer stales all 15,971 stored records and about 185 core-hours of
    compute. The measurement runs against this twin first and the blast radius
    is deferred to the case where it is actually earned.

    The recurring failure of a twin is that it drifts from the arm it claims to
    be, so this one does not restate anything it can import: the sampler builder
    is `methods._genetic` itself, the anchor seeding is `methods._anchor_seed`,
    and the campaign is assembled by `methods._campaign`. What is left is the
    one line `classical` writes and this rewrites -- passing ``generations``
    through. `verify_twin` closes the rest by running it at the shipped budget
    and demanding bit-for-bit agreement with the store.

    Args:
        generations: Inner iterations per round against the proxy.

    Returns:
        The arm, carrying `classical`'s own parameters for the record plus the
        ``generations`` it ran at -- which is the field the shipped arm's
        records cannot have, and the reason a rung is legible from its record
        alone rather than only from its name.

    Raises:
        ValueError: If ``generations`` is not positive.
    """
    if generations < 1:
        raise ValueError(f"generations must be at least 1, got {generations}")

    def methodology(task: Task, seed: int) -> Campaign:
        """Build one campaign, exactly as `classical` builds the shipped arm."""
        landscape, env, ensemble = _parts(task, seed)
        proxy = ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)
        generation = count()

        def make(anchored: MutationEnvironment) -> Sampler:
            """Rebuild the baseline at whichever anchor the campaign is at."""
            sampler = _genetic(anchored, _anchor_seed(seed, next(generation)), task.protocol)
            return ProxyOptimising(sampler, proxy=proxy, generations=generations)

        return _campaign(task, landscape, env, make, ensemble, pool_size=DEFAULT_POOL)

    return Arm(
        methodology,
        {
            "family": "classical",
            "sampler": "genetic",
            "surrogate": True,
            "proxy_access": True,
            "pool_size": DEFAULT_POOL,
            "distinct_batch": False,
            "acquisition": "Greedy",
            "bootstrap": False,
            "extra_rounds": 0,
            "generations": generations,
        },
    )


def generation_arms() -> dict[str, Methodology]:
    """The ``generations`` ladder as runnable arms by name."""
    return {twin_name(g): genetic_search(g) for g in GENERATIONS}


def twin_name(generations: int) -> str:
    """The store key for one rung of the ``generations`` ladder."""
    return f"{_TWIN_PREFIX}@g{generations}"


def twin_task() -> Task:
    """The headline task whose stored records the twin has to reproduce.

    Raises:
        KeyError: If the suite no longer defines it, which would mean the
            records the check reads against are gone too.
    """
    for task in MAIN:
        if task.name == TWIN_TASK:
            return task
    raise KeyError(f"{TWIN_TASK!r} is not a task in the suite any more")


def verify_twin(
    store: ResultStore,
    *,
    seeds: int = TWIN_SEEDS,
    report: Callable[[str], None] = print,
) -> bool:
    """Prove the twin *is* ``genetic+search`` at the shipped budget, seed for seed.

    A hard precondition of the ``generations`` ladder rather than a test that
    runs somewhere else, because the failure it guards is not a crash: a twin
    that has drifted still runs, still produces plausible regrets, and reports
    them under a name claiming to be the published ablation. Ten campaigns and
    an exact equality is a stronger guarantee than reading two definitions side
    by side, which is how the drift got in.

    ``regret``, ``best`` and ``proxy_calls`` are compared exactly, not to a
    tolerance. These campaigns are deterministic given their seed, so anything
    other than equality means the two pipelines differ somewhere, and a
    tolerance would decide how much difference is acceptable -- which is a
    question nobody should be answering here.

    Args:
        store: Where the shipped arm's records live, and where the twin's go.
        seeds: How many seeds to reproduce.
        report: Where progress lines go.

    Returns:
        Whether the twin matched on every seed compared.
    """
    task = twin_task()
    shipped = store.usable(task.name, _TWIN_PREFIX)
    wanted = [seed for seed in range(seeds) if seed in shipped]
    if not wanted:
        report(
            f"  twin check: no usable {_TWIN_PREFIX} records on {task.name}; there "
            f"is nothing to reproduce, so the ladder cannot be cleared to run"
        )
        return False

    name = twin_name(SHIPPED_GENERATIONS)
    run_task(task, {name: genetic_search(SHIPPED_GENERATIONS)}, store, wanted, report=report)
    mine = store.usable(task.name, name)

    mismatched = []
    for seed in wanted:
        if seed not in mine:
            mismatched.append(f"seed {seed}: the twin produced no usable record")
            continue
        theirs, ours = shipped[seed], mine[seed]
        for field in ("regret", "best", "proxy_calls"):
            if getattr(theirs, field) != getattr(ours, field):
                mismatched.append(
                    f"seed {seed}: {field} {getattr(ours, field)!r} against the "
                    f"stored {getattr(theirs, field)!r}"
                )
    for line in mismatched:
        report(f"  TWIN MISMATCH  {line}")
    if mismatched:
        report(
            f"  the twin is not {_TWIN_PREFIX}; the generations ladder does not run, "
            f"because its rungs would be a pipeline nothing in the table describes"
        )
        return False
    report(
        f"  twin check: {len(wanted)} seeds on {task.name} reproduce "
        f"{_TWIN_PREFIX} exactly at generations={SHIPPED_GENERATIONS}"
    )
    return True


def measured(
    store: ResultStore,
    task: str,
    arms: Mapping[int, str],
) -> tuple[dict[int, dict[int, float]], dict[int, int]]:
    """Per-seed regret by budget, and how many seeds each rung lost.

    A seed is dropped when its campaign exhausted or its regret is not finite.
    This departs from `evogfn.benchmark.selection`'s rule, which makes an arm
    that exhausted on any shared seed ineligible outright, and the departure is
    deliberate: that rule chooses a configuration to ship and one that cannot
    finish is not one, while this rule reads a *curve* and losing a whole rung
    to one failed seed would leave a gap in the ladder the rule then reads as a
    doubling it is not.

    The dropped count is returned rather than swallowed because the bias has a
    direction. Seeds are hard for a reason, so dropping the ones a *larger*
    budget failed on flatters the larger budget -- which is the direction that
    would make a knee look higher than it is.

    Args:
        store: Where the records are.
        task: The task name they were run on.
        arms: Arm name by budget.

    Returns:
        Regret per seed per budget, restricted to rungs that hold anything, and
        the dropped-seed count per budget.
    """
    regret: dict[int, dict[int, float]] = {}
    dropped: dict[int, int] = {}
    for budget, name in arms.items():
        held = store.usable(task, name)
        usable = {
            seed: float(record.regret)
            for seed, record in held.items()
            if record.regret is not None and not record.exhausted and math.isfinite(record.regret)
        }
        if usable:
            regret[budget] = usable
        dropped[budget] = len(held) - len(usable)
    return regret, dropped


def _rung_line(
    budget: int,
    name: str,
    records: Mapping[int, RunRecord],
    dropped: int,
) -> str:
    """One rung of the printed curve: what it cost and what it bought."""
    scored = [
        float(r.regret)
        for r in records.values()
        if r.regret is not None and not r.exhausted and math.isfinite(r.regret)
    ]
    if not scored:
        return f"  {budget:>5}  {name:<34}  nothing usable stored"
    values = np.array(scored, dtype=np.float64)
    error = float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else float("nan")
    # Measured rather than computed from `steps x TRAINING_BATCH`: a
    # `genetic-gfn` arm also spends proxy calls breeding, so the closed form
    # holds for the plain GFlowNet arms and understates the bred ones.
    proxy = int(statistics.median(int(r.proxy_calls) for r in records.values()))
    wall = statistics.median(float(r.wall_seconds) for r in records.values())
    lost = f"  {dropped} seed(s) dropped" if dropped else ""
    return (
        f"  {budget:>5}  {name:<34}  regret {values.mean():.4f} +-{error:.4f}  "
        f"n={values.size:<4} proxy {proxy:>8,}  {wall:6.1f}s/campaign{lost}"
    )


def describe(  # noqa: PLR0913 - a printed ladder is defined by what it varied
    axis: str,
    label: str,
    store: ResultStore,
    task: str,
    arms: Mapping[int, str],
    *,
    ceiling: int,
    margin: float = INDIFFERENCE_MARGIN,
    planned: int = SATURATION_SEEDS,
) -> str:
    """Lay out one ladder: every rung, every doubling, and the rule's verdict.

    The whole curve is printed whatever the verdict, in the same terms the
    selection grids fix their own. A ladder that flattened and one that never
    did produce the same table and differ only in the last two lines, which is
    what stops the reader from having to take the verdict on trust.

    Args:
        axis: The knob varied, for the caption.
        label: What was varied it on -- an objective, or an arm family.
        store: Where the records are.
        task: The task they ran on.
        arms: Arm name by budget.
        ceiling: The declared top of the grid.
        margin: The indifference margin.
        planned: The seed count the ladder was run at.

    Returns:
        The block, ready to print.
    """
    regret, dropped = measured(store, task, arms)
    lines = [f"\n--- {axis} ladder: {label} (on {task}) ---"]
    for budget in sorted(arms):
        held = store.usable(task, arms[budget])
        if not held:
            # "Not run" and "run under code that has since moved" are different
            # states and only the second is recoverable by a bless. A rung that
            # printed the first while holding a hundred stale campaigns would
            # send someone to re-run compute that already exists.
            stale = store.stale(task, arms[budget])
            standing = f"{len(stale)} stored but stale" if stale else "not run"
            lines.append(f"  {budget:>5}  {arms[budget]:<34}  {standing}")
            continue
        lines.append(_rung_line(budget, arms[budget], held, dropped.get(budget, 0)))

    if len(regret) < 2:  # noqa: PLR2004 - a doubling needs two rungs
        lines.append("  fewer than two rungs are stored, so no doubling can be read yet")
        return "\n".join(lines)

    verdict = saturation(regret, axis=axis, margin=margin, ceiling=ceiling)
    lines.append(f"  margin: a doubling must be bounded below {margin:g} regret to count")
    lines.extend(f"  {pair!r}" for pair in verdict.doublings)
    lines.append(f"  RULE: {verdict.reason}")
    lines.append(f"  {_power_note(verdict, planned=planned)}")
    return "\n".join(lines)


def _power_note(verdict: Saturation, *, planned: int) -> str:
    """What the realised spread says about the seed count, and nothing else.

    Reads the variance and never the effect, which is what keeps the top-up
    from being optional stopping dressed as diligence.
    """
    spread = pooled_spread(verdict.doublings)
    needed = required_seeds(spread, margin=verdict.margin, planned=planned)
    unresolved = [pair for pair in verdict.doublings if pair.verdict is Verdict.INCONCLUSIVE]
    if needed <= planned:
        return (
            f"seeds: pooled paired spread {spread:.3f} at {planned} seeds is within "
            f"what was planned for; no top-up ({len(unresolved)} doubling(s) unresolved)"
        )
    return (
        f"seeds: pooled paired spread {spread:.3f} exceeds what {planned} seeds "
        f"resolves; the pre-declared top-up runs to {needed} "
        f"({len(unresolved)} doubling(s) unresolved)"
    )


def _stop_after(  # noqa: PLR0913 - the rule reads a ladder, a store and a bound
    store: ResultStore,
    task: str,
    arms: Mapping[int, str],
    ceiling: int,
    *,
    seeds: int,
    margin: float,
) -> int | None:
    """The highest rung worth running, given what is already stored.

    The sequential stopping rule, and the reason it inflates nothing: the bound
    it stops on is an *equivalence* bound, so stopping happens only when a bound
    was already met rather than when a search for significance succeeded. The
    "two consecutive bounded doublings" requirement is what forces one
    confirming rung above the declared knee, so no rung the decision rests on is
    ever skipped.

    It reads the *store* rather than this process's own runs, and it declines to
    stop unless every lower rung holds the full seed count -- so a shard that
    starts before its siblings finish runs the whole grid rather than stopping
    on a partial ladder. Pin ``--upto`` to override that.

    Args:
        store: Where the records are.
        task: The task they ran on.
        arms: Arm name by budget.
        ceiling: The declared top of the grid.
        seeds: The seed count a rung must hold to count as complete.
        margin: The indifference margin.

    Returns:
        The budget to stop at, or ``None`` when the whole grid is still owed.
    """
    regret, _ = measured(store, task, arms)
    complete = {budget: held for budget, held in regret.items() if len(held) >= seeds}
    for top in sorted(complete):
        below = {budget: held for budget, held in complete.items() if budget <= top}
        if len(below) < 3:  # noqa: PLR2004 - two consecutive doublings need three rungs
            continue
        verdict = saturation(below, margin=margin, ceiling=ceiling)
        if verdict.measured and verdict.settled:
            return top
    return None


def run_ladder(  # noqa: PLR0913, PLR0917 - a ladder is named, scoped, and told what to run
    axis: str,
    label: str,
    arms: Mapping[int, str],
    build: Mapping[str, Methodology],
    store: ResultStore,
    args: argparse.Namespace,
    *,
    seeds: int,
    runnable: Sequence[int],
    sequential: bool,
) -> None:
    """Run a ladder's rungs bottom-up, stopping where the rule says it may.

    Args:
        axis: The knob varied.
        label: What it was varied on.
        arms: Arm name by budget, ascending.
        build: Methodology by arm name.
        store: Where campaigns are written.
        args: The parsed command line, for ``--only`` and ``--upto``.
        seeds: The seed count a rung must hold before stopping is considered.
        runnable: The seeds this process runs.
        sequential: Whether the stopping rule applies. False for a ladder cheap
            enough to run whole, where stopping would buy nothing and would
            leave the curve with rungs missing above the knee.
    """
    ceiling = max(arms)
    for budget in sorted(arms):
        if args.upto is not None and budget > args.upto:
            _flush(f"  {axis}={budget}: above --upto {args.upto}, not run")
            continue
        if sequential:
            stop = _stop_after(
                store, objective_task().name, arms, ceiling, seeds=seeds, margin=args.margin
            )
            if stop is not None and budget > stop:
                _flush(
                    f"  {axis}={budget}: the rule already settled at {stop} on "
                    f"{seeds} seeds, so this rung is not needed"
                )
                continue
        name = arms[budget]
        if args.only and name not in args.only:
            continue
        tier = Tier(
            f"proxy-saturation-{axis}", (objective_task(),), tuple(runnable), Purpose.SELECTION
        )
        run_tier(tier, {name: build[name]}, store, report=_flush)
    _flush(
        describe(
            axis,
            label,
            store,
            objective_task().name,
            arms,
            ceiling=ceiling,
            margin=args.margin,
            planned=seeds,
        )
    )


def _steps_ladders() -> dict[str, dict[int, str]]:
    """Arm name by budget, per objective, for the ``steps`` ladder."""
    return {
        objective: {rung.steps: rung.name for rung in steps_rungs(objective)}
        for objective in LADDER
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ladders this process is asked for, and print both curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Read the store without running.")
    parser.add_argument("--results", default="results", help="Where results live.")
    parser.add_argument(
        "--ladder",
        choices=("steps", "generations", "both"),
        default="both",
        help="Which axis to run. The two are independent and cost very "
        "differently, so they shard separately.",
    )
    parser.add_argument(
        "--objective",
        action="append",
        default=[],
        help="Restrict the steps ladder to these objectives. Both run by "
        "default: the objective comparison assumed a shared budget, and only "
        "running both can say whether that held.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=SATURATION_SEEDS,
        help="Seeds per rung of the steps ladder. Every decision here is a "
        "paired comparison and the measured paired spread is 0.15-0.20, so the "
        "half-width at 100 seeds is twice the indifference margin -- which is "
        "the regime where an underpowered curve reads as a flat one.",
    )
    parser.add_argument(
        "--generation-seeds",
        type=int,
        default=GENERATION_SEEDS,
        help="Seeds per rung of the generations ladder, which is cheap enough "
        "to run at higher resolution than the GFlowNet ladders.",
    )
    parser.add_argument("--seed-from", type=int, default=0, help="First seed this process runs.")
    parser.add_argument("--seed-to", type=int, default=None, help="One past the last seed.")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run just these arm names. Campaigns are independent and the store "
        "keeps one file per arm, so a sharded run and a serial one produce "
        "identical records.",
    )
    parser.add_argument(
        "--upto",
        type=int,
        default=None,
        help="Highest rung this process runs. Pins what the sequential stopping "
        "rule would otherwise decide from a store the shards are still filling.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=INDIFFERENCE_MARGIN,
        help="The indifference margin. May be tightened and never loosened -- "
        "the rule refuses a value above the one declared before any rung ran, "
        "since widening it once the curve exists would let a knee be declared "
        "anywhere on it.",
    )
    args = parser.parse_args(argv)

    configure_determinism()
    if not args.report and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    store = ResultStore(Path(args.results))
    started = time.perf_counter()
    ladders = _steps_ladders()
    unknown = [name for name in args.objective if name not in ladders]
    if unknown:
        print(f"no ladder is declared for {sorted(unknown)}", file=sys.stderr)
        return 2
    wanted = args.objective or sorted(ladders)

    if args.ladder in {"steps", "both"}:
        runnable = range(args.seed_from, args.seed_to or args.seeds)
        for objective in wanted:
            arms = ladders[objective]
            if args.report:
                _flush(
                    describe(
                        "steps",
                        objective,
                        store,
                        objective_task().name,
                        arms,
                        ceiling=max(arms),
                        margin=args.margin,
                        planned=args.seeds,
                    )
                )
                continue
            run_ladder(
                "steps",
                objective,
                arms,
                steps_arms(objective),
                store,
                args,
                seeds=args.seeds,
                runnable=list(runnable),
                sequential=True,
            )

    if args.ladder in {"generations", "both"}:
        arms = {generations: twin_name(generations) for generations in GENERATIONS}
        if args.report:
            _flush(
                describe(
                    "generations",
                    _TWIN_PREFIX,
                    store,
                    objective_task().name,
                    arms,
                    ceiling=max(arms),
                    margin=args.margin,
                    planned=args.generation_seeds,
                )
            )
        elif not verify_twin(store, report=_flush):
            print("the generations ladder did not run: see TWIN MISMATCH above", file=sys.stderr)
            return 4
        else:
            run_ladder(
                "generations",
                _TWIN_PREFIX,
                arms,
                generation_arms(),
                store,
                args,
                seeds=args.generation_seeds,
                runnable=list(range(args.seed_from, args.seed_to or args.generation_seeds)),
                # Cheap enough to run whole, and a comparator's curve with rungs
                # missing above the knee is the one an unsaturated reading would
                # hide behind.
                sequential=False,
            )

    _flush(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


def _flush(message: str) -> None:
    """Print immediately, so a long ladder can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
