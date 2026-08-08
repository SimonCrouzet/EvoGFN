"""When more proxy budget stops buying anything, decided before the numbers.

Proxy spend is a knob *we* chose, not a constant of an architecture. A
GFlowNet round costs ``steps x TRAINING_BATCH`` surrogate evaluations and the
``genetic+search`` ablation's inner loop costs ``generations x population``;
both numbers were picked by us and both are printed in the results table. A
column reporting an inherited number says nothing, and the decision taken is
that each pipeline gets the budget at which *its own* returns saturate, that the
point is measured, and that the number is reported.

This module is the rule that reads a budget ladder and says where -- or whether
-- that happened. It is written down here before any rung has been run, for the
same reason [evogfn.benchmark.selection][] writes its rule down before its
stages run: a knee read off a plot after the plot exists is not a criterion, and
"the curve looked flat from about here" is how a chosen budget becomes a story.

The rule
--------

> **The saturation point is the smallest rung ``s`` on the ladder such that the
> doubling above ``s`` buys less than the indifference margin, and so does every
> doubling above that one.**

"Buys less than the margin" is an *equivalence* claim and not a null result:
the 95% upper bound on the paired gain must fall below `INDIFFERENCE_MARGIN`.
"We failed to find a gain" is the other statement, and it is the one this rule
exists instead of.

Why it cannot be gamed by running too few seeds
-----------------------------------------------

This is the property that makes the whole design safe rather than merely
argued, so it is stated as a direction rather than as a caveat. Fewer seeds
means a wider interval, means a *higher* upper bound, means the bound is not
met and no saturation is declared. **Underpowering fails toward "not
saturated"** -- the direction that costs compute rather than the direction that
shrinks a reported column. A rule phrased on "the difference was not
significant" has exactly the opposite polarity, and would hand a flat-looking
answer to anyone who ran too few seeds.

Three more things are load-bearing and are enforced below rather than promised:

* **The margin can be tightened and never loosened.** `saturation` refuses a
  ``margin`` above `INDIFFERENCE_MARGIN`. Re-deciding the indifference bound
  upward once the curve is visible is the one edit that could manufacture a
  knee anywhere, and it raises instead.
* **The ladder must be a doubling ladder.** Adjacent rungs are checked to be a
  factor of two apart, so a rung cannot be inserted between two others after
  the fact to produce a small step that clears the bound. Rungs may still be
  *appended above*, and that direction is safe by construction: the "and every
  doubling above" clause means a new top rung can only ever revoke a
  declaration, never lower it.
* **The top-up trigger cannot see the effect.** `required_seeds` takes a
  spread and nothing else -- not a mean, not a sign -- so an internal pilot
  cannot be run until the answer is agreeable. It is variance-triggered, which
  is what keeps it from being optional stopping.

What the rule does *not* say
----------------------------

It reports a budget, not a reading. A gain whose mean is negative -- more
budget is worse, which is live for a sampler optimising against a surrogate
fitted to a few hundred assays -- also has a small upper bound, so the rule
fires at a low rung. That is the right budget (take the cheapest rung nothing
larger beats) and the wrong *word*: it is proxy over-optimisation, not a
plateau. `Doubling` therefore carries the mean, the interval and the win rate
for every adjacent pair rather than only for the one that fired, and the two
readings are distinguished by whoever writes the caption.

When the knee is above the grid
-------------------------------

Declared here rather than improvised then: if the top doubling on the ladder
does not clear the bound, **no saturation point has been measured**. The
reported budget is a cost ceiling we chose, and what is reported in its place
is `Saturation.residual` -- "at the top rung, doubling the proxy budget again
buys at most this much regret". That bound is the more useful number anyway,
since the claim the paper needs is that its ranking does not turn on the budget,
and a residual set beside the headline margin answers that with or without a
knee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.statistics import compare

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from evogfn.benchmark.statistics import PairedComparison

#: The indifference margin: a regret difference below this cannot move anything
#: the paper prints, so buying it is not worth reporting a budget for.
#:
#: Fixed here, before any rung has been run, on three independent grounds that
#: happen to agree. It is the precision the headline table is *printed* at --
#: the reported ``regret +- error`` carries a standard error of 0.014-0.018 for
#: these arms, so a difference below 0.02 cannot move a reported comparison. It
#: is 5% of the shipped arm's regret on the diagnostic landscape. And it is
#: below every paired difference the selection rule has ever acted on.
#:
#: `saturation` refuses a margin larger than this and accepts a smaller one, so
#: the bound can be tightened and never loosened.
INDIFFERENCE_MARGIN = 0.02

#: Seeds per rung, planned rather than inherited.
#:
#: The floor that matters here is **not** the standard error of one arm's mean.
#: Every decision runs through a paired comparison, and the measured paired
#: difference standard deviation on the diagnostic landscape is 0.15 for
#: near-identical arms and 0.196 for the one stored pair differing solely in
#: ``steps`` -- seed correlation is +0.52 for the first and only +0.16 for the
#: second, so common random numbers cancel least exactly where a budget ladder's
#: rungs are furthest apart. At 0.20 the 95% half-width at 100 seeds is 0.039,
#: which is twice the margin: every doubling that decides this question would
#: come back inconclusive, and an underpowered curve reads as a flat one.
#:
#: 250 rather than the ~392 that sd = 0.20 implies, because the region that
#: decides the answer is the flat end of the curve where adjacent rungs are
#: near-identical arms. `required_seeds` is what covers the gamble.
SATURATION_SEEDS = 250

#: The most seeds a top-up may ask for. A ceiling declared in advance so that
#: "run more seeds" cannot become an open-ended search for a bound that clears.
SEED_CAP = 500

#: Paired-difference spread above which `SATURATION_SEEDS` is not enough, and
#: the internal pilot tops up. Set at the near-identical-arm end of the measured
#: range: below this, 250 seeds already put the half-width under the margin.
TOPUP_SPREAD = 0.16

#: Seeds are topped up in blocks of this, so a shard boundary stays a round
#: number and a top-up cannot be tuned to the seed.
SEED_BLOCK = 50

#: Two-sided 95% critical value used for *planning* a seed count. The decision
#: itself uses [t_critical][evogfn.benchmark.statistics.t_critical] through
#: `compare`; this is only ever multiplied by a spread to size a run.
_PLANNING_CRITICAL = 1.98

#: How far from exactly 2.0 an adjacent rung ratio may sit and still be a
#: doubling. A doubling ladder anchored on a shipped value that is not a power
#: of two has to round somewhere -- 12, 25, 50 is a ratio of 2.083 at the bottom
#: -- and this is the tolerance that admits the rounding and nothing wider.
_RATIO_BOUNDS = (1.8, 2.2)


class Verdict(StrEnum):
    """What one doubling of the budget did, in the three states it can be in.

    Three and not two. "Bounded" and "improving" are both measurements;
    "inconclusive" is a statement about the seed count, and collapsing it into
    "bounded" is exactly the error that lets an underpowered ladder read as a
    saturated one. Only `required_seeds` reads this third state, and it is the
    one the internal pilot exists for.
    """

    #: The 95% upper bound on the gain falls below the margin. This doubling
    #: cannot buy enough to matter.
    BOUNDED = "saturated"

    #: The interval excludes zero on the better side: this doubling bought
    #: something real, and the budget is below the knee.
    IMPROVING = "still improving"

    #: The interval spans the margin. Nothing has been shown either way, and
    #: the honest answer is the seed count rather than a plateau.
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class Doubling:
    """One adjacent pair of rungs, and what the larger one bought.

    Attributes:
        smaller: The lower rung's budget.
        larger: The upper rung's budget, twice the lower up to rounding.
        margin: The indifference margin this pair was judged against, carried
            so a printed line cannot be read against a bound it was not judged
            by.
        comparison: The paired comparison, oriented so that ``mean`` is the
            regret *reduction* the doubling bought and ``high`` is the 95%
            upper bound on it.
        spread: Standard deviation of the paired differences. Reported because
            it is the only input `required_seeds` reads, so a top-up can be
            checked against the number that triggered it.
    """

    smaller: int
    larger: int
    margin: float
    comparison: PairedComparison
    spread: float

    @property
    def bounded(self) -> bool:
        """Whether this doubling provably buys less than the margin.

        The upper bound and not the mean. A mean below the margin says the
        point estimate is small; the bound says the *evidence excludes* a gain
        worth having, which is the claim the rule needs and the only one an
        underpowered run cannot accidentally make.
        """
        return self.comparison.high < self.margin

    @property
    def verdict(self) -> Verdict:
        """Which of the three states this pair is in."""
        if self.bounded:
            return Verdict.BOUNDED
        if self.comparison.low > 0.0:
            return Verdict.IMPROVING
        return Verdict.INCONCLUSIVE

    @property
    def n(self) -> int:
        """Seeds the pair was compared on."""
        return self.comparison.n

    def __repr__(self) -> str:
        """One line carrying the effect, its interval, the win rate and a verdict."""
        return (
            f"{self.smaller} -> {self.larger}: gain {self.comparison.mean:+.4f} "
            f"[{self.comparison.low:+.4f}, {self.comparison.high:+.4f}] "
            f"wins {self.comparison.wins}/{self.comparison.n}  {self.verdict}"
        )


@dataclass(frozen=True, slots=True)
class Saturation:
    """Where a budget ladder flattened, or the bound on what is left above it.

    Attributes:
        axis: The knob that was varied, for the caption -- ``steps`` for a
            GFlowNet, ``generations`` for a proxy-optimising baseline.
        budgets: The rungs measured, ascending.
        doublings: One entry per adjacent pair, in the same order. **Every**
            pair, not only the one that fired: a worsening curve and a plateau
            both satisfy the rule and only these lines tell them apart.
        margin: The indifference margin applied.
        budget: The declared saturation point, or ``None`` when the ladder
            never flattened.
        confirming: How many bounded doublings sit at or above the declared
            point. One is a declaration that a single tight interval decided;
            two is what the sequential stopping rule requires before a rung
            above the knee may be skipped.
        ceiling: The declared top of the grid, when the caller states one. A
            ladder that has reached its ceiling is finished whatever
            ``confirming`` says, because no rung above it will ever be run.
    """

    axis: str
    budgets: tuple[int, ...]
    doublings: tuple[Doubling, ...]
    margin: float
    budget: int | None
    confirming: int
    ceiling: int | None = None

    @property
    def measured(self) -> bool:
        """Whether a saturation point was found at all inside the ladder."""
        return self.budget is not None

    @property
    def settled(self) -> bool:
        """Whether the ladder can stop, rather than owing another rung.

        A declaration resting on one bounded doubling is provisional: one
        interval can be tight by chance, and the next rung up is entitled to
        revoke it. So a knee is settled once a second bounded doubling stands
        above it -- or once the ladder has reached the ceiling declared in
        advance, past which nothing more will be run whatever the rule wants.
        """
        if self.budget is None:
            return self.ceiling is not None and self.budgets[-1] >= self.ceiling
        reached = self.ceiling is not None and self.budgets[-1] >= self.ceiling
        return self.confirming >= 2 or reached  # noqa: PLR2004 - the confirming rung

    @property
    def residual(self) -> float:
        """The 95% upper bound on what the top doubling still bought.

        Reported whether or not a knee was found, and it is the number to quote
        when one was not: "at the top rung, doubling the budget again buys at
        most this much regret". Set that beside the margin a headline claims
        over its baseline and the exposure is quantified rather than asserted.
        """
        return self.doublings[-1].comparison.high

    @property
    def reason(self) -> str:
        """Why this budget, in a form that can be pasted into a caption."""
        top = self.budgets[-1]
        if self.budget is None:
            return (
                f"no saturation point measured on {self.axis} in "
                f"{self.budgets[0]}-{top}: at {top}, doubling the proxy budget "
                f"again buys at most {self.residual:+.4f} regret (95% upper "
                f"bound, {self.doublings[-1].n} seeds), which does not clear the "
                f"{self.margin:g} indifference margin. The reported budget is a "
                f"cost ceiling we chose, not a saturation point"
            )
        confirmation = (
            f"{self.confirming} bounded doubling(s) above it"
            if self.confirming > 1
            else "one bounded doubling above it, so this is provisional until the next rung"
        )
        return (
            f"{self.axis}={self.budget}: every doubling from there to {top} is "
            f"bounded below the {self.margin:g} margin ({confirmation}); the "
            f"residual at {top} is {self.residual:+.4f}"
        )

    def __repr__(self) -> str:
        """Name the axis, the point and its standing."""
        where = "unmeasured" if self.budget is None else str(self.budget)
        return f"{self.axis}={where} ({'settled' if self.settled else 'provisional'})"


def _paired(
    smaller: int,
    larger: int,
    regret: Mapping[int, Mapping[int, float]],
    margin: float,
) -> Doubling:
    """Compare one adjacent pair on the seeds both rungs hold.

    Args:
        smaller: The lower rung's budget.
        larger: The upper rung's budget.
        regret: Per-seed regret by budget then by seed.
        margin: The indifference margin to judge the pair against.

    Returns:
        The pair, with the comparison oriented so a positive mean means the
        larger budget won.

    Raises:
        ValueError: If the two rungs share fewer than two seeds. An unpaired
            or single-seed pair has no variance, and a bound computed from it
            would be a division by zero wearing a confidence interval.
    """
    shared = sorted(set(regret[smaller]) & set(regret[larger]))
    if len(shared) < 2:  # noqa: PLR2004 - a variance needs two observations
        raise ValueError(
            f"rungs {smaller} and {larger} share {len(shared)} seed(s), so the "
            f"doubling between them cannot be paired"
        )
    low = np.array([regret[smaller][seed] for seed in shared], dtype=np.float64)
    high = np.array([regret[larger][seed] for seed in shared], dtype=np.float64)
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError(
            f"rungs {smaller} and {larger} hold a non-finite regret on a shared "
            f"seed; a bound taken over it would be silently meaningless"
        )
    # `compare` forms `second - first` when lower is better, so passing the
    # larger budget first makes `mean` the regret reduction the doubling bought.
    comparison = compare(f"{larger} over {smaller}", high, low, higher_is_better=False)
    return Doubling(
        smaller=smaller,
        larger=larger,
        margin=margin,
        comparison=comparison,
        spread=float((low - high).std(ddof=1)),
    )


def _check_ladder(budgets: Sequence[int]) -> None:
    """Refuse anything that is not a contiguous doubling ladder.

    Args:
        budgets: The rungs, ascending.

    Raises:
        ValueError: If there are fewer than two rungs, if any is not positive,
            or if an adjacent ratio is not a doubling. The last one is the
            guard that matters: a rule reading "the doubling above ``s``"
            applied to a ladder with a rung inserted between two others is
            reading a smaller step, and a smaller step clears the bound more
            easily. The grid is fixed in advance and this is what makes that
            statement checkable rather than a promise.
    """
    if len(budgets) < 2:  # noqa: PLR2004 - a doubling needs two rungs
        raise ValueError(f"a ladder needs at least two rungs, got {list(budgets)}")
    if any(budget < 1 for budget in budgets):
        raise ValueError(f"every rung must be a positive budget, got {list(budgets)}")
    lower, upper = _RATIO_BOUNDS
    for smaller, larger in pairwise(budgets):
        ratio = larger / smaller
        if not lower <= ratio <= upper:
            raise ValueError(
                f"rungs {smaller} and {larger} sit a factor of {ratio:.2f} apart, "
                f"which is not a doubling; the rule reads adjacent pairs as "
                f"doublings and a ladder spaced otherwise would be judged at a "
                f"step size it was not declared at"
            )


def saturation(
    regret: Mapping[int, Mapping[int, float]],
    *,
    axis: str = "steps",
    margin: float = INDIFFERENCE_MARGIN,
    ceiling: int | None = None,
) -> Saturation:
    """Apply the pre-declared rule to a measured budget ladder.

    The rule, restated where it is executed: **the saturation point is the
    smallest rung whose doubling, and every doubling above it, has a 95% upper
    bound on the gain below the margin.** No rung qualifies when the top
    doubling does not clear the bound, and the answer is then the residual
    rather than a knee.

    Args:
        regret: Per-seed regret, keyed by budget and then by seed. Rungs need
            not hold identical seed sets: each adjacent pair is compared on the
            seeds both hold, which is what keeps a topped-up rung usable
            against one that was not topped up.
        axis: The knob varied, for the caption.
        margin: The indifference margin. Defaults to `INDIFFERENCE_MARGIN` and
            may only be tightened.
        ceiling: The top of the declared grid, when one has been declared.
            Read only by `Saturation.settled`, so that a ladder which has run
            out of grid stops owing a confirming rung it will never get.

    Returns:
        Where the ladder flattened, every adjacent pair's numbers, and the
        residual above the top rung.

    Raises:
        ValueError: If the ladder is not a contiguous doubling ladder, if a
            pair cannot be paired, or if ``margin`` exceeds
            `INDIFFERENCE_MARGIN`. The last is the one worth naming: widening
            the indifference bound after the curve exists would let a knee be
            declared anywhere on it, so the bound is fixed in advance and this
            function only accepts it or something stricter.
    """
    if margin <= 0.0:
        raise ValueError(f"margin must be positive, got {margin}")
    if margin > INDIFFERENCE_MARGIN:
        raise ValueError(
            f"margin {margin} is looser than the declared indifference margin "
            f"{INDIFFERENCE_MARGIN}; the bound was fixed before any rung was run "
            f"and may be tightened, never loosened"
        )
    budgets = tuple(sorted(regret))
    _check_ladder(budgets)

    pairs = tuple(_paired(smaller, larger, regret, margin) for smaller, larger in pairwise(budgets))

    budget: int | None = None
    confirming = 0
    for index, pair in enumerate(pairs):
        above = pairs[index:]
        if all(other.bounded for other in above):
            budget = pair.smaller
            confirming = len(above)
            break

    return Saturation(
        axis=axis,
        budgets=budgets,
        doublings=pairs,
        margin=margin,
        budget=budget,
        confirming=confirming,
        ceiling=ceiling,
    )


def pooled_spread(doublings: Sequence[Doubling]) -> float:
    """The paired-difference spread pooled over a ladder's adjacent pairs.

    Pooled rather than taken from the widest pair, because a top-up sized on
    the worst pair would be sized on whichever comparison happened to draw the
    noisiest seeds.

    Args:
        doublings: The pairs, each carrying its own spread and seed count.

    Returns:
        The degrees-of-freedom weighted pooled standard deviation, or ``0.0``
        for an empty ladder.

    Raises:
        ValueError: If no pair holds two seeds, so nothing can be pooled.
    """
    if not doublings:
        return 0.0
    weight = sum(pair.n - 1 for pair in doublings)
    if weight <= 0:
        raise ValueError("no pair holds enough seeds to pool a spread from")
    total = sum((pair.n - 1) * pair.spread**2 for pair in doublings)
    return math.sqrt(total / weight)


def required_seeds(
    spread: float,
    *,
    margin: float = INDIFFERENCE_MARGIN,
    planned: int = SATURATION_SEEDS,
    cap: int = SEED_CAP,
    block: int = SEED_BLOCK,
) -> int:
    """How many seeds a ladder needs, from its realised spread alone.

    The internal pilot. It reads the variance and **nothing else** -- no mean,
    no sign, no verdict -- and that is the whole of why it is not optional
    stopping: a trigger that cannot see which way the effect went cannot be
    used to keep running until the answer is agreeable. The signature is the
    guarantee; passing a `Doubling` here instead of a float would hand the
    function the effect it must not see.

    Args:
        spread: The realised paired-difference standard deviation, pooled over
            the ladder's adjacent pairs. See `pooled_spread`.
        margin: The indifference margin the half-width must fit under.
        planned: The seed count the ladder was run at, returned unchanged when
            the realised spread is no worse than planned for.
        cap: The declared maximum. Fixed in advance so that "run more seeds"
            cannot become an open-ended search.
        block: Seeds are topped up in whole blocks of this.

    Returns:
        The seed count to run to, never below ``planned`` and never above
        ``cap``.

    Raises:
        ValueError: If the spread is negative, or if ``margin`` exceeds
            `INDIFFERENCE_MARGIN` -- a looser margin would ask for fewer seeds,
            which is the same loophole `saturation` closes.
    """
    if spread < 0.0:
        raise ValueError(f"spread must not be negative, got {spread}")
    if margin <= 0.0:
        raise ValueError(f"margin must be positive, got {margin}")
    if margin > INDIFFERENCE_MARGIN:
        raise ValueError(
            f"margin {margin} is looser than the declared {INDIFFERENCE_MARGIN}; "
            f"a wider bound would ask for fewer seeds"
        )
    if spread <= TOPUP_SPREAD:
        return planned
    needed = math.ceil((_PLANNING_CRITICAL * spread / margin) ** 2)
    rounded = block * math.ceil(needed / block)
    return min(max(rounded, planned), cap)
