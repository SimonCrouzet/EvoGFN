"""Paired comparison across seeds, without taking a SciPy dependency.

Two methods run on the same seed share their surrogate initialisation, their
landscape, and their initial design. Pairing on that removes the seed-to-seed
variance, which on these landscapes is the dominant term; an unpaired test over
the same numbers throws that cancellation away and reports a weaker statistic
for the same experiment.

What is reported, and why all of it
-----------------------------------

A mean difference alone hides whether one seed carried the result. So each
comparison reports the mean, a confidence interval, the paired ``t``, and the
**win rate** -- how many seeds the method actually won. A method that wins by
0.9 on average while winning 9 of 15 seeds is not the same as one that wins
0.9 on every seed, and only the second is a method you would use.

The interval is Student's ``t`` on the paired differences, which assumes those
differences are roughly normal. With 100 seeds that is unobjectionable; below
about 20 the win rate is the more trustworthy of the two figures, which is part
of why both are here.

Seeds are not the only unit, and treating them as one is a bigger error
----------------------------------------------------------------------

Everything above pairs on the *seed*, which is right as long as the landscape is
held fixed: a seed varies the wild type and the surrogate's initialisation, and
100 of them say how much a result moves under those. They say nothing about how
much it moves under a different draw from the instance generator, because not one
of them varies it.

Once a comparison is replicated across instances there are two variance
components, and only one of them shrinks with seeds. `decompose` estimates both,
`draws_needed` turns them into a number of instances, and
`VarianceComponents.design_effect` is the factor by which pooling instance x seed
pairs as independent observations would understate the standard error -- which is
the arithmetic behind calling that pooling pseudo-replication rather than merely
a shortcut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Two-sided 95% critical values of Student's t, indexed by degrees of freedom.
#: Tabulated rather than computed so this module needs no SciPy; beyond the
#: table the normal approximation is accurate to better than a percent.
_T_CRITICAL = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}
_T_ASYMPTOTIC = 1.960

#: Lower 5% points of the chi-square distribution, by degrees of freedom.
#: Tabulated for the same reason `_T_CRITICAL` is, and read for one purpose
#: only: a 95% **upper** bound on a variance, which is what a design sized from
#: a pilot needs. A pilot that drew few instances estimates the between-instance
#: variance badly, and the bound is what makes that failure ask for more
#: instances rather than fewer.
_CHI2_LOWER_5 = {
    1: 0.00393,
    2: 0.10259,
    3: 0.35185,
    4: 0.71072,
    5: 1.14548,
    6: 1.63538,
    7: 2.16735,
    8: 2.73264,
    9: 3.32511,
    10: 3.94030,
    11: 4.57481,
    12: 5.22603,
    15: 7.26094,
    20: 10.85081,
    25: 14.61141,
    30: 18.49266,
    40: 26.50930,
    60: 43.18796,
    120: 95.70464,
}


def t_critical(degrees_of_freedom: int) -> float:
    """Two-sided 95% critical value, interpolating the table conservatively.

    Args:
        degrees_of_freedom: Sample size minus one.

    Returns:
        The critical value, or the normal approximation beyond the table.
    """
    if degrees_of_freedom < 1:
        return math.inf
    if degrees_of_freedom in _T_CRITICAL:
        return _T_CRITICAL[degrees_of_freedom]
    # The *largest tabulated value at or below* the requested one. `t` falls as
    # the degrees of freedom rise, so reading the next entry **up** would return
    # a smaller critical value than the true one and quietly narrow every
    # interval taken at an untabulated size -- this used to round that way while
    # its comment claimed the opposite, which put the error on the side that
    # flatters a result. Rounding down overstates the interval instead.
    smaller = [d for d in _T_CRITICAL if d < degrees_of_freedom]
    if not smaller:
        return _T_CRITICAL[min(_T_CRITICAL)]
    if degrees_of_freedom > max(_T_CRITICAL):
        return _T_ASYMPTOTIC
    return _T_CRITICAL[max(smaller)]


def variance_upper_bound(variance: float, degrees_of_freedom: int) -> float:
    """The 95% upper confidence bound on a variance, from its sample estimate.

    ``(df * S^2) / sigma^2`` is chi-square on ``df``, so dividing by the
    distribution's lower 5% point gives a bound the true variance sits under 95%
    of the time. Used to size a design from a pilot rather than to report
    anything: a pilot that drew three instances estimates their spread on two
    degrees of freedom, where the bound is nearly twenty times the estimate, and
    a design sized on the point estimate would inherit that ignorance as
    confidence.

    Args:
        variance: The sample variance.
        degrees_of_freedom: Observations minus one.

    Returns:
        The bound, or ``inf`` below one degree of freedom -- a single
        observation bounds nothing, and returning its own value would claim
        otherwise.

    Raises:
        ValueError: If the variance is negative.
    """
    if variance < 0.0:
        raise ValueError(f"a variance cannot be negative, got {variance}")
    if degrees_of_freedom < 1:
        return math.inf
    return degrees_of_freedom * variance / _chi2_lower_5(degrees_of_freedom)


def _chi2_lower_5(degrees_of_freedom: int) -> float:
    """The lower 5% point of chi-square, tabulated below and approximated above.

    Inside the table the entry is exact. Between two entries the smaller one is
    taken: the point rises with the degrees of freedom, so dividing by a smaller
    one widens the bound above -- wrong in the direction that asks for a larger
    design. Past the table the Wilson-Hilferty cube-root approximation is used
    instead of the top entry, which by then would be off by a factor of two and
    would demand a design nobody could run.
    """
    if degrees_of_freedom in _CHI2_LOWER_5:
        return _CHI2_LOWER_5[degrees_of_freedom]
    if degrees_of_freedom > max(_CHI2_LOWER_5):
        term = 2.0 / (9.0 * degrees_of_freedom)
        return degrees_of_freedom * (1.0 - term - 1.645 * math.sqrt(term)) ** 3
    return _CHI2_LOWER_5[max(d for d in _CHI2_LOWER_5 if d < degrees_of_freedom)]


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """One method against another, across shared seeds.

    Attributes:
        name: What was compared against what.
        mean: Mean paired difference, positive when the first method is better.
        low: Lower bound of the 95% interval.
        high: Upper bound.
        t: Paired t statistic.
        wins: Seeds on which the first method won.
        n: Seeds compared.
    """

    name: str
    mean: float
    low: float
    high: float
    t: float
    wins: int
    n: int

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero.

        A convenience, not a verdict. An interval that excludes zero on 15
        seeds and one that excludes it on 200 are different evidence.
        """
        return self.low > 0.0 or self.high < 0.0

    @property
    def win_rate(self) -> float:
        """Fraction of seeds won. Half is a coin flip."""
        return self.wins / self.n if self.n else 0.0

    def __repr__(self) -> str:
        """One line carrying the effect, its interval, and the win rate."""
        mark = "*" if self.significant else " "
        return (
            f"{self.name}: {self.mean:+.3f} [{self.low:+.3f}, {self.high:+.3f}]{mark} "
            f"t={self.t:.2f}  wins {self.wins}/{self.n}"
        )


def compare(
    name: str,
    first: np.ndarray,
    second: np.ndarray,
    *,
    higher_is_better: bool = True,
) -> PairedComparison:
    """Compare two methods on matched seeds.

    Args:
        name: Label for the comparison.
        first: Per-seed metric for the method under test.
        second: Per-seed metric for the method compared against, in the same
            seed order.
        higher_is_better: Whether a larger metric is a better result. Regret
            and other losses should pass ``False`` so a positive difference
            always means the first method won.

    Returns:
        The paired comparison.

    Raises:
        ValueError: If the arrays differ in length, or hold fewer than two
            seeds -- a single seed has no variance and a statistic computed
            from it would be a division by zero dressed up as a result.
    """
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"cannot pair {a.shape[0]} seeds against {b.shape[0]}")
    if a.shape[0] < 2:  # noqa: PLR2004 - a variance needs two observations
        raise ValueError(f"need at least 2 seeds to compare, got {a.shape[0]}")

    differences = (a - b) if higher_is_better else (b - a)
    mean = float(differences.mean())
    spread = float(differences.std(ddof=1))
    n = differences.shape[0]
    error = spread / math.sqrt(n) if spread else 0.0
    half = t_critical(n - 1) * error
    return PairedComparison(
        name=name,
        mean=mean,
        low=mean - half,
        high=mean + half,
        t=mean / error if error else math.inf if mean else 0.0,
        wins=int((differences > 0).sum()),
        n=n,
    )


def seeds_needed(observed: PairedComparison, *, power: float = 0.8) -> int:
    """Seeds required to resolve an effect of the size just observed.

    Answers the question a non-significant result actually raises: is this a
    null, or an underpowered look at something real? Reported so an
    inconclusive comparison names its own price rather than inviting a guess.

    Args:
        observed: A comparison already run.
        power: Probability of detecting the effect if it is real.

    Returns:
        Approximate number of seeds, by the normal approximation. Returns 0
        when the observed effect is exactly zero, since no sample resolves it.
    """
    spread = observed.mean / observed.t if observed.t and math.isfinite(observed.t) else 0.0
    if not spread or not observed.mean:
        return 0
    # sigma of the paired differences, recovered from the standard error.
    sigma = spread * math.sqrt(observed.n)
    required = ((_T_ASYMPTOTIC + _z_power(power)) * sigma / observed.mean) ** 2
    return math.ceil(required)


def _z_power(power: float) -> float:
    """Normal deviate for a stated power, defaulting to 80%."""
    return {0.8: 0.842, 0.9: 1.282, 0.95: 1.645}.get(power, 0.842)


@dataclass(frozen=True, slots=True)
class VarianceComponents:
    """How much of a paired effect varies with the instance, and how much with the seed.

    The model is the plainest one that has the two terms in it: on instance
    ``i`` and seed ``s`` the paired difference is ``mu + b_i + e_is``, with
    ``b_i`` the instance's own offset and ``e_is`` everything a seed varies.
    ``mu`` -- the effect -- is **deliberately not an attribute here**, and that
    absence is the same guarantee
    [required_seeds][evogfn.benchmark.saturation.required_seeds] gets from its
    signature: `draws_needed` takes this object, so it cannot see which way the
    effect went, so a design cannot be resized until the answer is agreeable.

    Attributes:
        name: What was decomposed, for the printed line.
        between: ``sigma_b``, the standard deviation of the per-instance offset.
            The term seeds cannot touch.
        within: ``sigma_w``, the standard deviation of the paired difference
            across seeds within one instance.
        instances: Instances the decomposition read.
        seeds: Seeds per instance it read, as a mean where they differed.
        conservative: Whether ``between`` is a 95% upper bound rather than a
            point estimate. Carried rather than assumed, because a design sized
            on a point estimate and one sized on a bound differ by a factor a
            reader has to be able to see.
    """

    name: str
    between: float
    within: float
    instances: int
    seeds: float
    conservative: bool

    @property
    def intraclass_correlation(self) -> float:
        """Share of the total variance that the instance owns.

        Zero means every draw from the generator behaves the same and seeds are
        the only unit that matters; one means seeds add nothing once the
        instance is known.
        """
        total = self.between**2 + self.within**2
        return self.between**2 / total if total else 0.0

    def design_effect(self, *, seeds: float) -> float:
        """How far pooling instance x seed pairs would understate the error.

        The arithmetic behind the word pseudo-replication. Pooling ``m x n``
        pairs as independent reports a standard error of
        ``sqrt((sigma_b^2 + sigma_w^2) / (m n))``; the honest one, with the
        instance as the unit, is ``sqrt(sigma_b^2 / m + sigma_w^2 / (m n))``.
        Their ratio is ``sqrt(1 + (n - 1) * rho)`` and depends on the seed count
        but **not** on the instance count -- which is why running more seeds
        makes the pooled figure look better while making it no more true.

        The seed count is a parameter and not `seeds`, because the factor is
        almost always wanted for a design other than the one it was measured on:
        a pilot at ten seeds and the tier at a hundred give figures a factor of
        three apart, and quoting the pilot's beside the tier's seed count is the
        one arithmetic slip this whole path exists to make impossible.

        Args:
            seeds: Seeds per instance in the design being described.

        Returns:
            A factor at or above one. Two means the pooled interval is half the
            width it should be.

        Raises:
            ValueError: If fewer than one seed.
        """
        if seeds < 1:
            raise ValueError(f"a design needs at least one seed per instance, got {seeds}")
        return math.sqrt(1.0 + (seeds - 1.0) * self.intraclass_correlation)

    def standard_error(self, *, instances: int, seeds: float) -> float:
        """Standard error of the mean effect over a design of this shape.

        Args:
            instances: Landscape draws the design would run.
            seeds: Seeds per draw.

        Returns:
            The standard error, honouring the instance as the unit of
            replication.

        Raises:
            ValueError: If either count is below one.
        """
        if instances < 1 or seeds < 1:
            raise ValueError(f"a design needs at least one of each, got {instances}x{seeds}")
        return math.sqrt(self.between**2 / instances + self.within**2 / (instances * seeds))

    def sign_resolution(self, *, seeds: float) -> float:
        """Smallest per-instance effect whose sign one instance resolves.

        An ordering flip is a claim about a *single* draw, so it is read off
        that draw's own comparison rather than off the mean. Below this the
        instance cannot say which way it went, and an apparent flip there is the
        seed count rather than the landscape.

        Args:
            seeds: Seeds run on that instance.

        Returns:
            The half-width of the 95% interval on one instance's mean effect.

        Raises:
            ValueError: If fewer than two seeds, which have no variance.
        """
        if seeds < 2:  # noqa: PLR2004 - a variance needs two observations
            raise ValueError(f"a single instance needs at least 2 seeds, got {seeds}")
        return t_critical(math.floor(seeds) - 1) * self.within / math.sqrt(seeds)

    def __repr__(self) -> str:
        """One line carrying both components and what pooling would cost."""
        bound = "95% upper bound" if self.conservative else "point estimate"
        return (
            f"{self.name}: between {self.between:.4f} ({bound})  within {self.within:.4f}  "
            f"ICC {self.intraclass_correlation:.3f}  pooling at {self.seeds:g} seeds would "
            f"understate the error {self.design_effect(seeds=self.seeds):.2f}x  "
            f"({self.instances} instances x {self.seeds:g} seeds)"
        )


def decompose(
    name: str,
    differences: Sequence[Sequence[float] | np.ndarray],
    *,
    conservative: bool = True,
) -> VarianceComponents:
    """Split a replicated paired effect into its instance and seed variances.

    A one-way random-effects decomposition, which is the whole of the arithmetic
    the instance-resampling design turns on. The per-instance mean differences
    vary by ``sigma_b^2 + sigma_w^2 / n``; the seed-level differences within an
    instance vary by ``sigma_w^2``; subtracting recovers the instance term.

    Args:
        name: What is being decomposed, for the printed line.
        differences: One sequence of paired differences per instance, each
            already oriented so a positive value means the arm under test won.
            Instances need not hold the same number of seeds.
        conservative: Whether to return the 95% upper bound on the instance
            term rather than its point estimate. Default true, and it is the
            setting a design should be sized at: a pilot that drew three
            instances estimates their spread on two degrees of freedom, and the
            point estimate there is a number with no information in it wearing
            four decimal places.

    Returns:
        The two components, with the instance term floored at zero -- the
        estimator is a difference of two variances and can go negative when the
        instances genuinely do not differ, and a negative standard deviation is
        not a smaller answer, it is no answer.

    Raises:
        ValueError: If fewer than two instances are given, or any instance
            holds fewer than two seeds. Both are the same failure: a variance
            needs two observations, and one computed from fewer would be a
            division by zero wearing a confidence interval.
    """
    groups = [np.asarray(group, dtype=np.float64).reshape(-1) for group in differences]
    if len(groups) < 2:  # noqa: PLR2004 - a between-instance variance needs two instances
        raise ValueError(
            f"need at least 2 instances to separate the instance term from the seed "
            f"term, got {len(groups)}; with one instance every difference is within it"
        )
    if any(group.size < 2 for group in groups):  # noqa: PLR2004
        raise ValueError(
            f"every instance needs at least 2 seeds, got {[int(g.size) for g in groups]}"
        )
    if not all(np.isfinite(group).all() for group in groups):
        raise ValueError(f"{name} holds a non-finite difference; a variance over it is silent")

    counts = np.array([group.size for group in groups], dtype=np.float64)
    means = np.array([group.mean() for group in groups], dtype=np.float64)
    # Pooled within-instance variance, weighted by each instance's own degrees
    # of freedom so an instance run at fewer seeds contributes proportionally.
    weights = counts - 1.0
    within_variance = float(
        sum(w * float(g.var(ddof=1)) for w, g in zip(weights, groups, strict=True)) / weights.sum()
    )
    spread_of_means = float(means.var(ddof=1))
    if conservative:
        spread_of_means = variance_upper_bound(spread_of_means, len(groups) - 1)
    # The per-instance means already carry `sigma_w^2 / n` of seed noise, so the
    # instance term is what is left after removing it. Removing the point
    # estimate rather than a lower bound on it: `sigma_w` is estimated on
    # `sum(n_i - 1)` degrees of freedom, which is two orders more than the
    # instance term has, and bounding a term that well determined would inflate
    # the answer for no gain in safety.
    between_variance = max(0.0, spread_of_means - within_variance * float(np.mean(1.0 / counts)))
    return VarianceComponents(
        name=name,
        between=math.sqrt(between_variance),
        within=math.sqrt(within_variance),
        instances=len(groups),
        seeds=float(counts.mean()),
        conservative=conservative,
    )


def pool_components(name: str, parts: Sequence[VarianceComponents]) -> VarianceComponents:
    """One set of components from several arms' decompositions.

    Pooled rather than taken from the widest arm, for the reason
    [pooled_spread][evogfn.benchmark.saturation.pooled_spread] gives: a design
    sized on the worst pair is sized on whichever comparison happened to draw
    the noisiest instances. Each component is averaged in the squared units it
    is a variance in, weighted by the degrees of freedom that estimated it.

    Args:
        name: What the pooled figure covers.
        parts: One entry per arm compared against the reference.

    Returns:
        The pooled components, conservative if every part was.

    Raises:
        ValueError: If nothing was given, or if the parts disagree about how
            many instances or seeds they were measured on -- pooling across
            different designs would produce a number belonging to neither.
    """
    if not parts:
        raise ValueError("nothing to pool")
    shapes = {(part.instances, part.seeds) for part in parts}
    if len(shapes) > 1:
        raise ValueError(
            f"cannot pool components measured on different designs: {sorted(shapes)}; "
            f"the pooled variance would describe a design nobody ran"
        )
    between_weight = sum(part.instances - 1 for part in parts)
    within_weight = sum(part.instances * (part.seeds - 1) for part in parts)
    between = sum((part.instances - 1) * part.between**2 for part in parts) / between_weight
    within = (
        sum(part.instances * (part.seeds - 1) * part.within**2 for part in parts) / within_weight
    )
    return VarianceComponents(
        name=name,
        between=math.sqrt(between),
        within=math.sqrt(within),
        instances=parts[0].instances,
        seeds=parts[0].seeds,
        # Only if every part was. One point estimate in the pool makes the
        # pooled figure a point estimate, and calling it a bound would be the
        # one claim this whole path exists to avoid making by accident.
        conservative=all(part.conservative for part in parts),
    )


def draws_needed(
    components: VarianceComponents,
    *,
    margin: float,
    seeds: float,
    power: float = 0.8,
    floor: int = 1,
) -> int:
    """Landscape draws a design needs, from its variance components alone.

    Reads the two variances and **nothing else** -- no mean, no sign, no
    verdict. That is why `VarianceComponents` does not carry the effect: a
    trigger that cannot see which way the answer went cannot be used to keep
    drawing instances until it is agreeable, and the signature is the guarantee
    rather than a promise in a docstring.

    The polarity is the safe one, for the same reason
    [saturation][evogfn.benchmark.saturation] states rather than caveats it: a
    thinner pilot widens the bound on the instance term, which asks for **more**
    draws. Underpowering costs compute rather than shrinking a design.

    Args:
        components: The decomposition, normally conservative.
        margin: The smallest effect the design must resolve. Declared before the
            numbers by whoever calls this; there is deliberately no default,
            because a margin chosen after the variance is known is a margin
            chosen to produce an affordable answer.
        seeds: Seeds per draw the design will run.
        power: Probability of resolving an effect of exactly ``margin``.
        floor: Fewest draws the design may run whatever the arithmetic says.
            Carries a requirement that is not about variance at all -- see
            `unanimity_floor`.

    Returns:
        The number of draws, at or above ``floor``. Uncapped: a requirement
        larger than anyone can afford is a finding, and clamping it here would
        hide the finding inside the cap.

    Raises:
        ValueError: If the margin is not positive, or the seed count is below
            one.
    """
    if margin <= 0.0:
        raise ValueError(f"margin must be positive, got {margin}")
    if seeds < 1:
        raise ValueError(f"a draw needs at least one seed, got {seeds}")
    # The variance of one draw's mean effect at the design's seed count. Only
    # the second term shrinks with seeds, which is the whole point: past the
    # knee, seeds buy nothing a replication design needs.
    per_draw = components.between**2 + components.within**2 / seeds
    required = ((_T_ASYMPTOTIC + _z_power(power)) ** 2) * per_draw / margin**2
    return max(floor, math.ceil(required))


def unanimity_floor(alpha: float = 0.05) -> int:
    """Fewest instances at which unanimous agreement is significant at all.

    The clause of a draw count that no variance can move. A per-instance
    analysis reports how many draws the ordering held on, and the only test that
    reads that column without assuming a distribution is the sign test, whose
    strongest possible verdict on ``m`` draws is unanimity at two-sided
    ``2 * 2^-m``. At three draws that is 0.25: three instances agreeing is the
    best outcome the design can produce and it is not evidence, whatever seed
    count sits underneath it.

    Args:
        alpha: The significance the unanimous case must reach.

    Returns:
        The smallest instance count whose unanimous sign test clears ``alpha``.

    Raises:
        ValueError: If ``alpha`` is not strictly between zero and one.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must sit strictly between 0 and 1, got {alpha}")
    return math.ceil(math.log2(2.0 / alpha))


def unanimity_p(instances: int, agreeing: int) -> float:
    """Two-sided sign-test p-value for how many draws agreed on the direction.

    Args:
        instances: Draws compared.
        agreeing: Draws on which the effect took the majority direction.

    Returns:
        The two-sided p-value, or ``1.0`` for fewer than one instance.

    Raises:
        ValueError: If more draws agreed than were run.
    """
    if agreeing > instances:
        raise ValueError(f"{agreeing} draws cannot agree out of {instances}")
    if instances < 1:
        return 1.0
    extreme = max(agreeing, instances - agreeing)
    tail = sum(math.comb(instances, k) for k in range(extreme, instances + 1))
    return min(1.0, 2.0 * tail / 2.0**instances)
