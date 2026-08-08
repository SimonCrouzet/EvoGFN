"""Tests for the paired-comparison machinery, and for the two-level variance split.

Three failures, and every one of them is silent in the direction that makes a
result look better established than it is.

**An interval narrowed by table lookup.** `t_critical` is read at whatever degrees
of freedom a comparison happens to have, and most counts are not tabulated. Rounding
to the *next entry up* returns a smaller critical value than the true one, so every
interval taken at an untabulated size comes back narrower than it should be -- by
about a percent, on every comparison in the project, with nothing to show for it.
Nothing crashes and no number looks wrong.

**A design sized on a variance nobody could estimate.** The between-instance term
is estimated on ``instances - 1`` degrees of freedom, which at the replication
tier's three draws is two. A design sized on the point estimate there would
inherit that ignorance as confidence, and would do so quietly: the number has
four decimal places either way. The bound is what makes a thin pilot ask for a
larger design rather than a smaller one.

**A rule that can see the effect.** `draws_needed` takes variances and never a
mean, which is what stops "draw more instances" becoming a search for an
agreeable answer. That guarantee lives in a signature, and a signature is exactly
the kind of thing a later edit widens for convenience -- so it is tested by
shifting every observation by a constant and demanding the answer not move.
"""

import math

import numpy as np
import pytest

from evogfn.benchmark.statistics import (
    VarianceComponents,
    compare,
    decompose,
    draws_needed,
    pool_components,
    t_critical,
    unanimity_floor,
    unanimity_p,
    variance_upper_bound,
)

#: A margin to size designs against in these tests. Any positive number does;
#: named so the tests read as "at one fixed margin" rather than as arithmetic.
MARGIN = 0.05


def _instances(offsets, *, noise, seeds, spread=1.0):
    """Paired differences for one instance per offset, with reproducible noise."""
    rng = np.random.default_rng(0)
    return [offset + spread * rng.standard_normal(seeds) * noise for offset in offsets]


class TestTheCriticalValueTable:
    """It must never return a smaller critical value than the true one."""

    def test_an_untabulated_size_rounds_toward_the_wider_interval(self):
        # 11 degrees of freedom sits between the 10 and 12 entries. The true
        # value is 2.201. Reading the 12 entry gives 2.179 -- narrower than the
        # truth, which silently overstates every comparison taken at a seed
        # count the table does not hold. Reading the 10 entry gives 2.228.
        assert t_critical(11) == pytest.approx(2.228)
        assert t_critical(11) > 2.201

    def test_a_tabulated_size_is_exact(self):
        assert t_critical(10) == pytest.approx(2.228)
        assert t_critical(120) == pytest.approx(1.980)

    @pytest.mark.parametrize("degrees", [2, 5, 11, 37, 99])
    def test_it_never_undershoots_the_next_tabulated_entry(self, degrees):
        # The property, rather than five separate literals: `t` falls with the
        # degrees of freedom, so a conservative lookup is one that is at least
        # as large as the value at the size actually asked for.
        assert t_critical(degrees) >= t_critical(degrees + 1)

    def test_beyond_the_table_it_is_the_normal_approximation(self):
        assert t_critical(500) == pytest.approx(1.960)

    def test_no_degrees_of_freedom_bounds_nothing(self):
        # Not a large number: a single observation has no interval at all, and
        # returning one would let a one-seed comparison print a bound.
        assert math.isinf(t_critical(0))


class TestTheVarianceBound:
    """A thin sample must widen the bound, not narrow it."""

    def test_two_degrees_of_freedom_bound_almost_nothing(self):
        # This is the number that condemns a three-draw design: the tier's own
        # instance count estimates its between-instance variance on two degrees
        # of freedom, where the 95% bound is nineteen times the estimate.
        assert variance_upper_bound(1.0, 2) == pytest.approx(19.495, rel=1e-3)

    def test_the_bound_tightens_as_the_sample_grows(self):
        bounds = [variance_upper_bound(1.0, degrees) for degrees in (2, 5, 10, 20, 60)]

        assert bounds == sorted(bounds, reverse=True)
        assert bounds[-1] > 1.0

    def test_past_the_table_it_stays_near_the_estimate(self):
        # The tabulated top entry would divide by 95.7 whatever the size, which
        # at 500 degrees of freedom would demand a design four times too large.
        assert variance_upper_bound(1.0, 500) == pytest.approx(1.0, abs=0.2)

    def test_a_negative_variance_is_refused(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            variance_upper_bound(-1.0, 10)


class TestTheDecomposition:
    """It has to tell an instance that matters from one that does not."""

    def test_instances_that_behave_alike_carry_no_instance_term(self):
        # All the variation is within. A positive between-instance term here
        # would be the estimator reading seed noise as landscape structure, and
        # would ask for a design of instances that buy nothing.
        groups = _instances([0.4, 0.4, 0.4], noise=0.2, seeds=40)

        assert decompose("alike", groups, conservative=False).between == pytest.approx(
            0.0, abs=0.05
        )

    def test_three_draws_cannot_show_even_that_much(self):
        # The same data, read conservatively, and the reason the tier cannot
        # size itself: on two degrees of freedom the bound admits an instance
        # term three times the seed term, from instances that are identical by
        # construction. This is not a defect in the estimator -- it is what
        # three draws are worth, and it is why the pilot draws more of them.
        groups = _instances([0.4, 0.4, 0.4], noise=0.2, seeds=40)

        assert decompose("alike", groups).between > 0.1

    def test_instances_that_disagree_carry_one(self):
        # Same seed noise, offsets a quarter apart. The effect is genuinely a
        # property of the draw, and a design that varied only seeds would never
        # see it however many it ran.
        components = decompose("apart", _instances([0.0, 0.3, 0.6], noise=0.05, seeds=40))

        assert components.between > 0.15
        assert components.within == pytest.approx(0.05, abs=0.02)

    def test_the_instance_term_is_floored_at_zero(self):
        # The estimator is a difference of two variances and can go negative.
        # A negative standard deviation is not a smaller answer, it is no
        # answer, and propagating one would produce a complex draw count.
        components = decompose("alike", _instances([0.4, 0.4, 0.4], noise=0.5, seeds=5))

        assert components.between >= 0.0

    def test_the_conservative_reading_is_never_the_smaller_one(self):
        groups = _instances([0.0, 0.2, 0.5], noise=0.1, seeds=20)

        assert decompose("g", groups).between >= decompose("g", groups, conservative=False).between

    def test_one_instance_cannot_be_decomposed(self):
        with pytest.raises(ValueError, match="at least 2 instances"):
            decompose("one", _instances([0.3], noise=0.1, seeds=20))

    def test_one_seed_in_an_instance_cannot_be_decomposed(self):
        with pytest.raises(ValueError, match="at least 2 seeds"):
            decompose("thin", [np.array([0.1, 0.2]), np.array([0.3])])


class TestTheDesignEffect:
    """The arithmetic that makes pseudo-replication a number rather than a word."""

    def test_one_seed_per_instance_pools_honestly(self):
        # With one seed per draw there is nothing to pool *within* an instance,
        # so the pooled reading and the honest one coincide. Any other answer
        # here would mean the factor was measuring the wrong thing.
        components = VarianceComponents(
            name="x", between=0.1, within=0.1, instances=5, seeds=1.0, conservative=False
        )

        assert components.design_effect(seeds=1) == pytest.approx(1.0)

    def test_more_seeds_make_the_pooled_reading_worse_not_better(self):
        # The trap this whole path exists for. Running more seeds narrows the
        # pooled interval without making it any more true, so the factor by
        # which it is wrong *grows* with the seed count -- which is why a tier at
        # 100 seeds and three draws is a worse offender than one at 10.
        components = VarianceComponents(
            name="x", between=0.05, within=0.1, instances=3, seeds=10.0, conservative=False
        )

        assert components.design_effect(seeds=100) > components.design_effect(seeds=10) > 1.0

    def test_no_instance_term_means_no_penalty(self):
        components = VarianceComponents(
            name="x", between=0.0, within=0.1, instances=3, seeds=10.0, conservative=False
        )

        assert components.design_effect(seeds=100) == pytest.approx(1.0)


class TestTheDrawCount:
    """It reads variance, never the effect, and it fails toward a larger design."""

    def test_it_cannot_see_which_way_the_effect_went(self):
        # The guarantee the signature is supposed to give, tested rather than
        # trusted. Shifting every observation by a constant changes the effect
        # and its sign while leaving both variances untouched; if the count
        # moved, the rule could be re-run until the answer was agreeable.
        groups = _instances([0.0, 0.2, 0.5], noise=0.1, seeds=20)
        shifted = [group - 5.0 for group in groups]

        assert draws_needed(decompose("a", groups), margin=MARGIN, seeds=100) == draws_needed(
            decompose("b", shifted), margin=MARGIN, seeds=100
        )

    def test_a_thinner_pilot_asks_for_more_draws(self):
        # The polarity that makes the number safe. Fewer instances widen the
        # bound on the instance term, which asks for a larger design -- the
        # direction that costs compute rather than the direction that shrinks a
        # reported requirement.
        offsets = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.05]
        thin = decompose("thin", _instances(offsets[:3], noise=0.1, seeds=20))
        thick = decompose("thick", _instances(offsets, noise=0.1, seeds=20))

        assert draws_needed(thin, margin=MARGIN, seeds=100) > draws_needed(
            thick, margin=MARGIN, seeds=100
        )

    def test_seeds_cannot_buy_away_the_instance_term(self):
        # The claim the framing rests on, as arithmetic. With a real instance
        # term, multiplying the seed count by ten leaves the draw count where it
        # was; the seed axis is precision on the wrong quantity.
        components = VarianceComponents(
            name="x", between=0.08, within=0.15, instances=15, seeds=10.0, conservative=True
        )

        assert draws_needed(components, margin=MARGIN, seeds=100) == draws_needed(
            components, margin=MARGIN, seeds=1000
        )

    def test_the_floor_holds_when_the_arithmetic_asks_for_less(self):
        # A design with no measurable instance term would otherwise be sized at
        # one draw, which is the fixed-instance design the whole exercise
        # replaces.
        components = VarianceComponents(
            name="x", between=0.0, within=0.01, instances=15, seeds=10.0, conservative=True
        )

        assert draws_needed(components, margin=MARGIN, seeds=100, floor=6) == 6

    def test_a_tighter_margin_asks_for_more(self):
        components = VarianceComponents(
            name="x", between=0.05, within=0.15, instances=15, seeds=10.0, conservative=True
        )

        assert draws_needed(components, margin=0.02, seeds=100) > draws_needed(
            components, margin=0.05, seeds=100
        )

    def test_a_margin_of_zero_is_refused(self):
        components = VarianceComponents(
            name="x", between=0.05, within=0.15, instances=5, seeds=10.0, conservative=True
        )

        with pytest.raises(ValueError, match="margin must be positive"):
            draws_needed(components, margin=0.0, seeds=100)


class TestPooling:
    """One design's components from several arms', and nothing across designs."""

    def test_it_lands_between_the_arms_it_pooled(self):
        groups = _instances([0.0, 0.2, 0.4], noise=0.1, seeds=20)
        quiet = decompose("quiet", _instances([0.0, 0.0, 0.0], noise=0.1, seeds=20))
        loud = decompose("loud", groups)

        pooled = pool_components("both", [quiet, loud])

        assert quiet.between <= pooled.between <= loud.between

    def test_components_from_different_designs_cannot_be_pooled(self):
        # A pooled variance across two designs describes neither, and the number
        # would then be quoted for whichever design the reader had in mind.
        few = decompose("few", _instances([0.0, 0.3], noise=0.1, seeds=20))
        many = decompose("many", _instances([0.0, 0.3, 0.6], noise=0.1, seeds=20))

        with pytest.raises(ValueError, match="different designs"):
            pool_components("mixed", [few, many])

    def test_one_point_estimate_makes_the_pool_a_point_estimate(self):
        # Calling a pooled figure a bound when one input was not one is the
        # single claim this path must not make by accident: a design sized on it
        # would be sized on less evidence than its label promises.
        groups = _instances([0.0, 0.2, 0.4], noise=0.1, seeds=20)
        pooled = pool_components(
            "mixed",
            [decompose("a", groups), decompose("b", groups, conservative=False)],
        )

        assert not pooled.conservative


class TestTheSignFloor:
    """A count no seed count can rescue, which is why it is arithmetic."""

    def test_three_draws_cannot_reach_significance_at_all(self):
        # The finding that condemns `REPLICATION_SEEDS` without any measurement:
        # unanimity over three draws is the strongest verdict the design can
        # produce and it is p = 0.25.
        assert unanimity_p(3, 3) == pytest.approx(0.25)

    def test_the_floor_is_the_first_count_that_clears(self):
        floor = unanimity_floor(0.05)

        assert unanimity_p(floor, floor) <= 0.05
        assert unanimity_p(floor - 1, floor - 1) > 0.05

    def test_a_split_verdict_is_not_evidence(self):
        assert unanimity_p(6, 4) > 0.05

    def test_a_tighter_alpha_raises_the_floor(self):
        assert unanimity_floor(0.01) > unanimity_floor(0.05)

    def test_more_draws_agreeing_than_were_run_is_refused(self):
        with pytest.raises(ValueError, match="cannot agree"):
            unanimity_p(3, 4)


class TestTheComparisonItStandsOn:
    """The existing paired machinery, at the edges the new callers reach."""

    def test_a_one_sample_interval_is_the_same_call_against_zeros(self):
        # How the across-draw interval is built: the per-draw means against a
        # vector of zeros. Reusing `compare` rather than writing a second
        # statistic is what stops the two disagreeing about the same numbers.
        means = np.array([0.2, 0.3, 0.1])

        outcome = compare("across", means, np.zeros_like(means))

        assert outcome.mean == pytest.approx(0.2)
        assert outcome.wins == 3
        assert outcome.n == 3
