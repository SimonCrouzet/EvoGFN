"""The pre-declared saturation rule, exercised where it decides things.

The properties under test are the ones the rule's whole credibility rests on:
that too few seeds fails toward "not saturated", that the indifference margin
cannot be widened after the fact, that a rung cannot be inserted to manufacture
a smaller step, and that appending a rung above can only ever revoke a
declaration. Everything else about this module is arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from evogfn.benchmark.saturation import (
    INDIFFERENCE_MARGIN,
    SATURATION_SEEDS,
    SEED_CAP,
    TOPUP_SPREAD,
    Verdict,
    pooled_spread,
    required_seeds,
    saturation,
)


def rung(level: float, *, n: int = 400, spread: float = 0.0, seed: int = 0) -> dict[int, float]:
    """One rung's per-seed regret: a constant level plus optional paired noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, spread, size=n) if spread else np.zeros(n)
    return {i: float(level + noise[i]) for i in range(n)}


def flat_ladder(budgets=(75, 150, 300, 600), *, knee=150, drop=0.10, n=400):
    """A ladder that improves up to `knee` and is exactly flat above it."""
    level = 0.6
    ladder = {}
    for budget in budgets:
        ladder[budget] = rung(level, n=n)
        if budget < knee:
            level -= drop
    return ladder


class TestTheRule:
    def test_finds_the_smallest_rung_nothing_above_it_beats(self):
        verdict = saturation(flat_ladder())
        assert verdict.budget == 150
        assert verdict.measured
        assert verdict.confirming == 2

    def test_a_still_improving_ladder_declares_nothing(self):
        ladder = {b: rung(0.6 - 0.1 * i) for i, b in enumerate((75, 150, 300, 600))}
        verdict = saturation(ladder)
        assert verdict.budget is None
        assert not verdict.measured
        # The number reported in a knee's place is the bound on what is left.
        assert verdict.residual == pytest.approx(0.1, abs=1e-9)
        assert "cost ceiling we chose" in verdict.reason

    def test_a_late_unbounded_doubling_vetoes_an_earlier_declaration(self):
        # Bounded at 75->150 and 150->300, then a real gain at 300->600. The
        # "and every doubling above" clause is what stops 75 being declared.
        ladder = {75: rung(0.5), 150: rung(0.5), 300: rung(0.5), 600: rung(0.3)}
        assert saturation(ladder).budget is None

    def test_appending_a_rung_above_can_only_revoke_never_lower(self):
        settled = saturation(flat_ladder())
        assert settled.budget == 150
        extended = dict(flat_ladder())
        extended[1200] = rung(0.3)  # a real gain at the new top
        assert saturation(extended).budget is None

    def test_a_worsening_curve_fires_at_the_cheapest_rung(self):
        # More budget is worse, so `high` is small and the rule takes the
        # cheapest budget nothing larger beats. The mean is what says this is
        # over-optimisation rather than a plateau, and it is carried.
        ladder = {12: rung(0.30), 25: rung(0.40), 50: rung(0.50), 100: rung(0.60)}
        verdict = saturation(ladder, axis="generations")
        assert verdict.budget == 12
        assert all(pair.comparison.mean < 0 for pair in verdict.doublings)

    def test_every_adjacent_pair_is_reported_not_only_the_one_that_fired(self):
        verdict = saturation(flat_ladder())
        assert len(verdict.doublings) == len(verdict.budgets) - 1
        assert [(p.smaller, p.larger) for p in verdict.doublings] == [
            (75, 150),
            (150, 300),
            (300, 600),
        ]


class TestFailsTowardNotSaturated:
    """The direction that makes an underpowered run cost compute, not a claim."""

    def test_a_noisy_ladder_at_few_seeds_refuses_to_declare(self):
        # Genuinely flat, but at 8 seeds with the measured paired spread the
        # interval is far too wide to exclude a gain worth having.
        ladder = {b: rung(0.5, n=8, spread=0.2, seed=i) for i, b in enumerate((75, 150, 300, 600))}
        assert saturation(ladder).budget is None

    def test_the_same_ladder_at_enough_seeds_declares(self):
        ladder = {
            b: rung(0.5, n=5000, spread=0.15, seed=i) for i, b in enumerate((75, 150, 300, 600))
        }
        assert saturation(ladder).budget == 75

    def test_the_bound_is_the_upper_interval_and_not_the_mean(self):
        # A mean well under the margin whose interval reaches far above it: the
        # doubling helps hugely on half its seeds and hurts as much on the rest.
        gains = [0.5, -0.5, 0.5, -0.5, 0.5, -0.49]
        ladder = {
            75: dict(enumerate([0.5] * len(gains))),
            150: {seed: 0.5 - gain for seed, gain in enumerate(gains)},
        }
        pair = saturation(ladder).doublings[0]
        assert abs(pair.comparison.mean) < INDIFFERENCE_MARGIN
        assert not pair.bounded
        assert pair.verdict is Verdict.INCONCLUSIVE


class TestTheMarginCannotBeLoosened:
    def test_a_wider_margin_is_refused(self):
        with pytest.raises(ValueError, match="may be tightened, never loosened"):
            saturation(flat_ladder(), margin=INDIFFERENCE_MARGIN * 2)

    def test_a_tighter_margin_is_accepted(self):
        assert saturation(flat_ladder(), margin=INDIFFERENCE_MARGIN / 2).budget == 150

    def test_a_wider_margin_is_refused_when_sizing_seeds_too(self):
        with pytest.raises(ValueError, match="looser than the declared"):
            required_seeds(0.2, margin=0.05)

    @pytest.mark.parametrize("margin", [0.0, -0.01])
    def test_a_nonpositive_margin_is_refused(self, margin):
        with pytest.raises(ValueError, match="must be positive"):
            saturation(flat_ladder(), margin=margin)


class TestTheLadderMustBeADoublingLadder:
    def test_an_inserted_rung_is_refused(self):
        with pytest.raises(ValueError, match="not a doubling"):
            saturation({75: rung(0.5), 100: rung(0.5), 150: rung(0.5)})

    def test_an_order_of_magnitude_gap_is_refused(self):
        with pytest.raises(ValueError, match="not a doubling"):
            saturation({75: rung(0.5), 750: rung(0.5)})

    def test_integer_rounding_of_a_doubling_is_accepted(self):
        # 12 -> 25 is a ratio of 2.083, which is the generations ladder's bottom.
        saturation({12: rung(0.5), 25: rung(0.5), 50: rung(0.5)}, axis="generations")

    def test_one_rung_is_refused(self):
        with pytest.raises(ValueError, match="at least two rungs"):
            saturation({300: rung(0.5)})


class TestPairing:
    def test_rungs_are_compared_on_the_seeds_they_share(self):
        # A topped-up rung against one that was not topped up.
        ladder = {75: rung(0.6, n=250), 150: rung(0.5, n=100)}
        assert saturation(ladder).doublings[0].n == 100

    def test_rungs_sharing_nothing_are_refused(self):
        ladder = {75: {0: 0.5, 1: 0.5}, 150: {9: 0.5, 10: 0.5}}
        with pytest.raises(ValueError, match="cannot be paired"):
            saturation(ladder)

    def test_a_non_finite_regret_is_refused_rather_than_averaged_over(self):
        ladder = {75: {0: 0.5, 1: float("nan"), 2: 0.5}, 150: {0: 0.5, 1: 0.5, 2: 0.5}}
        with pytest.raises(ValueError, match="non-finite regret"):
            saturation(ladder)

    def test_a_positive_gain_means_the_larger_budget_won(self):
        ladder = {75: rung(0.6), 150: rung(0.5)}
        assert saturation(ladder).doublings[0].comparison.mean == pytest.approx(0.1)


class TestSettled:
    def test_one_bounded_doubling_above_the_knee_is_provisional(self):
        ladder = {75: rung(0.6), 150: rung(0.5), 300: rung(0.5)}
        verdict = saturation(ladder, ceiling=2400)
        assert verdict.budget == 150
        assert verdict.confirming == 1
        assert not verdict.settled
        assert "provisional" in verdict.reason

    def test_two_bounded_doublings_settle_it(self):
        verdict = saturation(flat_ladder(), ceiling=2400)
        assert verdict.confirming == 2
        assert verdict.settled

    def test_reaching_the_declared_ceiling_settles_it_either_way(self):
        ladder = {600: rung(0.6), 1200: rung(0.5), 2400: rung(0.5)}
        verdict = saturation(ladder, ceiling=2400)
        assert verdict.confirming == 1
        assert verdict.settled

    def test_an_unmeasured_ladder_at_the_ceiling_owes_no_further_rung(self):
        ladder = {600: rung(0.7), 1200: rung(0.6), 2400: rung(0.5)}
        verdict = saturation(ladder, ceiling=2400)
        assert not verdict.measured
        assert verdict.settled


class TestTheTopUpTrigger:
    def test_it_reads_a_spread_and_nothing_else(self):
        # The signature is the guarantee: there is no argument through which the
        # effect, its sign, or a verdict could reach this function.
        assert required_seeds(0.30) == required_seeds(0.30, planned=SATURATION_SEEDS)

    def test_a_spread_within_plan_asks_for_no_top_up(self):
        assert required_seeds(TOPUP_SPREAD) == SATURATION_SEEDS
        assert required_seeds(0.0) == SATURATION_SEEDS

    def test_a_wider_spread_tops_up_in_whole_blocks(self):
        needed = required_seeds(0.20)
        assert needed > SATURATION_SEEDS
        assert needed % 50 == 0

    def test_the_cap_is_a_ceiling_declared_in_advance(self):
        assert required_seeds(5.0) == SEED_CAP

    def test_a_negative_spread_is_refused(self):
        with pytest.raises(ValueError, match="must not be negative"):
            required_seeds(-0.1)

    def test_pooling_weights_by_degrees_of_freedom(self):
        ladder = {b: rung(0.5, n=200, spread=0.2, seed=i) for i, b in enumerate((75, 150, 300))}
        spread = pooled_spread(saturation(ladder).doublings)
        # Two independent rungs at sd 0.2 give paired differences at sd ~0.283.
        assert spread == pytest.approx(0.2 * math.sqrt(2), rel=0.15)

    def test_pooling_an_empty_ladder_is_zero_rather_than_an_error(self):
        assert pooled_spread(()) == 0.0
