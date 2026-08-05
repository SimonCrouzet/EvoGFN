"""Tests for the benchmarking facility.

What is under test is not whether numbers come out, but whether they mean what
the report says. Three things would make a comparison silently wrong: pairing
arms that did not share a seed, comparing arms run at different budgets, and
reporting a loss as though larger were better. Each has its own test.
"""

import numpy as np
import pytest

from evogfn.algorithms.base import Sampler
from evogfn.benchmark import (
    ML_CONVENTION,
    PLATE,
    WET_LAB_PROTOCOLS,
    Protocol,
    compare,
    round_sweep,
    run_benchmark,
    seeds_needed,
)
from evogfn.core.types import Alphabet
from evogfn.landscapes.base import FitnessLandscape
from evogfn.loop import Campaign

ALPHABET = Alphabet.from_string("ACGT")
LENGTH = 6


class Additive(FitnessLandscape):
    @property
    def alphabet(self):
        return ALPHABET

    @property
    def sequence_length(self):
        return LENGTH

    @property
    def optimum(self):
        return np.array([float(LENGTH)])

    def _evaluate(self, sequences):
        return (sequences == 1).sum(axis=1, keepdims=True).astype(np.float64)


class Uniform(Sampler):
    """Proposes uniformly. `bias` tilts it toward the optimum."""

    def __init__(self, seed, bias=0.0):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._bias = bias

    def propose(self, n):
        self._count(n)
        drawn = self._rng.integers(0, ALPHABET.size, size=(n, LENGTH), dtype=np.int32)
        if self._bias:
            flip = self._rng.random((n, LENGTH)) < self._bias
            drawn = np.where(flip, 1, drawn)
        return drawn


class Stuck(Sampler):
    """One design and nothing else, which is how a campaign runs out.

    The campaign will not re-measure a design an earlier round already assayed,
    so this fills its first plate and can then propose nothing the run is allowed
    to charge for.
    """

    def propose(self, n):
        self._count(n)
        return np.zeros((n, LENGTH), dtype=np.int32)


def arm(bias):
    def build(seed, protocol):
        return Campaign(
            landscape=Additive(),
            sampler=Uniform(seed, bias=bias),
            rounds=protocol.rounds,
            batch_size=protocol.batch_size,
            pool_size=max(protocol.batch_size * 4, 32),
        )

    return build


def stuck_arm():
    """An arm that exhausts on every seed."""

    def build(seed, protocol):  # noqa: ARG001 - this arm ignores its seed
        return Campaign(
            landscape=Additive(),
            sampler=Stuck(),
            rounds=protocol.rounds,
            batch_size=protocol.batch_size,
            pool_size=32,
        )

    return build


def flaky_arm():
    """An arm that exhausts on even seeds and runs normally on odd ones.

    Partial failure is the case worth testing: an arm that fails everywhere has
    no mean to protect, while one that fails half the time has seven good
    campaigns for a reader to lose.
    """

    def build(seed, protocol):
        return Campaign(
            landscape=Additive(),
            sampler=Stuck() if seed % 2 == 0 else Uniform(seed, bias=0.2),
            rounds=protocol.rounds,
            batch_size=protocol.batch_size,
            pool_size=32,
        )

    return build


class TestProtocol:
    def test_the_budget_is_the_product(self):
        assert Protocol(rounds=4, batch_size=96).budget == 384

    def test_it_reports_plates(self):
        assert Protocol(rounds=4, batch_size=96).plates == pytest.approx(4.0)

    def test_the_wet_lab_protocols_are_three_digit(self):
        # The grounding for C5. Every real campaign surveyed lands in the
        # hundreds; a protocol here that did not would be a transcription error.
        for protocol in WET_LAB_PROTOCOLS:
            assert 50 <= protocol.budget <= 999, f"{protocol!r} is not a wet-lab budget"

    def test_the_ml_convention_is_an_order_of_magnitude_larger(self):
        # The gap the survey found, encoded so it cannot be forgotten.
        largest_lab = max(p.budget for p in WET_LAB_PROTOCOLS)
        assert min(p.budget for p in ML_CONVENTION) > largest_lab

    def test_it_reports_when_the_mutation_budget_does_nothing(self):
        # GB1: four sites, four mutations. Every sequence is reachable, so the
        # constraint is vacuous and a result there says nothing about search
        # under a mutation limit.
        vacuous = Protocol(rounds=4, batch_size=PLATE, max_mutations=4)
        assert not vacuous.constrains_search(sequence_length=4)
        assert vacuous.constrains_search(sequence_length=64)

    def test_no_constraint_declared_is_not_a_constraint(self):
        assert not Protocol(rounds=1, batch_size=8).constrains_search(sequence_length=10)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"rounds": 0, "batch_size": 8}, "rounds must be at least 1"),
            ({"rounds": 1, "batch_size": 0}, "batch_size must be at least 1"),
            ({"rounds": 1, "batch_size": 8, "max_mutations": 0}, "max_mutations"),
        ],
    )
    def test_impossible_designs_are_refused(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            Protocol(**kwargs)


class TestRoundSweep:
    def test_it_holds_the_budget_roughly_fixed(self):
        for protocol in round_sweep(384):
            assert abs(protocol.budget - 384) <= protocol.batch_size

    def test_it_spans_many_rounds_to_few(self):
        sweep = round_sweep(384)
        assert sweep[0].rounds > sweep[-1].rounds

    def test_it_stops_at_a_plausible_number_of_rounds(self):
        # Rounds are bounded by turnaround, not budget. Nobody runs 96 rounds.
        assert all(p.rounds <= 8 for p in round_sweep(768))

    def test_a_non_positive_budget_is_refused(self):
        with pytest.raises(ValueError, match="budget must be at least 1"):
            round_sweep(0)


class TestPairedComparison:
    def test_a_loss_is_compared_in_the_right_direction(self):
        # The error that would invert every regret table in the paper.
        better = np.array([1.0, 1.0, 1.0])
        worse = np.array([2.0, 2.0, 2.0])
        assert compare("x", better, worse, higher_is_better=False).mean > 0
        assert compare("x", better, worse, higher_is_better=True).mean < 0

    def test_pairing_removes_shared_variance(self):
        # The reason to pair at all: a constant per-seed offset should not
        # widen the interval, because both arms felt it equally.
        offset = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        first, second = offset + 1.0, offset
        result = compare("x", first, second)
        assert result.mean == pytest.approx(1.0)
        assert result.low == pytest.approx(1.0)

    def test_the_win_rate_is_reported_separately_from_the_mean(self):
        # A method winning big on one seed and losing narrowly on four is not
        # the same as one winning everywhere, and the mean alone hides that.
        first = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
        second = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        result = compare("x", first, second)
        assert result.mean > 0
        assert result.wins == 1

    def test_an_identical_pair_is_not_significant(self):
        same = np.array([1.0, 2.0, 3.0, 4.0])
        assert not compare("x", same, same).significant

    def test_a_clear_effect_is_significant(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=40)
        assert compare("x", base + 2.0, base).significant

    def test_mismatched_seed_counts_are_refused(self):
        with pytest.raises(ValueError, match="cannot pair"):
            compare("x", np.zeros(5), np.zeros(4))

    def test_a_single_seed_is_refused(self):
        # A statistic from one seed is a division by zero dressed as a result.
        with pytest.raises(ValueError, match="at least 2 seeds"):
            compare("x", np.zeros(1), np.zeros(1))


class TestSeedsNeeded:
    def test_a_smaller_effect_needs_more_seeds(self):
        # The differences must carry real noise: a constant offset has zero
        # variance, so any sample resolves it and the question is empty.
        rng = np.random.default_rng(1)
        base = rng.normal(size=20)
        big = seeds_needed(compare("x", base + 1.0 + rng.normal(size=20), base))
        small = seeds_needed(compare("x", base + 0.2 + rng.normal(size=20), base))
        assert small > big

    def test_no_effect_needs_no_answer(self):
        same = np.array([1.0, 2.0, 3.0, 4.0])
        assert seeds_needed(compare("x", same, same)) == 0


class TestHarness:
    def test_every_arm_sees_the_same_seeds(self):
        result = run_benchmark(
            Additive(),
            {"a": arm(0.0), "b": arm(0.3)},
            Protocol(rounds=2, batch_size=8),
            seeds=[0, 1, 2],
        )
        assert result.seeds == (0, 1, 2)
        assert all(len(a.best) == 3 for a in result.arms.values())

    def test_it_detects_a_genuinely_better_arm(self):
        result = run_benchmark(
            Additive(),
            {"biased": arm(0.5), "plain": arm(0.0)},
            Protocol(rounds=3, batch_size=8),
            seeds=list(range(12)),
        )
        comparison = result.against("plain", metric="regret")[0]
        assert comparison.mean > 0
        assert comparison.significant

    def test_it_reports_no_difference_between_identical_arms(self):
        # The property that keeps the harness honest: two copies of one method
        # must not separate, however the report is phrased.
        result = run_benchmark(
            Additive(),
            {"one": arm(0.2), "two": arm(0.2)},
            Protocol(rounds=2, batch_size=8),
            seeds=list(range(8)),
        )
        assert not result.against("one", metric="regret")[0].significant

    def test_underspending_is_surfaced(self):
        result = run_benchmark(
            Additive(), {"stuck": stuck_arm()}, Protocol(rounds=3, batch_size=8), seeds=[0, 1]
        )
        assert result.arms["stuck"].underspent
        assert "underspent" in result.report()

    def test_a_seed_that_exhausted_does_not_erase_the_ones_that_did_not(self):
        # `nan` in one seed's `best` makes every figure on the arm's line `nan`
        # if it is averaged in, so one exhausted seed out of eight would hide
        # seven perfectly good campaigns behind a broken-looking row.
        result = run_benchmark(
            Additive(),
            {"flaky": flaky_arm()},
            Protocol(rounds=3, batch_size=8),
            seeds=list(range(8)),
        )
        arm_result = result.arms["flaky"]

        assert arm_result.failed == 4
        assert np.isfinite(arm_result.regret[np.isfinite(arm_result.regret)]).any()
        assert "nan" not in arm_result.summary()

    def test_the_seeds_it_lost_are_named_rather_than_dropped(self):
        # The other half. A mean quietly taken over survivors reports a subset
        # while the header goes on claiming the full seed count, which is the
        # same absence -- an arm that looks complete and is not.
        result = run_benchmark(
            Additive(),
            {"flaky": flaky_arm()},
            Protocol(rounds=3, batch_size=8),
            seeds=list(range(8)),
        )
        assert "4 exhausted" in result.arms["flaky"].summary()
        assert "4 exhausted" in result.report()

    def test_the_report_names_the_budget(self):
        result = run_benchmark(
            Additive(),
            {"a": arm(0.0), "b": arm(0.2)},
            Protocol(rounds=2, batch_size=8),
            seeds=[0, 1, 2],
        )
        assert "16" in result.report()

    def test_an_inconclusive_comparison_names_its_price(self):
        # An underpowered null and a real null look identical unless the report
        # says how many seeds would tell them apart.
        result = run_benchmark(
            Additive(),
            {"a": arm(0.2), "b": arm(0.21)},
            Protocol(rounds=2, batch_size=8),
            seeds=list(range(4)),
        )
        report = result.report(reference="b")
        assert "inconclusive" in report or "significant" not in report

    def test_diversity_can_be_compared_too(self):
        result = run_benchmark(
            Additive(),
            {"a": arm(0.0), "b": arm(0.6)},
            Protocol(rounds=2, batch_size=8),
            seeds=list(range(6)),
        )
        assert result.against("a", metric="diversity")

    def test_an_unknown_reference_is_refused(self):
        result = run_benchmark(
            Additive(), {"a": arm(0.0)}, Protocol(rounds=1, batch_size=8), seeds=[0, 1]
        )
        with pytest.raises(KeyError, match="no arm named"):
            result.against("nope")

    def test_an_unknown_metric_is_refused(self):
        result = run_benchmark(
            Additive(),
            {"a": arm(0.0), "b": arm(0.1)},
            Protocol(rounds=1, batch_size=8),
            seeds=[0, 1],
        )
        with pytest.raises(ValueError, match="unknown metric"):
            result.against("a", metric="vibes")

    def test_running_no_arms_is_refused(self):
        with pytest.raises(ValueError, match="no arms"):
            run_benchmark(Additive(), {}, Protocol(rounds=1, batch_size=8))

    def test_a_single_seed_is_refused(self):
        with pytest.raises(ValueError, match="at least 2 seeds"):
            run_benchmark(Additive(), {"a": arm(0.0)}, Protocol(rounds=1, batch_size=8), seeds=[0])
