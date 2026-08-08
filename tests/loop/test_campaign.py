"""Tests for the campaign round engine.

The properties under test are budget properties. A campaign that quietly spends
more oracle calls than it was given produces numbers that look like results and
are not, and no assertion about fitness would catch it -- so the accounting is
tested harder than the optimisation.
"""

import itertools

import numpy as np
import pytest

from evogfn.acquisition import DiverseTopK, ExpectedImprovement, Greedy, TopK
from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines.mutagenesis import RandomMutagenesis
from evogfn.core.types import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.base import FitnessLandscape
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.loop import Campaign
from evogfn.surrogate import DeepEnsemble

ALPHABET = Alphabet.from_string("ABCD")
LENGTH = 6


class CountingLandscape(FitnessLandscape):
    """Counts every evaluation, so the budget can be checked from the outside."""

    def __init__(self, *, infeasible_token=None):
        self.calls = 0
        self._infeasible_token = infeasible_token

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
        self.calls += sequences.shape[0]
        value = (sequences == 1).sum(axis=1, keepdims=True).astype(np.float64)
        if self._infeasible_token is not None:
            blocked = (sequences == self._infeasible_token).any(axis=1)
            value[blocked] = -np.inf
        return value


class RandomSampler(Sampler):
    """Uniform proposals. Learns nothing, which makes it a clean control."""

    def __init__(self, seed=0):
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def propose(self, n):
        self._count(n)
        return self._rng.integers(0, ALPHABET.size, size=(n, LENGTH), dtype=np.int32)


class CollapsedSampler(Sampler):
    """Proposes one sequence over and over -- the mode-collapse failure mode."""

    def propose(self, n):
        self._count(n)
        return np.ones((n, LENGTH), dtype=np.int32)


def surrogate():
    return DeepEnsemble(n_tokens=ALPHABET.size, sequence_length=LENGTH, epochs=30, seed=0)


class TestBudget:
    def test_the_oracle_is_called_exactly_the_budget(self):
        landscape = CountingLandscape()
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert landscape.calls == 24
        assert result.oracle_calls == 24

    def test_the_ledger_agrees_with_the_landscape(self):
        # Two independent counts of the same thing. If the ledger could drift
        # from the oracle, every budget-indexed claim in the paper would be
        # unfalsifiable from inside the repo.
        landscape = CountingLandscape()
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=4,
            batch_size=5,
            pool_size=40,
        ).run()
        assert result.oracle_calls == landscape.calls == len(result.sequences)

    def test_training_the_sampler_is_not_charged(self):
        # The structural error this loop exists to prevent. The sampler generates
        # a 512-candidate pool per round; only the selected batch is measured.
        landscape = CountingLandscape()
        sampler = RandomSampler()
        result = Campaign(
            landscape=landscape,
            sampler=sampler,
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=512,
        ).run()
        assert landscape.calls == 24
        assert sampler.proposals_made > 1000
        assert result.proposals > result.oracle_calls

    def test_the_budget_is_rounds_times_batch(self):
        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            rounds=4,
            batch_size=96,
            pool_size=1024,
        )
        assert campaign.budget == 384

    def test_the_default_budget_matches_real_campaigns(self):
        # 4 x 96 = 384. ALDE screened 396 as six 96-well plates over three
        # rounds; LaMBO-2's wet lab measured 374. The 1,000-10,000 evaluations
        # common in iterative benchmarks is a different experiment.
        campaign = Campaign(landscape=CountingLandscape(), sampler=RandomSampler())
        assert campaign.budget == 384

    def test_a_pool_smaller_than_the_batch_is_refused(self):
        with pytest.raises(ValueError, match="nothing to select from"):
            Campaign(
                landscape=CountingLandscape(),
                sampler=RandomSampler(),
                batch_size=96,
                pool_size=32,
            )

    @pytest.mark.parametrize(
        "build",
        [
            lambda **kw: Campaign(rounds=0, pool_size=64, batch_size=8, **kw),
            lambda **kw: Campaign(batch_size=0, pool_size=64, **kw),
            lambda **kw: Campaign(pool_size=0, batch_size=8, **kw),
        ],
        ids=["rounds", "batch_size", "pool_size"],
    )
    def test_non_positive_sizes_are_refused(self, build):
        with pytest.raises(ValueError, match="at least 1"):
            build(landscape=CountingLandscape(), sampler=RandomSampler())


class TestRounds:
    def test_every_round_is_recorded(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=6,
            pool_size=48,
        ).run()
        assert [record.index for record in result.rounds] == [0, 1, 2]

    def test_best_so_far_never_decreases(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=4,
            batch_size=8,
            pool_size=64,
        ).run()
        trace = result.trace()
        assert trace == sorted(trace)

    def test_the_first_round_runs_without_a_surrogate(self):
        # There is nothing to fit it on. A campaign that tried would fail on the
        # empty-data check the surrogate raises.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=1,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.oracle_calls == 8

    def test_a_supplied_initial_design_is_measured_first(self):
        design = np.stack([np.full(LENGTH, i, dtype=np.int32) for i in range(4)])
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=2,
            batch_size=4,
            pool_size=32,
            initial_design=design,
        ).run()
        assert np.array_equal(result.sequences[:4], design)

    def test_a_short_initial_design_is_topped_up_rather_than_under_spent(self):
        # Round 0 is charged like every other round. A two-row opening design on
        # an eight-well plate used to leave six wells of the budget unspent with
        # nothing in the ledger saying so.
        landscape = CountingLandscape()
        design = np.stack([np.full(LENGTH, i, dtype=np.int32) for i in range(2)])
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            rounds=2,
            batch_size=8,
            pool_size=64,
            initial_design=design,
        ).run()
        assert landscape.calls == 16
        assert np.array_equal(result.sequences[:2], design)


def design_at(index):
    """The ``index``-th sequence, as its digits in base ``|alphabet|``.

    Module level so that two samplers can enumerate the *same* designs in the
    same order. A test that a design measured in round one comes back in round
    two only means anything if both rounds agree on what design number seven is.
    """
    digits = [(index // ALPHABET.size**place) % ALPHABET.size for place in range(LENGTH)]
    return np.asarray(digits, dtype=np.int32)


class RepetitiveSampler(Sampler):
    """Proposes from a tiny fixed menu, so a plate is mostly repeats.

    A stand-in for a converged genetic algorithm: it still produces something
    new every round, but most of any one plate is the same handful of designs.
    That is the case the plate rule is about -- a wholly collapsed sampler is a
    terminal condition, and this one is the ordinary expensive one.
    """

    def __init__(self, menu=4, seed=0):
        super().__init__()
        self._menu = menu
        self._rng = np.random.default_rng(seed)
        self._offset = 0

    def propose(self, n):
        self._count(n)
        # A fresh block of `menu` designs each round, drawn from with
        # replacement, so a plate of eight holds four designs twice over.
        base = np.arange(self._offset, self._offset + self._menu)
        return np.stack([self._design(i) for i in self._rng.choice(base, size=n)])

    def observe(self, sequences, values):  # noqa: ARG002 - it advances on the round, not the data
        self._offset += self._menu

    @staticmethod
    def _design(index):
        """The ``index``-th sequence, as its digits in base ``|alphabet|``."""
        return design_at(index)


class LingeringSampler(Sampler):
    """Proposes a window that slides slower than the plate consumes it.

    The cross-round case, and deliberately not `RepetitiveSampler`'s: every
    design in any one call here is distinct, so this sampler *cannot* produce a
    repeat within a plate. What it does produce is designs an earlier round has
    already measured, which is the other half of the accounting -- the half the
    campaign used to absorb without saying so.

    The two samplers together are what make the twin counters falsifiable. Each
    exercises exactly one of them, so a counter that were secretly measuring the
    other would come back zero where it must be positive.
    """

    def __init__(self, stride=4):
        super().__init__()
        self._stride = stride
        self._offset = 0

    def propose(self, n):
        self._count(n)
        # Distinct within the call, and starting from a window that advances by
        # less than the plate holds -- so every round overlaps the last one's
        # measurements and still has something new below the overlap.
        return np.stack([design_at(self._offset + i) for i in range(n)])

    def observe(self, sequences, values):  # noqa: ARG002 - it advances on the round, not the data
        self._offset += self._stride


class TestThePlateIsAlwaysFull:
    """The invariant every budget-indexed claim rests on.

    Measured before this was enforced: at a pool the size of the plate, a random
    arm assayed 6 wells of 8 and then 4, a genetic arm 4 and then 3. Half the
    oracle budget went unspent, every round reported success, and the number the
    paper indexes its results by was wrong by a factor of two.
    """

    @pytest.mark.parametrize("pool_size", [8, 9, 64], ids=["plate", "just-over", "library"])
    def test_every_round_spends_the_whole_plate(self, pool_size):
        landscape = CountingLandscape()
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=4,
            batch_size=8,
            pool_size=pool_size,
        ).run()
        assert landscape.calls == 32
        assert [record.evaluated for record in result.rounds] == [8, 8, 8, 8]

    def test_a_duplicate_heavy_sampler_still_spends_the_whole_budget(self):
        # The failure this exists for. A sampler whose plate is mostly repeats
        # used to have the repeats silently dropped and the round shortened; now
        # the repeats are charged and the plate is filled either way.
        landscape = CountingLandscape()
        result = Campaign(
            landscape=landscape,
            sampler=RepetitiveSampler(),
            rounds=4,
            batch_size=8,
            pool_size=8,
        ).run()
        assert landscape.calls == 32
        assert result.oracle_calls == 32
        assert any(record.duplicates > 0 for record in result.rounds)

    def test_topping_up_does_not_launder_a_repeat(self):
        # Topping up serves the campaign's memory of earlier rounds and nothing
        # else. If it also collapsed within-plate repeats, a converged method
        # would get free measurements and convergence would look costless.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RepetitiveSampler(menu=2),
            rounds=2,
            batch_size=8,
            pool_size=8,
        ).run()
        first = result.sequences[:8]
        assert len({row.tobytes() for row in np.ascontiguousarray(first)}) < 8


class TestThePlateRule:
    def test_a_repeat_within_a_plate_is_charged(self):
        # A collapsed sampler's opening plate consumes eight wells and buys one
        # measurement's worth of information. Both halves of that are the
        # method's, and the ledger says so rather than hiding the first half.
        landscape = CountingLandscape()
        result = Campaign(
            landscape=landscape,
            sampler=CollapsedSampler(),
            rounds=1,
            batch_size=8,
            pool_size=64,
        ).run()
        assert landscape.calls == 8
        assert result.rounds[0].duplicates == 7
        assert result.rounds[0].duplicate_fraction == pytest.approx(7 / 8)

    def test_nothing_is_measured_in_two_different_rounds(self):
        # Across rounds the campaign remembers, which is what makes a round
        # informed by the last one rather than a re-run of it.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RepetitiveSampler(),
            rounds=4,
            batch_size=8,
            pool_size=8,
        ).run()
        start = 0
        earlier: set[bytes] = set()
        for record in result.rounds:
            plate = np.ascontiguousarray(result.sequences[start : start + record.evaluated])
            start += record.evaluated
            keys = {row.tobytes() for row in plate}
            assert not (keys & earlier), "a design was re-ordered in a later round"
            earlier |= keys

    def test_a_distinct_plate_holds_no_repeat_and_still_fills(self):
        # The `genetic+distinct` arm. Same sampler, same budget, different plate
        # rule -- and it has to fill the plate, or it would be measuring the
        # other rule's cost as a shorter campaign instead of as a duplicate share.
        landscape = CountingLandscape()
        result = Campaign(
            landscape=landscape,
            sampler=RepetitiveSampler(menu=12),
            rounds=4,
            batch_size=8,
            pool_size=8,
            distinct_batch=True,
        ).run()
        assert landscape.calls == 32
        assert all(record.duplicates == 0 for record in result.rounds)
        assert result.duplicate_fraction == 0.0

    def test_the_distinct_plate_diverges_from_the_bare_one(self):
        # Why this is an arm rather than a re-reading of the bare ledger:
        # dropping a repeat changes which design takes that well, so the two
        # campaigns measure different things from round one onward.
        def run(distinct):
            return Campaign(
                landscape=CountingLandscape(),
                sampler=RepetitiveSampler(menu=12, seed=3),
                rounds=3,
                batch_size=8,
                pool_size=8,
                distinct_batch=distinct,
            ).run()

        assert not np.array_equal(run(distinct=False).sequences, run(distinct=True).sequences)

    def test_a_sampler_that_can_produce_nothing_new_fails_loudly(self):
        # The alternative was stopping quietly, which reported a full-looking
        # ledger against a budget that was never spent. There is no honest way
        # to fill this plate, so the campaign says so instead of pretending.
        with pytest.raises(RuntimeError, match="cannot produce designs"):
            Campaign(
                landscape=CountingLandscape(),
                sampler=CollapsedSampler(),
                surrogate=surrogate(),
                rounds=4,
                batch_size=8,
                pool_size=64,
            ).run()

    def test_the_rounds_that_finished_survive_the_failure(self):
        # `run` raises, so its ledger is lost with it -- and a caller storing a
        # record of the failure would then have to write zeros, which say the
        # campaign measured nothing. It measured one plate. "Gave up in round one
        # having measured nothing" and "gave up in round four having measured
        # 288" are different findings, and zeros report them identically.
        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=CollapsedSampler(),
            surrogate=surrogate(),
            rounds=4,
            batch_size=8,
            pool_size=64,
        )
        with pytest.raises(RuntimeError, match="cannot produce designs"):
            campaign.run()

        assert len(campaign.completed_rounds) == 1
        assert sum(record.evaluated for record in campaign.completed_rounds) == 8

    def test_a_campaign_that_finished_reports_every_round(self):
        # The same field on the ordinary path, so a reader does not need to know
        # whether the run raised to know what it is looking at.
        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=RepetitiveSampler(menu=64),
            rounds=3,
            batch_size=8,
            pool_size=32,
        )
        result = campaign.run()

        assert campaign.completed_rounds == result.rounds

    def test_the_memory_of_earlier_rounds_can_be_turned_off(self):
        landscape = CountingLandscape()
        Campaign(
            landscape=landscape,
            sampler=CollapsedSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
            skip_measured=False,
        ).run()
        assert landscape.calls == 24


class TestTheTwoRepetitionCounts:
    """``duplicates`` and ``redundant``: one plate, one memory, never one number.

    What was silent before these ran together: the campaign skipped a candidate
    it had measured in an earlier round and recorded nothing about the skip, so
    the only visible trace was a proposals-to-screened gap that ``distinct_batch``
    also widens. A converged sampler produces both kinds of repeat at once, and a
    reader with one number had no way to tell which cost had produced it -- they
    would attribute a plate of duplicates to mode collapse across rounds, or the
    reverse, and either reading is a claim about the method.

    Each test below drives a sampler that can only produce one of the two, so a
    counter that were quietly measuring the other would read zero where it has
    to be positive.
    """

    def test_repeats_confined_to_one_plate_are_not_counted_against_the_memory(self):
        # `RepetitiveSampler` retires its menu every round, so it repeats itself
        # constantly and never re-proposes something already measured. Anything
        # non-zero in `redundant` here would be `duplicates` leaking into it.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RepetitiveSampler(menu=2),
            rounds=3,
            batch_size=8,
            pool_size=8,
        ).run()

        assert all(record.duplicates > 0 for record in result.rounds)
        assert all(record.redundant == 0 for record in result.rounds)
        assert all(record.redundant_fraction == 0.0 for record in result.rounds)

    def test_repeats_against_earlier_rounds_are_not_counted_against_the_plate(self):
        # The mirror image, and the case that had no number at all. Every call
        # this sampler makes holds distinct designs, so no well can hold a
        # repeat; what it does re-propose is what earlier rounds measured, and
        # the count of those has to rise as the memory fills.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=LingeringSampler(stride=4),
            rounds=3,
            batch_size=8,
            pool_size=16,
        ).run()
        # The memory is on, so every round has a count rather than a `None`.
        assert all(record.redundant is not None for record in result.rounds)
        redundant = [record.redundant or 0 for record in result.rounds]

        assert all(record.duplicates == 0 for record in result.rounds)
        assert redundant[0] == 0, "round zero has no earlier round to repeat"
        assert redundant == sorted(redundant)
        assert redundant[-1] > 0
        # Denominated in proposals, not in wells: a refused candidate never
        # reached one. Reading it against `evaluated` would make the two twins
        # look addable, and their sum is not a quantity.
        last = result.rounds[-1]
        assert last.redundant is not None
        assert last.redundant_fraction == pytest.approx(last.redundant / last.proposed)

    def test_a_campaign_with_no_memory_reports_nothing_rather_than_zero(self):
        # `skip_measured=False` is the ablation that removes the screening
        # entirely. Zero here would say "this sampler never repeated itself",
        # which is the opposite of what an unscreened campaign knows -- it
        # never looked. `nan` propagates; a zero would average into a column.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RepetitiveSampler(menu=2),
            rounds=3,
            batch_size=8,
            pool_size=8,
            skip_measured=False,
        ).run()

        assert all(record.redundant is None for record in result.rounds)
        assert all(np.isnan(record.redundant_fraction) for record in result.rounds)
        # And the duplicates twin is unaffected: the plate rule did not change.
        assert any(record.duplicates > 0 for record in result.rounds)


class TestFeasibility:
    def test_infeasible_designs_are_charged(self):
        # They cost the same to build. Not charging for them would make a
        # method that proposes unbuildable constructs look free, which is
        # exactly the comparison the masked sampler is meant to win.
        landscape = CountingLandscape(infeasible_token=3)
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.oracle_calls == 24
        assert result.feasible_fraction < 1.0

    def test_a_fully_feasible_campaign_reports_one(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=2,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.feasible_fraction == pytest.approx(1.0)

    def test_infeasible_values_do_not_poison_the_best(self):
        landscape = CountingLandscape(infeasible_token=3)
        result = Campaign(
            landscape=landscape,
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert np.isfinite(result.best_value)


class TestSurrogateEffect:
    def test_the_surrogate_improves_on_the_unassisted_sampler(self):
        # The point of the whole apparatus. Same sampler, same budget, same
        # seed -- the only difference is whether proposals are screened.
        blind = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(seed=1),
            rounds=4,
            batch_size=8,
            pool_size=256,
        ).run()
        guided = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(seed=1),
            surrogate=surrogate(),
            rounds=4,
            batch_size=8,
            pool_size=256,
        ).run()
        assert guided.best_value >= blind.best_value

    def test_it_runs_without_a_surrogate_at_all(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.oracle_calls == 24

    @pytest.mark.parametrize("rule", [Greedy(), ExpectedImprovement()])
    @pytest.mark.parametrize("selector", [TopK(), DiverseTopK(penalty=1.0)])
    def test_every_rule_and_selector_combination_runs(self, rule, selector):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            acquisition=rule,
            selector=selector,
            rounds=2,
            batch_size=6,
            pool_size=48,
        ).run()
        assert result.oracle_calls == 12


class TestResult:
    def test_regret_is_measured_against_the_true_optimum(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.simple_regret == pytest.approx(LENGTH - result.best_value)

    def test_regret_is_none_when_the_optimum_is_unknown(self):
        class Unknowable(CountingLandscape):
            @property
            def optimum(self):
                return None

        result = Campaign(
            landscape=Unknowable(),
            sampler=RandomSampler(),
            rounds=1,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.simple_regret is None
        assert "simple_regret" not in result.summary()

    def test_the_summary_carries_the_budget(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            rounds=2,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.summary()["oracle_calls"] == 16

    def test_the_rejection_ratio_exposes_wasted_generation(self):
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(),
            surrogate=surrogate(),
            rounds=2,
            batch_size=4,
            pool_size=256,
        ).run()
        assert result.rounds[-1].rejection_ratio > 10


class BallSampler(Sampler):
    """Proposes everything its environment can build, in enumeration order.

    Deterministic on purpose. Re-anchoring is a statement about *which designs
    are reachable*, and a stochastic sampler would blur that into a question
    about how lucky the draw was.
    """

    def __init__(self, env):
        super().__init__()
        self._env = env

    def propose(self, n):
        states = self._env.reachable_terminal_states()
        self._count(states.shape[0])
        return states[:n]

    def reanchored(self, env):
        return BallSampler(env)


def mutation_env(max_mutations=2, parent=None, transitions=None):
    return MutationEnvironment(
        np.zeros(LENGTH, dtype=np.int32) if parent is None else parent,
        ALPHABET,
        max_mutations=max_mutations,
        transitions=transitions,
    )


def small_ehrlich():
    """A toy Ehrlich whose optimum is several mutations out, with a budget of one.

    Chosen so the planted optimum is further than one round's mutation budget and
    well within the campaign's total, which is the regime the benchmark's real
    tasks are in -- 61 to 248 mutations away against a per-round budget of four.

    Sixteen positions rather than eight, which is not cosmetic: the one-mutation
    ball has 20 reachable designs against 13, and a campaign now fills every
    plate it opens, so a ball smaller than the budget is a campaign that cannot
    legally spend it. That is the right refusal and the wrong test instance.

    Returns:
        The landscape and a feasible wild type.
    """
    landscape = EhrlichLandscape(
        sequence_length=16,
        vocab_size=4,
        n_motifs=2,
        motif_length=2,
        quantization=2,
        max_spacing=2,
        transition_density=0.7,
        seed=1,
    )
    return landscape, landscape.feasible_sequence(seed=0)


def ball_of(landscape, wild_type, max_mutations):
    """The designs one anchor can reach, and the best value among them."""
    env = MutationEnvironment(
        wild_type,
        landscape.alphabet,
        max_mutations=max_mutations,
        transitions=landscape.transition_matrix,
    )
    return env, float(landscape.evaluate(env.reachable_terminal_states()).max())


class TestReanchoringIsOffByDefault:
    """Nothing already measured moves because this mechanism was added."""

    def test_a_campaign_without_an_environment_is_unchanged(self):
        # Bit-identical against a fixed seed. The numbers are hard-coded rather
        # than compared to a second run, so that a change to the loop cannot
        # move both sides together and still pass.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(seed=7),
            rounds=3,
            batch_size=8,
            pool_size=64,
        ).run()
        assert result.oracle_calls == 24
        assert result.best_value == 4.0
        assert result.trace() == [4.0, 4.0, 4.0]
        assert result.sequences[0].tolist() == [3, 2, 2, 3, 2, 3]
        assert int(result.sequences.sum()) == 232

    def test_supplying_an_environment_alone_changes_no_measurement(self):
        def run(**extra):
            return Campaign(
                landscape=CountingLandscape(),
                sampler=RandomSampler(seed=7),
                rounds=3,
                batch_size=8,
                pool_size=64,
                **extra,
            ).run()

        plain = run()
        watched = run(environment=mutation_env())
        assert np.array_equal(plain.sequences, watched.sequences)
        assert np.array_equal(plain.values, watched.values)
        assert plain.trace() == watched.trace()

    def test_a_fixed_anchor_never_leaves_the_wild_type(self):
        # The failure this mechanism exists to fix, stated as a measurement: with
        # the anchor held still, every round searches the same Hamming ball.
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=BallSampler(mutation_env(max_mutations=2)),
            rounds=4,
            batch_size=16,
            pool_size=256,
            environment=mutation_env(max_mutations=2),
        ).run()
        assert result.anchor_trace() == [0, 0, 0, 0]
        assert result.best_value <= 2.0


class TestReanchoringMovesTheSearch:
    def test_cumulative_distance_outgrows_the_per_round_budget(self):
        # The property the whole mechanism exists for. Two mutations per round
        # is the budget; after four rounds the search is standing further out
        # than that, which one fixed environment can never do.
        env = mutation_env(max_mutations=2)
        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=BallSampler(env),
            rounds=4,
            batch_size=16,
            pool_size=256,
            environment=env,
            reanchor=True,
        )
        result = campaign.run()
        assert result.anchor_trace() == [0, 1, 2, 3]
        assert max(result.anchor_trace()) > env.max_mutations
        moved = campaign.environment
        assert moved is not None
        assert moved.parent.tolist() == [1, 1, 1, 1, 0, 0]

    def test_it_reaches_fitness_a_fixed_anchor_cannot(self):
        def run(reanchor):
            env = mutation_env(max_mutations=2)
            return Campaign(
                landscape=CountingLandscape(),
                sampler=BallSampler(env),
                rounds=4,
                batch_size=16,
                pool_size=256,
                environment=env,
                reanchor=reanchor,
            ).run()

        assert run(reanchor=True).best_value > run(reanchor=False).best_value

    def test_the_ledger_names_the_design_each_round_started_from(self):
        env = mutation_env(max_mutations=2)
        result = Campaign(
            landscape=CountingLandscape(),
            sampler=BallSampler(env),
            rounds=3,
            batch_size=16,
            pool_size=256,
            environment=env,
            reanchor=True,
        ).run()
        anchors = [record.anchor for record in result.rounds]
        assert anchors[0] == (0, 0, 0, 0, 0, 0)
        # Each anchor is a design that was actually measured, not a construction.
        measured = {tuple(int(t) for t in row) for row in result.sequences}
        assert all(anchor in measured for anchor in anchors[1:])

    def test_the_anchor_only_moves_on_an_improvement(self):
        # A round that learns nothing must not walk the search off a peak it
        # has already found.
        env = mutation_env(max_mutations=2)
        result = Campaign(
            landscape=CountingLandscape(infeasible_token=None),
            sampler=BallSampler(env),
            rounds=6,
            batch_size=16,
            pool_size=256,
            environment=env,
            reanchor=True,
        ).run()
        distances = result.anchor_trace()
        assert distances == sorted(distances)
        assert all(
            second - first <= env.max_mutations for first, second in itertools.pairwise(distances)
        )


class TestProposalsStayInsideTheMovedEnvironment:
    def test_every_measured_design_is_reachable_from_that_round_s_anchor(self):
        landscape, wild_type = small_ehrlich()
        env = MutationEnvironment(
            wild_type,
            landscape.alphabet,
            max_mutations=1,
            transitions=landscape.transition_matrix,
        )
        result = Campaign(
            landscape=landscape,
            sampler=RandomMutagenesis(env, feasible_only=True, seed=0),
            rounds=4,
            batch_size=12,
            pool_size=64,
            environment=env,
            reanchor=True,
        ).run()

        start = 0
        for record in result.rounds:
            batch = result.sequences[start : start + record.evaluated]
            start += record.evaluated
            anchored = MutationEnvironment(
                np.array(record.anchor, dtype=np.int32),
                landscape.alphabet,
                max_mutations=1,
                transitions=landscape.transition_matrix,
            )
            assert anchored.is_reachable(batch).all()
            assert landscape.is_feasible(batch).all()
        # Without this the assertions above would hold vacuously on a campaign
        # that never moved: every round would be checked against the wild type,
        # which is the environment the sampler was built on anyway. The sampler
        # follows the anchor through its own `reanchored` hook rather than
        # through a factory, which is the path a real campaign takes now that
        # every baseline implements one.
        assert max(result.anchor_trace()) > 0


class TestReanchoringOnEhrlich:
    """The test that says the mechanism matters rather than merely runs."""

    def test_the_optimum_is_out_of_reach_of_one_round_and_inside_the_campaign(self):
        # The guard on the instance itself. Without this the comparison below
        # could pass on a landscape where re-anchoring was never needed.
        landscape, wild_type = small_ehrlich()
        _, within_one = ball_of(landscape, wild_type, 1)
        _, within_four = ball_of(landscape, wild_type, 4)
        assert within_one < within_four == 1.0

    def test_re_anchoring_reaches_fitness_the_fixed_ball_does_not_contain(self):
        # Not "better on this run" but *outside what the fixed anchor could ever
        # measure*: the ceiling is the best value in the one-mutation ball, and
        # the fixed campaign is capped at it however many rounds it runs. A
        # comparison of the two best values alone would also pass on a run that
        # merely got luckier inside the same ball.
        landscape, wild_type = small_ehrlich()
        _, ceiling = ball_of(landscape, wild_type, 1)

        def run(reanchor):
            # Twenty wells against a ball of twenty designs. The fixed arm spends
            # its budget exactly once over, which is the most a fixed anchor can
            # legally do: every plate is filled, and nothing is re-ordered.
            env, _ = ball_of(landscape, wild_type, 1)
            return Campaign(
                landscape=landscape,
                sampler=BallSampler(env),
                rounds=4,
                batch_size=5,
                pool_size=64,
                environment=env,
                reanchor=reanchor,
            ).run()

        fixed = run(reanchor=False)
        moving = run(reanchor=True)
        assert fixed.best_value <= ceiling
        assert moving.best_value > ceiling
        assert max(moving.anchor_trace()) > 1
        assert moving.oracle_calls == fixed.oracle_calls == 20


class TestReanchoringIsRefusedWhenItCannotBeDone:
    def test_an_infeasible_anchor_is_refused(self):
        # The landscape scores this design finite; the environment cannot build
        # it. Anchoring there would void feasibility-by-construction for every
        # later round, silently.
        transitions = np.ones((ALPHABET.size, ALPHABET.size))
        transitions[1, 1] = 0.0
        env = mutation_env(max_mutations=2, transitions=transitions)
        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=BallSampler(env),
            rounds=2,
            batch_size=4,
            pool_size=64,
            initial_design=np.array([[1, 1, 0, 0, 0, 0]], dtype=np.int32),
            environment=env,
            reanchor=True,
        )
        with pytest.raises(ValueError, match="infeasible design"):
            campaign.run()

    def test_re_anchoring_without_an_environment_is_refused(self):
        with pytest.raises(ValueError, match="needs the environment"):
            Campaign(
                landscape=CountingLandscape(),
                sampler=BallSampler(mutation_env()),
                reanchor=True,
            )

    def test_a_sampler_that_can_neither_be_told_nor_rebuilt_is_refused(self):
        # Refused at construction rather than after a round of oracle calls.
        with pytest.raises(ValueError, match="cannot follow a moved anchor"):
            Campaign(
                landscape=CountingLandscape(),
                sampler=RandomSampler(),
                environment=mutation_env(),
                reanchor=True,
            )

    def test_a_factory_is_enough_for_a_sampler_that_cannot_be_told(self):
        # RandomSampler deliberately has no `reanchored`, which is what makes
        # this the factory path. Every baseline in the package implements the
        # hook, so using one of those here would silently test the other branch.
        env = mutation_env(max_mutations=2)
        rebuilt = []

        def factory(moved):
            rebuilt.append(moved)
            return RandomSampler(seed=len(rebuilt))

        campaign = Campaign(
            landscape=CountingLandscape(),
            sampler=RandomSampler(seed=0),
            rounds=3,
            batch_size=8,
            pool_size=64,
            environment=env,
            reanchor=True,
            sampler_factory=factory,
        )
        result = campaign.run()
        assert max(result.anchor_trace()) > 0
        assert rebuilt, "the factory was never called, so nothing was rebuilt"
        assert campaign.sampler is not None
