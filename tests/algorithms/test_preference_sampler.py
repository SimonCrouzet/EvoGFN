"""Tests for the preference-conditioned sampler: where omega enters, and what it costs.

This is the object that makes MOGFN-PC runnable without editing `loop/campaign.py`.
It fits its own multi-output surrogate at `observe` -- which the campaign calls
with the landscape's raw `(n, k)` matrix, before any scalarisation -- and it
spends its proposal across the evaluation grid rather than collapsing it back onto
one trade-off.

Five failures are under test, and the first two would each turn the experiment
into a measurement of something else while every column looked healthy.

**A plate collapsed onto one preference.** If the pool is preference-diverse and
the campaign then takes a prefix that is all one slice, the arm's whole advantage
is destroyed at the last step. Nothing raises and the indicators simply come back
flat, which reads as "conditioning does not help".

**Conditioning that is inert at inference.** "One trained model covers the whole
front" means a new trade-off costs a forward pass. If querying at two omegas
returns the same designs, that sentence is false and the arm is an expensive
`gfn-tb`.

**An implementation that has quietly become an ensemble.** One policy and one
surrogate whatever the grid size, or the arm beats its own ablation for exactly
the reason the ablation exists to rule out.

**Accounting reset by a moved anchor.** `proxy_calls` sits next to the oracle
budget in the results table, so an undercount lands there as a cost this method
did not pay -- and the amortisation claim is a compute claim as much as a
coverage one.

**Observations dropped by a moved anchor.** Unlike `GFlowNetSampler`, this
sampler holds measurements: they are what its surrogate is fitted to. Restarting
them at each anchor would refit the model on one round's worth of data while the
ledger recorded four.
"""

import numpy as np
import pytest
import torch

from evogfn.algorithms.gflownet.objectives import TrajectoryBalance
from evogfn.algorithms.gflownet.preference_sampler import PreferenceConditionedSampler
from evogfn.algorithms.gflownet.training import TrainingConfig
from evogfn.core.types import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.base import FitnessLandscape
from evogfn.models.preference_policy import PreferenceConditionedPolicy
from evogfn.rewards.base import TemperedReward
from evogfn.rewards.scalarization import ScalarizedReward, WeightedSum
from evogfn.surrogate.multi_output import MultiObjectiveProxy, MultiOutputEnsemble

ALPHABET = Alphabet.from_string("ACGT")
LENGTH = 6


class TwoObjectiveToy(FitnessLandscape):
    """Two objectives in direct conflict: count of A against count of T."""

    @property
    def alphabet(self):
        return ALPHABET

    @property
    def sequence_length(self):
        return LENGTH

    @property
    def n_objectives(self):
        return 2

    def _evaluate(self, sequences):
        array = np.asarray(sequences)
        return np.column_stack(
            [(array == 0).sum(axis=1) + 1.0, (array == 3).sum(axis=1) + 1.0]
        ).astype(np.float64)


def grid(count):
    """An evaluation grid of `count` trade-offs on the two-objective simplex."""
    first = np.linspace(0.0, 1.0, count)
    return np.column_stack([first, 1.0 - first])


def build(*, preferences=4, steps=4, seed=0, env=None):
    """A sampler with its own policy, surrogate and proxy, none of them shared."""
    landscape = TwoObjectiveToy()
    environment = env or MutationEnvironment(
        np.ones(LENGTH, dtype=np.int64), ALPHABET, max_mutations=3
    )
    policy = PreferenceConditionedPolicy(
        n_objectives=2,
        n_tokens=ALPHABET.size,
        sequence_length=LENGTH,
        n_actions=environment.n_actions,
        embedding_dim=8,
        hidden_dim=16,
        seed=seed,
    )
    surrogate = MultiOutputEnsemble(
        n_tokens=ALPHABET.size,
        sequence_length=LENGTH,
        n_objectives=2,
        n_members=2,
        hidden_dim=8,
        epochs=15,
        seed=seed,
    )
    proxy = MultiObjectiveProxy(surrogate, alphabet=ALPHABET, sequence_length=LENGTH)
    sampler = PreferenceConditionedSampler(
        environment,
        policy,
        surrogate=surrogate,
        proxy=proxy,
        preferences=grid(preferences),
        reward=ScalarizedReward(WeightedSum(), [0.5, 0.5], reward=TemperedReward(beta=2.0)),
        config=TrainingConfig(steps=steps, batch_size=8, seed=seed),
        seed=seed,
    )
    return landscape, environment, sampler


def measured(landscape, sampler, n=16):
    """One round of propose-and-observe, so the surrogate has something to fit."""
    batch = sampler.propose(n)
    sampler.observe(batch, landscape.evaluate(batch))
    return batch


class TestThePlateIsSpentAcrossPreferences:
    @pytest.mark.parametrize("count", [2, 4, 8])
    def test_every_prefix_is_balanced_across_the_grid(self, count):
        # The property the whole evaluation half turns on. The campaign takes a
        # *prefix* of the pool, so an ordering that put all of one preference
        # first would give the plate to one trade-off however diverse the pool
        # was -- and the pool statistics would still look right.
        landscape, _, sampler = build(preferences=count)
        measured(landscape, sampler, n=count * 3)
        pool = sampler.propose(count * 4)
        origin = sampler.last_preference_index
        assert origin is not None
        assert len(origin) == pool.shape[0]
        for prefix in range(count, pool.shape[0] + 1):
            counts = np.bincount(origin[:prefix], minlength=count)
            assert counts.min() >= prefix // count
            assert counts.max() <= -(-prefix // count)

    def test_every_preference_contributes_something(self):
        landscape, _, sampler = build(preferences=4)
        measured(landscape, sampler, n=12)
        sampler.propose(12)
        assert set(np.unique(sampler.last_preference_index)) == {0, 1, 2, 3}

    def test_a_pool_smaller_than_the_grid_still_returns_what_was_asked_for(self):
        # The campaign asks for `pool_size` and expects `pool_size`; a slice
        # arithmetic that rounded down would leave the plate short and the
        # campaign would spend proposal calls papering over it.
        landscape, _, sampler = build(preferences=8)
        measured(landscape, sampler, n=8)
        assert sampler.propose(3).shape == (3, LENGTH)

    def test_the_proposal_count_is_what_was_returned(self):
        landscape, _, sampler = build(preferences=4)
        measured(landscape, sampler, n=8)
        before = sampler.proposals_made
        sampler.propose(10)
        assert sampler.proposals_made - before == 10


class TestQueryingAtANewTradeOffCostsAForwardPass:
    def test_the_same_preference_and_the_same_stream_give_the_same_designs(self):
        # Establishes that the draw is controlled, which is what makes the test
        # below attributable to omega rather than to the random stream.
        landscape, _, sampler = build()
        measured(landscape, sampler)
        first = sampler.designs_at([0.9, 0.1], 8, generator=torch.Generator().manual_seed(5))
        second = sampler.designs_at([0.9, 0.1], 8, generator=torch.Generator().manual_seed(5))
        np.testing.assert_array_equal(first, second)

    def test_two_preferences_on_one_stream_give_different_designs(self):
        # The amortisation claim, at its smallest testable size: one model, two
        # trade-offs, no retraining. If these agree the conditioning is inert and
        # every front this arm reports is one point wearing eight labels.
        landscape, _, sampler = build()
        measured(landscape, sampler)
        first = sampler.designs_at([1.0, 0.0], 8, generator=torch.Generator().manual_seed(5))
        second = sampler.designs_at([0.0, 1.0], 8, generator=torch.Generator().manual_seed(5))
        assert not np.array_equal(first, second)

    def test_a_queried_slice_is_ranked_under_its_own_preference(self):
        # The second half of "spend the plate across preferences": each slice is
        # screened by the trade-off that produced it. Screening all of them under
        # one preference is the collapse this arm is built to avoid, and it would
        # leave every prefix balanced while making the balance meaningless.
        landscape, _, sampler = build()
        measured(landscape, sampler, n=24)
        designs = sampler.designs_at([1.0, 0.0], 12)
        mean, _ = sampler.surrogate.predict(designs)
        # omega = (1, 0) under a weighted sum is objective 0 alone, so the slice
        # must come back sorted by the model's prediction for that objective.
        assert np.all(np.diff(mean[:, 0]) <= 1e-9)

    def test_an_unfitted_sampler_returns_designs_rather_than_refusing(self):
        # Round zero has no model. Refusing here would mean the campaign could
        # not open, and ranking against an unfitted ensemble would order the
        # plate by an initialisation.
        _, _, sampler = build()
        assert sampler.designs_at([0.5, 0.5], 5).shape == (5, LENGTH)

    def test_a_preference_off_the_simplex_is_refused(self):
        _, _, sampler = build()
        with pytest.raises(ValueError, match="sum to 1"):
            sampler.designs_at([0.9, 0.9], 4)


class TestOneModelRatherThanN:
    @pytest.mark.parametrize("count", [1, 2, 8])
    def test_the_compute_does_not_scale_with_the_number_of_preferences(self, count):
        # The arm's claim is that one policy serves N trade-offs for the price of
        # one. If `proxy_calls` grew with the grid it would have quietly become a
        # `PreferenceEnsemble`, and it would beat its own ablation for exactly the
        # reason the ablation exists to rule out.
        landscape, _, sampler = build(preferences=count, steps=5)
        measured(landscape, sampler, n=8)
        sampler.propose(8)
        assert sampler.proxy_calls == 5 * 8
        assert sampler.rounds_trained == 1

    def test_there_is_exactly_one_policy_and_one_surrogate(self):
        landscape, _, sampler = build(preferences=8)
        measured(landscape, sampler, n=8)
        sampler.propose(16)
        assert isinstance(sampler.policy, PreferenceConditionedPolicy)
        assert isinstance(sampler.surrogate, MultiOutputEnsemble)

    def test_nothing_is_trained_before_there_is_a_model_to_train_against(self):
        # Round zero: the proxy is not ready, so the policy samples from its
        # initialisation. Training against an unfitted ensemble would optimise
        # toward its initialisation and spend the compute doing it.
        _, _, sampler = build()
        sampler.propose(8)
        assert sampler.proxy_calls == 0
        assert sampler.rounds_trained == 0


class TestWhatItLearnsFrom:
    def test_it_is_fitted_to_the_objective_vectors_rather_than_a_scalarisation(self):
        # The seam the whole arm hangs on: `Campaign` calls `observe` with the
        # landscape's raw (n, k), *before* `reduce_objectives`. A sampler fitted
        # to the reduced column would be back inside F3's reduction and the
        # preference would have nowhere left to enter.
        landscape, _, sampler = build()
        measured(landscape, sampler, n=16)
        assert sampler.surrogate.is_fitted
        mean, _ = sampler.surrogate.predict(sampler.observed_sequences)
        assert mean.shape == (16, 2)

    def test_it_accumulates_rather_than_refitting_on_the_last_round_alone(self):
        # The campaign hands `observe` one round at a time, while
        # `Surrogate.fit` is specified as taking the full accumulated dataset. A
        # sampler that forwarded the batch straight through would throw away
        # three quarters of a four-round campaign's data.
        landscape, _, sampler = build()
        measured(landscape, sampler, n=8)
        measured(landscape, sampler, n=8)
        assert sampler.observed_sequences.shape == (16, LENGTH)
        assert sampler.observed_values.shape == (16, 2)

    def test_a_scalarised_batch_is_refused(self):
        # An (n, 1) arriving here means somebody scalarised upstream, and the
        # multi-output ensemble would fit one objective and leave the other at
        # its initialisation.
        landscape, _, sampler = build()
        batch = sampler.propose(4)
        values = landscape.evaluate(batch)
        with pytest.raises(ValueError, match="2 objectives"):
            sampler.observe(batch, values[:, :1])

    def test_a_round_with_nothing_finite_leaves_the_model_unfitted(self):
        # A masked landscape can return a whole round of -inf. That is a result
        # about the method rather than an error, and it must not raise out of the
        # middle of a campaign that has already spent a plate.
        _, _, sampler = build()
        batch = sampler.propose(4)
        sampler.observe(batch, np.full((4, 2), -np.inf))
        assert not sampler.surrogate.is_fitted


class TestFollowingAMovedAnchor:
    def _moved(self, sampler, env):
        return sampler.reanchored(env.reanchored(np.zeros(LENGTH, dtype=np.int64)))

    def test_the_policy_and_the_surrogate_are_shared_rather_than_rebuilt(self):
        # Rebuilding would discard a trained policy and a fitted model at every
        # round boundary, which is the campaign's forgetful fallback path and the
        # reason this hook exists at all.
        landscape, env, sampler = build()
        measured(landscape, sampler)
        moved = self._moved(sampler, env)
        assert moved.policy is sampler.policy
        assert moved.surrogate is sampler.surrogate

    def test_the_accounting_continues(self):
        # `proxy_calls` is a campaign total printed beside the oracle budget.
        # Restarted at each anchor it would report the last anchor's rounds as
        # the arm's whole compute -- an undercount, in the direction that
        # flatters the method under test.
        landscape, env, sampler = build()
        measured(landscape, sampler)
        sampler.propose(8)
        moved = self._moved(sampler, env)
        assert moved.proxy_calls == sampler.proxy_calls
        assert moved.rounds_trained == sampler.rounds_trained
        assert moved.proposals_made == sampler.proposals_made

    def test_the_measurements_come_across(self):
        landscape, env, sampler = build()
        measured(landscape, sampler, n=8)
        moved = self._moved(sampler, env)
        assert moved.observed_sequences.shape == (8, LENGTH)
        np.testing.assert_array_equal(moved.observed_values, sampler.observed_values)

    def test_the_evaluation_grid_comes_across(self):
        # Two arms queried at different trade-offs are not the same arm, and a
        # grid regenerated at each move would make the second half of a campaign
        # answer a different question from the first.
        _, env, sampler = build(preferences=4)
        moved = self._moved(sampler, env)
        np.testing.assert_array_equal(moved.preferences, sampler.preferences)

    def test_a_move_that_changes_the_action_space_is_refused(self):
        # The policy's heads are sized to the action space; silently mis-indexed
        # against a new one it would propose designs nobody chose, and nothing
        # downstream would raise.
        _, _, sampler = build()
        other = MutationEnvironment(np.ones(LENGTH + 1, dtype=np.int64), ALPHABET, max_mutations=3)
        with pytest.raises(ValueError, match="sequence length"):
            sampler.reanchored(other)


class TestWhatItRefuses:
    def test_a_grid_that_does_not_match_the_policy_is_refused(self):
        # A four-point grid over two objectives handed to a three-objective
        # policy: the encoder would refuse it at the first `set_preference`,
        # which is inside the first proposal call and a round of construction
        # too late to be a configuration error.
        landscape, env, _ = build()
        policy = PreferenceConditionedPolicy(
            n_objectives=3,
            n_tokens=ALPHABET.size,
            sequence_length=LENGTH,
            n_actions=env.n_actions,
            seed=0,
        )
        surrogate = MultiOutputEnsemble(
            n_tokens=ALPHABET.size, sequence_length=LENGTH, n_objectives=3, n_members=2, epochs=5
        )
        with pytest.raises(ValueError, match="conditioned on 3"):
            PreferenceConditionedSampler(
                env,
                policy,
                surrogate=surrogate,
                proxy=MultiObjectiveProxy(
                    surrogate, alphabet=landscape.alphabet, sequence_length=LENGTH
                ),
                preferences=grid(4),
                reward=ScalarizedReward(WeightedSum(), [1 / 3, 1 / 3, 1 / 3]),
                config=TrainingConfig(steps=2, batch_size=4),
            )

    def test_an_objective_that_reads_the_scalar_log_z_is_refused_at_construction(self):
        # Refused where the arm is built rather than at the first gradient step,
        # which is a round of oracle calls later.
        landscape, env, _ = build()
        policy = PreferenceConditionedPolicy(
            n_objectives=2,
            n_tokens=ALPHABET.size,
            sequence_length=LENGTH,
            n_actions=env.n_actions,
            seed=0,
        )
        surrogate = MultiOutputEnsemble(
            n_tokens=ALPHABET.size, sequence_length=LENGTH, n_objectives=2, n_members=2, epochs=5
        )
        with pytest.raises(ValueError, match="scalar log Z"):
            PreferenceConditionedSampler(
                env,
                policy,
                surrogate=surrogate,
                proxy=MultiObjectiveProxy(
                    surrogate, alphabet=landscape.alphabet, sequence_length=LENGTH
                ),
                preferences=grid(2),
                reward=ScalarizedReward(WeightedSum(), [0.5, 0.5]),
                config=TrainingConfig(steps=2, batch_size=4),
                objective=TrajectoryBalance(),
            )


def test_the_name_says_what_was_trained_and_at_how_many_trade_offs():
    # Written into every ledger and every stored record. An arm whose name did
    # not carry the grid size would be indistinguishable from one queried at a
    # single preference, which is a different experiment.
    _, _, sampler = build(preferences=4)
    assert "MOGFN-PC" in sampler.name
    assert "4" in sampler.name
