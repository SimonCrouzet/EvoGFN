"""Tests for the preference-conditioned training loop and its balance objective.

The loop is a near-copy of `train_trajectory_balance` and the copy is deliberate:
that function takes one fixed reward and cannot be edited without invalidating
every stored single-objective record. What the copy adds is one thing -- omega is
redrawn every step, the policy is conditioned on it, and the reward is rebuilt at
it -- and that one thing is where every failure below lives.

**Training a policy conditioned on omega_1 against `R(.|omega_2)`.** The worst
bug available here. The loss converges, nothing raises, and the arm learns
nothing about the trade-off it is being asked for. It cannot be caught by
inspecting the loop's own bookkeeping, because a loop that drives both from one
variable and a loop that drives them from two look identical from the outside --
so it is caught by a spy inside the reward that reads the policy's live state.

**A fixed omega.** If the draw is not actually redrawn per step, the arm is
`gfn-tb` with a conditioning input that never varies, and the front collapses for
a reason nobody would look for.

**An unseeded draw.** The benchmark's comparisons are paired, and "same seed"
must mean "same omega schedule" or the pairing the statistics rest on is broken
in a way no assertion downstream would catch.

**A conditional log Z read off the wrong policy.** `ConditionalTrajectoryBalance`
needs a policy that has one; handed a plain `SequencePolicy` it must refuse
rather than fall back to the scalar, which is the failure mode the head exists to
remove.
"""

import numpy as np
import pytest
import torch

from evogfn.algorithms.gflownet.objectives import ContrastiveBalance, TrajectoryBalance
from evogfn.algorithms.gflownet.preference_conditioned import (
    ConditionalTrajectoryBalance,
    train_preference_conditioned,
)
from evogfn.algorithms.gflownet.sampling import sample_trajectories
from evogfn.algorithms.gflownet.training import TrainingConfig
from evogfn.core.types import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.base import FitnessLandscape
from evogfn.models.policy import SequencePolicy
from evogfn.models.preference_policy import PreferenceConditionedPolicy
from evogfn.rewards.base import TemperedReward
from evogfn.rewards.scalarization import Scalarization, ScalarizedReward, WeightedSum

ALPHABET = Alphabet.from_string("ACGT")
LENGTH = 5


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


def parts(seed=0):
    """A toy landscape, the environment over it, and a conditioned policy."""
    landscape = TwoObjectiveToy()
    env = MutationEnvironment(np.ones(LENGTH, dtype=np.int64), ALPHABET, max_mutations=3)
    policy = PreferenceConditionedPolicy(
        n_objectives=2,
        n_tokens=ALPHABET.size,
        sequence_length=LENGTH,
        n_actions=env.n_actions,
        embedding_dim=8,
        hidden_dim=16,
        seed=seed,
    )
    return landscape, env, policy


def reward():
    """The reward template the loop rebuilds at each step's omega."""
    return ScalarizedReward(WeightedSum(), [0.5, 0.5], reward=TemperedReward(beta=2.0))


class SpyScalarization(Scalarization):
    """Records the policy's live preference beside the one the reward was given.

    An independent observation rather than a readback of the loop's own
    bookkeeping: it sits inside the reward path and reads the *policy*, so a loop
    that conditioned on one omega and scalarised with another is visible here and
    nowhere else.
    """

    def __init__(self, inner, policy):
        self._inner = inner
        self._policy = policy
        self.seen = []

    def _combine(self, values, weights):
        self.seen.append((self._policy.preference, weights[0].copy()))
        return self._inner._combine(values, weights)


class TestOmegaReachesBothThePolicyAndTheReward:
    def test_the_reward_is_scalarised_at_the_preference_the_policy_is_holding(self):
        landscape, env, policy = parts()
        spy = SpyScalarization(WeightedSum(), policy)
        train_preference_conditioned(
            env,
            policy,
            landscape,
            ScalarizedReward(spy, [0.5, 0.5], reward=TemperedReward(beta=2.0)),
            TrainingConfig(steps=6, batch_size=4, seed=0),
        )
        assert len(spy.seen) == 6
        for conditioned_on, scalarised_with in spy.seen:
            np.testing.assert_allclose(conditioned_on, scalarised_with)

    def test_the_reward_actually_changes_with_the_preference(self):
        # The other half: a spy that agreed on a *constant* omega would pass the
        # test above while the arm did nothing. This pins that the preference the
        # reward saw genuinely moved across the run.
        landscape, env, policy = parts()
        spy = SpyScalarization(WeightedSum(), policy)
        train_preference_conditioned(
            env,
            policy,
            landscape,
            ScalarizedReward(spy, [0.5, 0.5], reward=TemperedReward(beta=2.0)),
            TrainingConfig(steps=8, batch_size=4, seed=0),
        )
        used = np.stack([weights for _, weights in spy.seen])
        assert len(np.unique(used[:, 0])) == used.shape[0]

    def test_the_template_preference_is_never_the_one_trained_at(self):
        # `ScalarizedReward` is immutable and the loop rebuilds it per step with
        # `with_preference`. A loop that forgot to would train every step at the
        # template's trade-off while the policy moved around it -- which is the
        # omega_1/omega_2 mismatch, arrived at by omission rather than by a swap.
        landscape, env, policy = parts()
        spy = SpyScalarization(WeightedSum(), policy)
        template = ScalarizedReward(spy, [0.25, 0.75], reward=TemperedReward(beta=2.0))
        train_preference_conditioned(
            env, policy, landscape, template, TrainingConfig(steps=5, batch_size=4, seed=0)
        )
        used = np.stack([weights for _, weights in spy.seen])
        assert not np.allclose(used, [0.25, 0.75])
        # And the template is unchanged, so it can be reused for the next round.
        np.testing.assert_allclose(template.preference, [0.25, 0.75])


class TestTheDrawItself:
    def test_a_distinct_preference_is_drawn_every_step(self):
        # A single omega reused for the whole run makes this arm `gfn-tb` with
        # extra machinery, and the only symptom is a front that covers one point.
        landscape, env, policy = parts()
        result = train_preference_conditioned(
            env, policy, landscape, reward(), TrainingConfig(steps=12, batch_size=4, seed=0)
        )
        drawn = np.stack(result.preferences)
        assert drawn.shape == (12, 2)
        assert len(np.unique(drawn[:, 0])) == 12

    def test_every_preference_lies_on_the_simplex(self):
        landscape, env, policy = parts()
        result = train_preference_conditioned(
            env, policy, landscape, reward(), TrainingConfig(steps=10, batch_size=4, seed=3)
        )
        drawn = np.stack(result.preferences)
        assert (drawn >= 0).all()
        np.testing.assert_allclose(drawn.sum(axis=1), 1.0)

    def test_one_seed_gives_one_schedule(self):
        # The pairing every comparison in this benchmark rests on. An unseeded
        # draw would make two runs at "the same seed" differ in the one axis the
        # arm is about, and the paired statistics would be comparing different
        # experiments while reporting them as the same.
        outcomes = []
        for _ in range(2):
            landscape, env, policy = parts()
            outcomes.append(
                np.stack(
                    train_preference_conditioned(
                        env,
                        policy,
                        landscape,
                        reward(),
                        TrainingConfig(steps=6, batch_size=4, seed=11),
                    ).preferences
                )
            )
        np.testing.assert_array_equal(outcomes[0], outcomes[1])

    def test_two_seeds_give_two_schedules(self):
        landscape, env, policy = parts()
        first = np.stack(
            train_preference_conditioned(
                env, policy, landscape, reward(), TrainingConfig(steps=6, batch_size=4, seed=0)
            ).preferences
        )
        landscape, env, policy = parts()
        second = np.stack(
            train_preference_conditioned(
                env, policy, landscape, reward(), TrainingConfig(steps=6, batch_size=4, seed=1)
            ).preferences
        )
        assert not np.allclose(first, second)

    def test_the_concentration_moves_where_the_capacity_goes(self):
        # `alpha` is the one knob on *which* trade-offs get trained. Below one it
        # puts mass at the corners, which is how the extremes of a front get
        # covered; a loop that ignored it would silently train a uniform sweep
        # whatever the caller asked for.
        landscape, env, policy = parts()
        corners = np.stack(
            train_preference_conditioned(
                env,
                policy,
                landscape,
                reward(),
                TrainingConfig(steps=40, batch_size=2, seed=0),
                alpha=0.05,
            ).preferences
        )
        landscape, env, policy = parts()
        middle = np.stack(
            train_preference_conditioned(
                env,
                policy,
                landscape,
                reward(),
                TrainingConfig(steps=40, batch_size=2, seed=0),
                alpha=20.0,
            ).preferences
        )
        assert np.abs(corners[:, 0] - 0.5).mean() > np.abs(middle[:, 0] - 0.5).mean()


class TestTheConditionalObjective:
    def test_it_measures_against_the_conditional_partition_function(self):
        landscape, env, policy = parts()
        generator = torch.Generator().manual_seed(0)
        trajectories = sample_trajectories(env, policy, 4, generator=generator)
        log_rewards = torch.as_tensor(
            reward().log_reward(landscape.evaluate(trajectories.terminal)), dtype=torch.float32
        )
        objective = ConditionalTrajectoryBalance()

        policy.set_preference([1.0, 0.0])
        first = float(objective.loss(trajectories, log_rewards, policy).detach())
        policy.set_preference([0.0, 1.0])
        second = float(objective.loss(trajectories, log_rewards, policy).detach())
        # Same trajectories, same rewards, different omega: the only thing that
        # can move is log Z(omega). If these agree the objective is reading a
        # constant and the arm is trajectory balance against a scalar.
        assert first != pytest.approx(second)

    def test_it_refuses_a_policy_with_no_conditional_partition_function(self):
        # Falling back to `policy.log_z` would be the exact failure the head
        # exists to remove, and it would run to completion.
        _, env, _ = parts()
        plain = SequencePolicy(
            n_tokens=ALPHABET.size,
            sequence_length=LENGTH,
            n_actions=env.n_actions,
            embedding_dim=8,
            hidden_dim=16,
            seed=0,
        )
        generator = torch.Generator().manual_seed(0)
        trajectories = sample_trajectories(env, plain, 4, generator=generator)
        with pytest.raises(TypeError, match="preference-conditioned policy"):
            ConditionalTrajectoryBalance().loss(trajectories, torch.zeros(4), plain)

    def test_it_does_not_advertise_the_vestigial_scalar(self):
        # `uses_log_z` decides only whether a trainer logs `policy.log_z`. For
        # this objective that scalar is the base class's untrained parameter, and
        # logging a constant zero under the name `log_z` is worse than logging
        # nothing at all.
        assert ConditionalTrajectoryBalance().uses_log_z is False
        assert ConditionalTrajectoryBalance().needs_state_rewards is False

    def test_the_head_moves_during_training(self):
        # A `Z_theta(omega)` that is initialised at random and never updated
        # would pass every test above while behaving exactly like the scalar it
        # replaced.
        landscape, env, policy = parts()
        policy.set_preference([0.3, 0.7])
        before = float(policy.conditional_log_z().detach())
        train_preference_conditioned(
            env, policy, landscape, reward(), TrainingConfig(steps=25, batch_size=8, seed=0)
        )
        policy.set_preference([0.3, 0.7])
        assert float(policy.conditional_log_z().detach()) != pytest.approx(before)


class TestWhichObjectivesTheLoopWillRun:
    def test_the_default_is_the_conditional_balance(self):
        landscape, env, policy = parts()
        result = train_preference_conditioned(
            env, policy, landscape, reward(), TrainingConfig(steps=3, batch_size=4, seed=0)
        )
        assert result.objective == "ConditionalTrajectoryBalance"

    def test_contrastive_balance_is_a_legal_alternative(self):
        # It is legal *because* omega is drawn once per batch: what cancels in a
        # contrasted pair is -log Z(omega), and it only cancels when both members
        # share their omega. Under a per-trajectory draw this would be wrong and
        # would still run.
        landscape, env, policy = parts()
        result = train_preference_conditioned(
            env,
            policy,
            landscape,
            reward(),
            TrainingConfig(steps=4, batch_size=8, seed=0),
            objective=ContrastiveBalance(),
        )
        assert len(result.losses) == 4
        assert all(np.isfinite(result.losses))

    def test_the_unconditional_balance_is_refused(self):
        # It reads `policy.log_z`, which this policy never trains, so the run
        # would complete against a constant partition function and produce a
        # front collapsed toward whichever trade-off that constant suited.
        landscape, env, policy = parts()
        with pytest.raises(ValueError, match="scalar log Z"):
            train_preference_conditioned(
                env,
                policy,
                landscape,
                reward(),
                TrainingConfig(steps=2, batch_size=4, seed=0),
                objective=TrajectoryBalance(),
            )

    def test_an_objective_needing_state_rewards_is_refused_at_the_start(self):
        # The detailed-balance family scores every visited state, which this loop
        # does not do. Refused before the first gradient step rather than after a
        # round of compute, and recorded as a known gap rather than as support.
        landscape, env, policy = parts()

        class NeedsStates(ConditionalTrajectoryBalance):
            @property
            def needs_state_rewards(self):
                return True

        with pytest.raises(ValueError, match="does not score intermediate states"):
            train_preference_conditioned(
                env,
                policy,
                landscape,
                reward(),
                TrainingConfig(steps=2, batch_size=4, seed=0),
                objective=NeedsStates(),
            )


def test_the_loop_charges_its_proxy_evaluations():
    # The compute column the amortisation claim is read against. One evaluation
    # per sampled trajectory per step, and independent of how many preferences
    # the arm will later be *queried* at -- which is the whole point.
    landscape, env, policy = parts()
    result = train_preference_conditioned(
        env, policy, landscape, reward(), TrainingConfig(steps=7, batch_size=6, seed=0)
    )
    assert result.oracle_calls == 42


def test_a_single_objective_reward_is_refused():
    # `ScalarizedReward` holds the preference the loop is about to replace, so a
    # plain `TemperedReward` here would silently ignore every omega drawn.
    landscape, env, policy = parts()
    with pytest.raises(TypeError, match="ScalarizedReward"):
        train_preference_conditioned(
            env,
            policy,
            landscape,
            TemperedReward(beta=2.0),  # type: ignore[arg-type]
            TrainingConfig(steps=2, batch_size=4, seed=0),
        )
