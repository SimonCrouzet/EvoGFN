"""Tests for the preference-conditioned policy: does omega reach the network at all.

Four failures are under test and none of them raises, which is why they are
tested here rather than left to a campaign to reveal.

**A policy that widens its trunk and then ignores the preference.** This is the
failure `models/conditioning.py` was written to prevent -- the loss looks fine
and the only symptom is a Pareto front collapsing to a point, which is
indistinguishable by eye from "preference conditioning does not help at this
budget". That would be read as the experimental result.

**A trunk sized from one expression and fed from another.** The base class's
`_trunk_input_dim` docstring exists for exactly this, and the failure surfaces
as a shape error at the first forward pass -- after a policy has been built and
handed to a sampler -- rather than at construction.

**A `Z_theta(omega)` that is a constant in disguise.** A conditional partition
function that does not vary with omega silently reproduces the scalar `log Z`,
which biases the policy toward whichever trade-off the scalar happened to fit
while every reported number stays plausible.

**A one-objective simplex.** A preference over one objective is the constant
`[1.0]`, so the conditioning carries no information and the arm is `gfn-tb` with
extra machinery. Refused at construction rather than trained.
"""

import numpy as np
import pytest
import torch

from evogfn.models.conditioning import encoding_dim
from evogfn.models.policy import SequencePolicy
from evogfn.models.preference_policy import (
    DEFAULT_PREFERENCE_BINS,
    LOG_Z_HEAD_PREFIX,
    PreferenceConditionedPolicy,
    conditional_parameter_groups,
)

LENGTH = 6
TOKENS = 4
ACTIONS = LENGTH * TOKENS + 1


def build(*, n_objectives=2, n_bins=DEFAULT_PREFERENCE_BINS, seed=0):
    """A small preference-conditioned policy over a toy action space."""
    return PreferenceConditionedPolicy(
        n_objectives=n_objectives,
        n_bins=n_bins,
        n_tokens=TOKENS,
        sequence_length=LENGTH,
        n_actions=ACTIONS,
        embedding_dim=8,
        hidden_dim=16,
        seed=seed,
    )


def states(n=3):
    """A batch of states and the two masks a policy needs to score them."""
    sequences = torch.zeros((n, LENGTH), dtype=torch.long)
    forward = torch.ones((n, ACTIONS), dtype=torch.bool)
    backward = torch.zeros((n, ACTIONS), dtype=torch.bool)
    backward[:, 0] = True
    return sequences, forward, backward


class TestThePreferenceReachesTheNetwork:
    """The one property the whole arm rests on, and the quietest to lose."""

    def test_two_preferences_give_two_different_distributions(self):
        # The load-bearing test. Same weights, same state, different omega: if
        # these agree the conditioning is inert and every downstream number is a
        # measurement of `gfn-tb` reported under another name.
        policy = build()
        sequences, forward, backward = states()

        policy.set_preference([1.0, 0.0])
        first, _ = policy.log_probs(sequences, forward, backward)
        policy.set_preference([0.0, 1.0])
        second, _ = policy.log_probs(sequences, forward, backward)

        assert not torch.allclose(first, second)
        # And materially, not to within floating-point noise: a conditioning
        # signal a thousand times smaller than the state's would pass an
        # inequality test while doing nothing to the sampled distribution.
        assert float((first - second).detach().abs().max()) > 1e-3

    def test_nearby_preferences_give_nearer_distributions_than_distant_ones(self):
        # What the thermometer encoding buys over a one-hot binning, checked on
        # the policy rather than on the encoder: the conditioning is continuous
        # in omega, so a policy can interpolate to a trade-off it never trained
        # at. A one-hot encoding would make all three gaps roughly equal.
        policy = build()
        sequences, forward, backward = states()

        def distribution(preference):
            policy.set_preference(preference)
            forward_log_probs, _ = policy.log_probs(sequences, forward, backward)
            return forward_log_probs

        anchor = distribution([0.5, 0.5])
        near = distribution([0.55, 0.45])
        far = distribution([1.0, 0.0])
        assert float((anchor - near).detach().abs().max()) < float(
            (anchor - far).detach().abs().max()
        )

    def test_the_preference_survives_a_round_trip(self):
        policy = build(n_objectives=3)
        policy.set_preference([0.2, 0.3, 0.5])
        np.testing.assert_allclose(policy.preference, [0.2, 0.3, 0.5])

    def test_a_freshly_built_policy_is_at_the_neutral_trade_off(self):
        # The same preference `preference_vectors(k, 1)` returns, so a policy
        # queried before anyone set one is at the trade-off the benchmark calls
        # neutral rather than at whatever zeros encode to.
        np.testing.assert_allclose(build(n_objectives=4).preference, [0.25] * 4)

    def test_a_preference_off_the_simplex_is_refused(self):
        policy = build()
        with pytest.raises(ValueError, match="sum to 1"):
            policy.set_preference([0.9, 0.9])
        with pytest.raises(ValueError, match="non-negative"):
            policy.set_preference([1.5, -0.5])

    def test_a_preference_of_the_wrong_width_is_refused(self):
        # Refused rather than broadcast: a two-entry preference on a
        # three-objective policy would encode to a shorter vector than the trunk
        # was sized for, which is a shape error one forward pass later.
        policy = build(n_objectives=3)
        with pytest.raises(ValueError, match="3 objectives"):
            policy.set_preference([0.5, 0.5])


class TestTheTrunkIsSizedFromWhatItIsFed:
    """The failure the base class's seam docstring exists to prevent."""

    @pytest.mark.parametrize("n_objectives", [2, 3, 4])
    @pytest.mark.parametrize("n_bins", [2, 16])
    def test_the_declared_width_is_the_width_produced(self, n_objectives, n_bins):
        policy = build(n_objectives=n_objectives, n_bins=n_bins)
        sequences, _, _ = states()
        declared = policy._trunk_input_dim(sequence_length=LENGTH, embedding_dim=8)
        assert policy._trunk_input(sequences).shape[1] == declared
        assert declared == LENGTH * 8 + encoding_dim(n_objectives, n_bins=n_bins)

    def test_the_conditioning_is_concatenated_once_rather_than_per_position(self):
        # A preference is not a per-position quantity, unlike an anchor. Tiling
        # it would multiply the conditioning width by the sequence length and add
        # no information; the cost of not tiling is that omega is a smaller share
        # of the trunk input, which is what `n_bins` is the dial for.
        policy = build()
        width = policy._trunk_input_dim(sequence_length=LENGTH, embedding_dim=8)
        assert width - LENGTH * 8 == encoding_dim(2, n_bins=DEFAULT_PREFERENCE_BINS)

    def test_a_forward_pass_runs_at_every_objective_count(self):
        for n_objectives in (2, 3, 5):
            policy = build(n_objectives=n_objectives)
            sequences, forward, backward = states()
            forward_log_probs, backward_log_probs = policy.log_probs(sequences, forward, backward)
            assert forward_log_probs.shape == (3, ACTIONS)
            assert backward_log_probs.shape == (3, ACTIONS)


class TestTheConditionalPartitionFunction:
    """`Z_theta(omega)`, and the failure of it being a constant in disguise."""

    def test_it_varies_with_the_preference(self):
        # A conditional log Z that does not move with omega is the scalar log Z
        # wearing a head, and the residual then absorbs the variation in
        # log Z(omega) by pushing the policy toward one trade-off. The symptom is
        # a collapsed front, which is what the experiment is trying to measure.
        policy = build()
        policy.set_preference([1.0, 0.0])
        first = float(policy.conditional_log_z().detach())
        policy.set_preference([0.0, 1.0])
        second = float(policy.conditional_log_z().detach())
        assert first != pytest.approx(second)

    def test_it_carries_a_gradient_back_to_its_own_parameters(self):
        # Without this the head is a constant by optimisation rather than by
        # architecture, which fails in exactly the same direction and would pass
        # the test above.
        policy = build()
        policy.conditional_log_z().backward()
        head = [p for name, p in policy.named_parameters() if name.startswith(LOG_Z_HEAD_PREFIX)]
        assert head
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in head)

    def test_the_head_gets_the_elevated_learning_rate_and_the_trunk_does_not(self):
        # `objectives.parameter_groups` selects the 10x group by the *name*
        # `log_z`, so a head named anything else lands in the policy group at the
        # policy's rate. That is not an error anything raises: it is a log Z that
        # converges an order of magnitude too slowly and reads as a policy that
        # has not finished training.
        policy = build()
        groups = conditional_parameter_groups(policy, learning_rate=1e-3, log_z_multiplier=10.0)
        rates = {id(p): group["lr"] for group in groups for p in group["params"]}
        for name, parameter in policy.named_parameters():
            if name == "log_z":
                # The inherited scalar is in no group at all; see below.
                continue
            expected = 1e-2 if name.startswith(LOG_Z_HEAD_PREFIX) else 1e-3
            assert rates[id(parameter)] == pytest.approx(expected), name

    def test_the_inherited_scalar_log_z_is_optimised_by_nothing(self):
        # It exists only because the base class builds it, and it is never read
        # by this policy's objective. Leaving it in an optimiser group would let
        # it drift under weight decay or momentum and give a later reader a
        # plausible-looking number that trained against nothing.
        policy = build()
        groups = conditional_parameter_groups(policy, learning_rate=1e-3, log_z_multiplier=10.0)
        held = {id(p) for group in groups for p in group["params"]}
        assert id(policy.log_z) not in held
        assert float(policy.log_z.detach()) == 0.0

    def test_every_parameter_the_base_class_would_train_is_still_trained(self):
        # The other half of the check above: partitioning by name is easy to get
        # wrong in the direction of dropping a real parameter, which presents as
        # a layer that never learns.
        policy = build()
        groups = conditional_parameter_groups(policy, learning_rate=1e-3, log_z_multiplier=10.0)
        held = {id(p) for group in groups for p in group["params"]}
        expected = {id(p) for name, p in policy.named_parameters() if name != "log_z"}
        assert held == expected


class TestWhatItRefuses:
    def test_a_single_objective_policy_is_refused(self):
        # A simplex over one objective is the constant [1.0]: the conditioning
        # carries no information, `Z_theta(omega)` is a scalar with extra steps,
        # and the arm is `gfn-tb` reported under a name that claims more.
        with pytest.raises(ValueError, match="at least 2 objectives"):
            build(n_objectives=1)

    def test_one_bin_is_refused(self):
        # One bin encodes every preference identically, which is the inert
        # conditioning this whole file is about, arrived at by configuration.
        with pytest.raises(ValueError, match="n_bins"):
            build(n_bins=1)

    def test_it_is_still_a_sequence_policy(self):
        # The sampler, the objectives and the rollout all take a `SequencePolicy`
        # and would silently accept something that merely quacked like one; this
        # pins the inheritance the trunk-widening seam depends on.
        assert isinstance(build(), SequencePolicy)


def test_two_policies_at_one_seed_are_the_same_network():
    # The pairing every comparison in the benchmark rests on. The base class
    # seeds inside `fork_rng`, and a head built outside that block would draw
    # from global torch state -- so two arms at "the same seed" would differ in
    # their conditional log Z alone, which is exactly the kind of unexplained
    # run-to-run variance the seeding exists to remove.
    left, right = build(seed=7), build(seed=7)
    for (name, first), (_, second) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        assert torch.equal(first, second), name
    other = build(seed=8)
    assert not torch.equal(
        dict(left.named_parameters())[f"{LOG_Z_HEAD_PREFIX}.0.weight"],
        dict(other.named_parameters())[f"{LOG_Z_HEAD_PREFIX}.0.weight"],
    )
