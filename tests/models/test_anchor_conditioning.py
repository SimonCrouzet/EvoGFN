"""Tests for the policy that is told which parent it is evolving from.

`AnchorConditionedPolicy` is one added arm, so the two halves are separate
classes below.

**Off is unchanged.** `SequencePolicy` had its trunk input factored behind a
seam to make the subclass possible, and a refactor that moved a weight, changed
an initialisation or widened an input would make every existing GFlowNet result
incomparable to every new one while every test that only checks shapes still
passed. So the unconditioned policy is pinned against a fresh one built the same
way: same parameter count, same values, same output.

**On changes the conditioning and nothing else.** The claim is that the policy
is now a function *over anchors* -- that it can tell an inherited residue from a
substitution, and that moving the anchor moves the distribution without touching
a weight. Each of those is a separate test, because the interesting failure is a
policy that accepts an anchor and ignores it: nothing raises, the loss looks
fine, and the arm quietly measures the unconditioned method under another name.
"""

import numpy as np
import pytest
import torch

from evogfn.core import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.models import AnchorConditionedPolicy, SequencePolicy, to_tensor

SYMBOLS = "ABC"
LENGTH = 4


@pytest.fixture
def env():
    return MutationEnvironment(
        np.zeros(LENGTH, dtype=np.int32), Alphabet.from_string(SYMBOLS), max_mutations=3
    )


def build(cls, *, seed=0, **extra):
    return cls(
        n_tokens=len(SYMBOLS),
        sequence_length=LENGTH,
        n_actions=LENGTH * len(SYMBOLS) + 1,
        hidden_dim=32,
        embedding_dim=8,
        seed=seed,
        **extra,
    )


def forward_log_probs(policy, env, state):
    forward, _ = policy.log_probs(
        to_tensor(state.sequences),
        torch.as_tensor(env.forward_mask(state)),
        torch.as_tensor(env.backward_mask(state)),
    )
    return forward


class TestTheUnconditionedPolicyIsUnchanged:
    """The seam the subclass hangs on must not have moved anything.

    A widened input or a shifted initialisation here would make every result
    already in the store incomparable to every result after it, and no assertion
    about shapes would notice.
    """

    def test_the_trunk_still_reads_one_embedding_per_position(self):
        policy = build(SequencePolicy)
        first = policy.trunk[0]
        assert first.in_features == LENGTH * 8

    def test_two_policies_at_one_seed_are_still_identical(self):
        left = build(SequencePolicy, seed=7)
        right = build(SequencePolicy, seed=7)
        for a, b in zip(left.parameters(), right.parameters(), strict=True):
            assert torch.equal(a, b)

    def test_the_representation_is_still_the_flattened_embedding(self):
        # The refactor routes forward() through a helper; this pins that the
        # helper computes what the inlined expression used to.
        policy = build(SequencePolicy)
        sequences = to_tensor(np.array([[0, 1, 2, 0], [1, 1, 0, 2]]))
        expected = policy.trunk(policy.embedding(sequences).flatten(start_dim=1))
        assert torch.equal(policy(sequences), expected)

    def test_it_carries_no_anchor_at_all(self):
        # The mechanism must be opt-in: an unconditioned policy that acquired an
        # anchor buffer would change every existing arm's state dict.
        assert not hasattr(build(SequencePolicy), "anchor")


class TestTheAnchorReachesTheNetwork:
    """The failure guarded against is a policy that accepts an anchor and ignores it.

    Nothing raises when the conditioning is dropped on the floor: the shapes are
    right, the loss is finite, and the arm reports the unconditioned method under
    a name that claims otherwise.
    """

    def test_the_trunk_is_widened_to_hold_the_anchor_and_the_mutation_mask(self):
        policy = build(AnchorConditionedPolicy, anchor=np.zeros(LENGTH, dtype=np.int32))
        assert policy.trunk[0].in_features == LENGTH * (2 * 8 + 1)

    def test_the_same_state_under_two_anchors_gives_two_distributions(self, env):
        # The whole claim. If these agree, the anchor is an input the network
        # was handed and never read.
        policy = build(AnchorConditionedPolicy, anchor=env.parent)
        state = env.step(env.initial(1), np.array([1 * 3 + 1]))
        at_wild_type = forward_log_probs(policy, env, state)

        policy.set_anchor(np.array([2, 1, 0, 0], dtype=np.int32))
        assert not torch.allclose(at_wild_type, forward_log_probs(policy, env, state))

    def test_the_untouched_parent_is_not_confused_with_a_design_that_matches_it(self):
        # Conditioned, "this sequence is my anchor" and "this sequence is three
        # substitutions from somewhere else" are different states even when the
        # tokens are identical -- which is exactly what the unconditioned policy
        # cannot represent.
        policy = build(AnchorConditionedPolicy, anchor=np.zeros(LENGTH, dtype=np.int32))
        sequences = to_tensor(np.array([[0, 0, 0, 0]]))
        as_the_anchor = policy(sequences).clone()
        policy.set_anchor(np.array([1, 1, 1, 0], dtype=np.int32))
        assert not torch.allclose(as_the_anchor, policy(sequences))

    def test_moving_the_anchor_moves_no_weight(self):
        # What makes it a function over anchors rather than a policy for one:
        # the anchor is an input, so a move costs nothing that was learned.
        policy = build(AnchorConditionedPolicy, anchor=np.zeros(LENGTH, dtype=np.int32))
        before = [p.detach().clone() for p in policy.parameters()]
        policy.set_anchor(np.array([2, 2, 1, 0], dtype=np.int32))
        for was, now in zip(before, policy.parameters(), strict=True):
            assert torch.equal(was, now)

    def test_the_anchor_is_not_a_parameter_the_optimiser_would_train(self):
        policy = build(AnchorConditionedPolicy, anchor=np.zeros(LENGTH, dtype=np.int32))
        assert "_anchor" not in {name for name, _ in policy.named_parameters()}
        assert "_anchor" in policy.state_dict()

    def test_it_is_the_same_policy_everywhere_else(self, env):
        # One thing changes and nothing else: masking, normalisation and the
        # uniform backward policy are the properties trajectory balance rests
        # on, and the subclass touches none of them.
        policy = build(AnchorConditionedPolicy, anchor=env.parent)
        state = env.initial(3)
        mask = torch.as_tensor(env.forward_mask(state))
        forward, backward = policy.log_probs(
            to_tensor(state.sequences), mask, torch.as_tensor(env.backward_mask(state))
        )
        assert forward.shape == (3, env.n_actions)
        assert torch.isinf(forward[~mask]).all()
        assert torch.allclose(forward[mask].exp().sum(), torch.tensor(3.0), atol=1e-5)
        # The source state has no parents, so P_B is empty rather than nan.
        assert torch.isinf(backward).all()


class TestAnAnchorThatCannotBeMeantIsRefused:
    """A mis-shaped anchor conditions on a truncated parent or indexes out of range.

    Only one of those raises on its own, so both are refused up front rather
    than left to present as a wrong number in one case and a crash in the other.
    """

    def test_a_batch_is_not_an_anchor(self):
        with pytest.raises(ValueError, match="one sequence of length 4"):
            build(AnchorConditionedPolicy, anchor=np.zeros((2, LENGTH), dtype=np.int32))

    def test_a_different_length_is_refused(self):
        with pytest.raises(ValueError, match="one sequence of length 4"):
            build(AnchorConditionedPolicy, anchor=np.zeros(3, dtype=np.int32))

    def test_tokens_outside_the_alphabet_are_refused(self):
        policy = build(AnchorConditionedPolicy, anchor=np.zeros(LENGTH, dtype=np.int32))
        with pytest.raises(ValueError, match=r"lie in \[0, 3\)"):
            policy.set_anchor(np.array([0, 9, 0, 0]))

    def test_a_refused_anchor_leaves_the_old_one_in_place(self):
        # Half-applying it would leave the policy conditioning on nothing while
        # continuing to look conditioned.
        policy = build(AnchorConditionedPolicy, anchor=np.array([1, 0, 0, 0], dtype=np.int32))
        with pytest.raises(ValueError, match="one sequence of length 4"):
            policy.set_anchor(np.zeros(2, dtype=np.int32))
        assert policy.anchor.tolist() == [1, 0, 0, 0]
