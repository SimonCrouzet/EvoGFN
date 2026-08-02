"""Tests for the environment that constrains the design rather than the order.

`TerminalFeasibilityEnvironment` is one added arm, not a change of method, so
two things have to hold and each is a separate class below.

**Off is unchanged.** The flag that selects it is inert wherever there is no
transition matrix to defer, and the base class it extends behaves exactly as it
did. If that slips, an arm named for feasibility is measuring something else as
well and no column in the results would say which.

**On changes what it claims and keeps the graph legal.** Infeasible
intermediates become edges, feasible designs whose every construction order was
blocked become reachable, and stopping is what carries the constraint instead.
Moving edges is how a subclass silently breaks acyclicity, backward consistency
or mask honesty -- the loss still converges, to the wrong thing -- so all three
are re-checked against the new masks rather than inherited on trust.

The dead-end is the deliberate cost and is pinned like a feature, because that
is what it is: a trajectory that spends its last budget breaking an adjacency
has no legal move and no entitlement to stop, and the decision taken is to force
the stop and terminate on a design nobody can build.
"""

import itertools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from evogfn.core import Alphabet
from evogfn.env.base import State
from evogfn.env.mutation import MutationEnvironment, TerminalFeasibilityEnvironment


def transitions_forbidding(vocab, forbidden):
    matrix = np.ones((vocab, vocab), dtype=np.float64)
    for a, b in forbidden:
        matrix[a, b] = 0.0
    return matrix


def make_env(cls, *, length=4, symbols="ABC", max_mutations=None, transitions=None, **rest):
    return cls(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
        transitions=transitions,
        **rest,
    )


def as_set(sequences):
    return {tuple(int(t) for t in row) for row in sequences}


def every_state(env):
    """Every sequence in the ball, once running and once stopped."""
    sequences = env.enumerate_terminal_states()
    return State(
        sequences=np.concatenate([sequences, sequences]),
        stopped=np.concatenate(
            [np.zeros(len(sequences), dtype=np.bool_), np.ones(len(sequences), dtype=np.bool_)]
        ),
    )


def rollout(env, rng, n=1):
    """Walk random legal actions to termination."""
    state = env.initial(n)
    while not env.is_terminal(state).all():
        mask = env.forward_mask(state)
        actions = np.array(
            [rng.choice(np.flatnonzero(row)) if row.any() else env.stop_action for row in mask]
        )
        live = ~env.is_terminal(state)
        subset = State(sequences=state.sequences[live], stopped=state.stopped[live])
        stepped = env.step(subset, actions[live])
        sequences = state.sequences.copy()
        stopped = state.stopped.copy()
        sequences[live] = stepped.sequences
        stopped[live] = stepped.stopped
        state = State(sequences=sequences, stopped=stopped)
    return state


class TestTheFlagIsInertWithoutAConstraint:
    """With nothing forbidden the two classes must be the same graph.

    Otherwise the terminal-only arm differs from `gfn-tb` on an unconstrained
    landscape for reasons that have nothing to do with feasibility, and every
    such row is a comparison of two accidents.
    """

    def test_the_masks_agree_on_every_state(self):
        for length, symbols, budget in [(4, "ABC", 2), (3, "AB", 3), (5, "ABCD", 1)]:
            base = make_env(
                MutationEnvironment, length=length, symbols=symbols, max_mutations=budget
            )
            variant = make_env(
                TerminalFeasibilityEnvironment,
                length=length,
                symbols=symbols,
                max_mutations=budget,
            )
            states = every_state(base)
            assert np.array_equal(base.forward_mask(states), variant.forward_mask(states))
            assert np.array_equal(base.backward_mask(states), variant.backward_mask(states))

    def test_the_reachable_sets_agree(self):
        base = make_env(MutationEnvironment, max_mutations=2)
        variant = make_env(TerminalFeasibilityEnvironment, max_mutations=2)
        assert as_set(variant.reachable_terminal_states()) == as_set(
            base.reachable_terminal_states()
        )

    def test_they_agree_when_early_stopping_is_forbidden_too(self):
        base = make_env(MutationEnvironment, max_mutations=2, allow_stop_before_max=False)
        variant = make_env(
            TerminalFeasibilityEnvironment, max_mutations=2, allow_stop_before_max=False
        )
        assert np.array_equal(
            base.forward_mask(every_state(base)), variant.forward_mask(every_state(base))
        )


class TestTheConstraintMovesToTheDesign:
    """Feasibility of what gets built, not of the order it is written in.

    An intermediate is never synthesised, so masking it deletes designs for a
    reason with no physical referent. These pin the exchange that makes.
    """

    # Parent AAAA, with B -> A and A -> C forbidden. ABCA is feasible and both
    # of its intermediates -- ABAA and AACA -- are not, so the base class can
    # build it by no order at all.
    MATRIX = transitions_forbidding(3, [(1, 0), (0, 2)])
    TARGET = (0, 1, 2, 0)
    INFEASIBLE = (0, 1, 0, 0)

    def env(self, max_mutations=2):
        return make_env(
            TerminalFeasibilityEnvironment, max_mutations=max_mutations, transitions=self.MATRIX
        )

    def base(self, max_mutations=2):
        return make_env(MutationEnvironment, max_mutations=max_mutations, transitions=self.MATRIX)

    def test_a_substitution_through_a_forbidden_adjacency_is_offered(self):
        # The refusal the base class makes here is the thing under dispute, so
        # it has to actually be gone rather than merely argued about.
        base = self.base()
        assert not base.forward_mask(base.initial(1))[0, 1 * 3 + 1]
        env = self.env()
        assert env.forward_mask(env.initial(1))[0, 1 * 3 + 1]

    def test_a_design_with_no_feasible_construction_order_becomes_reachable(self):
        assert self.TARGET not in as_set(self.base().reachable_terminal_states())
        assert self.TARGET in as_set(self.env().reachable_terminal_states())

    def test_every_feasible_design_within_the_budget_is_reachable(self):
        # The claim in full: the reachable set stops being a strict subset of
        # the feasible ball, which is the gap the base class's tests pin.
        env = self.env()
        ball = env.enumerate_terminal_states()
        assert as_set(ball[env.is_reachable(ball)]) <= as_set(env.reachable_terminal_states())

    def test_stopping_is_refused_on_an_infeasible_sequence(self):
        # Where the deferred constraint lands. Budget remains, so the
        # trajectory must spend it rather than terminate on a broken design.
        env = self.env()
        state = State(
            sequences=np.array([self.INFEASIBLE], dtype=np.int32),
            stopped=np.zeros(1, dtype=np.bool_),
        )
        assert not env.forward_mask(state)[0, env.stop_action]

    def test_stopping_stays_available_on_a_feasible_sequence(self):
        env = self.env()
        assert env.forward_mask(env.initial(1))[0, env.stop_action]

    def test_a_dead_ended_trajectory_is_forced_to_stop_on_an_infeasible_design(self):
        # The cost of the design decision, pinned rather than left implicit.
        # Out of budget on a broken adjacency, with reversion not an edge of an
        # acyclic graph, nothing is legal and the stop is forced.
        env = self.env(max_mutations=1)
        state = env.step(env.initial(1), np.array([1 * 3 + 1]))
        assert state.sequences[0].tolist() == list(self.INFEASIBLE)

        mask = env.forward_mask(state)
        assert not mask[0, : env.n_mutation_actions].any(), "no budget should remain"
        assert mask[0, env.stop_action], "the trajectory must still have somewhere to go"
        assert not env.is_reachable(state.sequences)[0], "and lands on a design nobody can build"

    def test_the_support_reports_the_dead_ends_rather_than_hiding_them(self):
        # reachable_terminal_states is what a distributional comparison
        # normalises over, so it must describe what a trajectory can actually
        # end on -- including what it ends on by mistake.
        env = self.env(max_mutations=1)
        terminal = env.reachable_terminal_states()
        assert self.INFEASIBLE in as_set(terminal)
        assert not env.is_reachable(terminal).all()

    def test_is_reachable_still_refuses_an_infeasible_design(self):
        # Read by the replay buffer and the genetic teacher to decide what is
        # worth constructing a path to, and a dead end is worth none.
        env = self.env()
        assert not env.is_reachable(np.array([self.INFEASIBLE], dtype=np.int32))[0]
        assert env.is_reachable(np.array([self.TARGET], dtype=np.int32))[0]


class TestTheGraphStaysWellFormed:
    """Trajectory balance is valid only on a graph with these three properties.

    Deferring the constraint moves edges, and moving edges is how a subclass
    breaks acyclicity, backward consistency or mask honesty without anything
    raising.
    """

    @given(
        length=st.integers(min_value=2, max_value=5),
        budget=st.integers(min_value=1, max_value=3),
        forbidden=st.lists(
            st.tuples(st.integers(0, 2), st.integers(0, 2)), min_size=0, max_size=5, unique=True
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_every_unstopped_state_has_somewhere_to_go(self, length, budget, forbidden):
        # A state with no legal action leaves the policy nothing to normalise
        # over, and the masked softmax returns nan for the whole batch.
        env = make_env(
            TerminalFeasibilityEnvironment,
            length=length,
            max_mutations=min(budget, length),
            transitions=transitions_forbidding(3, forbidden),
        )
        states = every_state(env)
        assert env.forward_mask(states)[~states.stopped].any(axis=1).all()

    @given(
        length=st.integers(min_value=2, max_value=4),
        forbidden=st.lists(
            st.tuples(st.integers(0, 2), st.integers(0, 2)), min_size=0, max_size=5, unique=True
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_every_backward_edge_is_a_forward_edge_reversed(self, length, forbidden):
        # P_B must describe the graph P_F walks. Dropping the intermediate
        # constraint from one mask and not the other is how to get this wrong,
        # and the loss would still converge -- to a graph nobody samples.
        env = make_env(
            TerminalFeasibilityEnvironment,
            length=length,
            max_mutations=2,
            transitions=transitions_forbidding(3, forbidden),
        )
        states = every_state(env)
        for row, action in zip(*np.nonzero(env.backward_mask(states)), strict=True):
            if action == env.stop_action:
                continue
            single = State(
                sequences=states.sequences[row : row + 1], stopped=states.stopped[row : row + 1]
            )
            assert env.forward_mask(env.backward_step(single, np.array([action])))[0, action]

    def test_every_forward_action_still_increases_the_mutation_count(self):
        env = make_env(
            TerminalFeasibilityEnvironment,
            length=5,
            max_mutations=3,
            transitions=transitions_forbidding(3, [(0, 1)]),
        )
        rng = np.random.default_rng(0)
        state = env.initial(1)
        while not env.is_terminal(state).all():
            before = env.n_mutations(state)[0]
            action = rng.choice(np.flatnonzero(env.forward_mask(state)[0]))
            state = env.step(state, np.array([action]))
            if action != env.stop_action:
                assert env.n_mutations(state)[0] == before + 1

    def test_a_state_really_is_reached_by_k_factorial_orders(self):
        # log k! is exact here in a way it is not in the base class: no ordering
        # is refused, so the count of paths to a k-mutation state is every
        # permutation of its k substitutions.
        env = make_env(
            TerminalFeasibilityEnvironment,
            max_mutations=3,
            transitions=transitions_forbidding(3, [(0, 1)]),
        )
        substitutions = [(0, 1), (1, 2), (2, 1)]
        state = State(
            sequences=np.array([[1, 2, 1, 0]], dtype=np.int32), stopped=np.zeros(1, dtype=np.bool_)
        )
        assert env.log_n_trajectories(state)[0] == pytest.approx(math.log(math.factorial(3)))

        orders = 0
        for order in itertools.permutations(substitutions):
            walk = env.initial(1)
            for position, token in order:
                action = position * 3 + token
                if not env.forward_mask(walk)[0, action]:
                    break
                walk = env.step(walk, np.array([action]))
            else:
                orders += 1
        assert orders == math.factorial(3)

    def test_sampled_trajectories_only_terminate_in_the_reported_support(self):
        env = make_env(
            TerminalFeasibilityEnvironment,
            max_mutations=2,
            transitions=transitions_forbidding(3, [(1, 0), (0, 2)]),
        )
        support = as_set(env.reachable_terminal_states())
        for sequence in rollout(env, np.random.default_rng(0), n=64).sequences:
            assert tuple(int(t) for t in sequence) in support


class TestTheRuleSurvivesAMovedAnchor:
    """A campaign that re-anchors must not revert to the other method mid-run.

    `reanchored` builds the environment for the next round. Naming a class there
    rather than reading it off the instance would leave round one on the
    terminal-only graph and every later round on the intermediate-masked one,
    under a single arm name, with nothing in the record to say so.
    """

    def test_the_moved_environment_keeps_the_terminal_only_rule(self):
        env = make_env(
            TerminalFeasibilityEnvironment,
            max_mutations=2,
            transitions=transitions_forbidding(3, [(0, 2)]),
        )
        moved = env.reanchored(np.array([1, 0, 0, 0]))
        assert isinstance(moved, TerminalFeasibilityEnvironment)
        # A substitution creating the forbidden A -> C stays legal from here.
        assert moved.forward_mask(moved.initial(1))[0, 1 * 3 + 2]

    def test_the_base_class_still_moves_to_its_own_class(self):
        moved = make_env(MutationEnvironment, max_mutations=2).reanchored(np.array([1, 0, 0, 0]))
        assert type(moved) is MutationEnvironment
