"""Constructing variants by accumulating mutations on a parent sequence.

This is directed evolution as a construction graph. A trajectory starts at the
parent, applies point mutations one at a time, and stops. It is the formulation
used by MOGFN-AL (Jain et al., ICML 2023), and unlike autoregressive generation
it models *evolution from something* rather than design from nothing.

Structure of the graph
----------------------

Each position may be mutated **at most once**. That restriction is what makes the
graph acyclic: without it, mutating a site and then reverting it would return to
a state already visited, and the flow equations would have no solution.

With it, a state is exactly the parent plus a *set* of applied mutations, and the
graph is the **subset lattice** over mutated positions, graded by how many have
been applied. Two consequences follow, and both matter:

* A variant carrying ``k`` mutations is reachable by exactly ``k!`` trajectories,
  one per order in which the mutations could have been applied.
* Its parents in the graph are the ``k`` states reached by undoing any single
  one, so the uniform backward policy is exactly ``1/k`` per parent.

The second point makes the uniform backward policy cheap to compute exactly:
``1/k`` per parent, with no model and no learning. Note carefully what this does
*not* mean. ``P_B`` is not a quantity that can be got wrong in a way that biases
the result -- Malkin et al. show that "for any choice of backward policy
``P_B``, there is a unique flow ... and thus a unique corresponding forward
policy", so every valid ``P_B`` still yields a ``P_F`` sampling proportional to
reward. Choosing uniform is a matter of cost and variance, not correctness, and
they report a *learned* ``P_B`` converging faster on some tasks.

MOGFN-AL makes the same observation about this graph -- "``P_B`` here is not
trivial as there are multiple ways (orders) of generating the set" -- and also
settles on uniform.

Action encoding
---------------

Action ``a < length * alphabet_size`` sets position ``a // alphabet_size`` to
token ``a % alphabet_size``. The final index is the stop action. A single integer
per action lets a policy emit one logit vector per state and apply the mask to it
without any reshaping.

The anchor is per round, not per campaign
-----------------------------------------

One environment describes one Hamming ball: everything it can build lies within
``max_mutations`` of *its* parent. That is the right model for a single round and
the wrong model for a campaign. Real directed evolution re-anchors between
rounds -- round two starts from the best variant round one produced -- so
cumulative distance from the wild type grows round over round while the per-round
budget stays at four or five substitutions.

Anchoring a whole campaign to the wild type is therefore not a conservative
choice, it is a different experiment. On this repository's Ehrlich tasks the
planted optimum sits 61-248 mutations from the parent against a per-round budget
of four, so a fixed anchor makes the optimum unreachable *in principle* and any
regret measured against it is a regret no method could have closed.
[reanchored][evogfn.env.mutation.MutationEnvironment.reanchored] is what a
campaign calls between rounds to move the ball.

Where feasibility is enforced is a choice
-----------------------------------------

`MutationEnvironment` masks the transition constraint at every intermediate,
which is the conservative reading and the one every existing arm runs.
[TerminalFeasibilityEnvironment][evogfn.env.mutation.TerminalFeasibilityEnvironment]
is the other reading: an intermediate is a notepad entry rather than a molecule,
so the constraint belongs on the terminal. It is a subclass rather than a flag on
the base so that no arm can acquire it by default -- the two are different search
spaces, and a result produced under one is not comparable to a result produced
under the other.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from evogfn.env.base import SequenceEnvironment, State

if TYPE_CHECKING:
    from typing import Self

    import numpy.typing as npt

    from evogfn.core.types import Alphabet, Tokens

# How many outgoing edges the forward search expands per call. The frontier of a
# large graph has more edges than fit comfortably in one array, and the walk is
# breadth-first regardless of how the work is split.
_SEARCH_BATCH = 4096


class MutationEnvironment(SequenceEnvironment):
    """Builds variants by applying point mutations to a fixed parent.

    Args:
        parent: The starting sequence, shape ``(length,)``.
        alphabet: The alphabet ``parent`` is written in.
        max_mutations: Most mutations a trajectory may accumulate. Defaults to
            the sequence length, meaning unrestricted. Capping it restricts the
            search to a mutational neighbourhood, which is what a real
            directed-evolution round does.
        transitions: Optional ``(v, v)`` matrix whose zeros mark token pairs that
            may not be adjacent. When given, mutations producing a forbidden
            adjacency are masked out, so every sequence the environment can reach
            is feasible by construction rather than filtered afterwards.
        allow_stop_before_max: Whether a trajectory may stop early. When
            ``False``, trajectories run until ``max_mutations`` and every
            terminal state carries exactly that many mutations.

    Raises:
        ValueError: If the parent is not one-dimensional, contains tokens outside
            the alphabet, or the arguments disagree in shape.
    """

    def __init__(
        self,
        parent: Tokens,
        alphabet: Alphabet,
        *,
        max_mutations: int | None = None,
        transitions: npt.NDArray[np.floating] | None = None,
        allow_stop_before_max: bool = True,
    ) -> None:
        """Validate the parent and prepare the action layout."""
        parent_array = np.asarray(parent)
        if parent_array.ndim != 1:
            raise ValueError(f"parent must be a single sequence, got ndim {parent_array.ndim}")
        if not np.issubdtype(parent_array.dtype, np.integer):
            raise ValueError(f"parent must hold token indices, got dtype {parent_array.dtype}")
        if parent_array.size and (parent_array.min() < 0 or parent_array.max() >= alphabet.size):
            raise ValueError(
                f"parent tokens must lie in [0, {alphabet.size}), got "
                f"[{parent_array.min()}, {parent_array.max()}]"
            )

        self._parent = parent_array.astype(np.int32)
        self._alphabet = alphabet
        self._length = int(parent_array.shape[0])
        self._max_mutations = self._length if max_mutations is None else max_mutations
        if not 0 <= self._max_mutations <= self._length:
            raise ValueError(
                f"max_mutations must lie in [0, {self._length}], got {self._max_mutations}"
            )

        if transitions is not None:
            expected = (alphabet.size, alphabet.size)
            if transitions.shape != expected:
                raise ValueError(
                    f"transitions must be {expected} to match the alphabet, got {transitions.shape}"
                )
        self._transitions = transitions
        self._allow_stop_before_max = allow_stop_before_max
        # log(k!) for every reachable k. A table rather than a call to lgamma
        # because k is bounded by the sequence length and this is read on every
        # balance computation.
        self._log_factorial = np.concatenate(
            [[0.0], np.cumsum(np.log(np.arange(1, self._length + 1, dtype=np.float64)))]
        )

    @property
    def alphabet(self) -> Alphabet:
        """The alphabet sequences are written in."""
        return self._alphabet

    @property
    def sequence_length(self) -> int:
        """Length of the sequences this environment constructs."""
        return self._length

    @property
    def parent(self) -> Tokens:
        """The sequence every trajectory starts from."""
        return self._parent.copy()

    @property
    def max_mutations(self) -> int:
        """Most mutations a trajectory may accumulate."""
        return self._max_mutations

    @property
    def n_mutation_actions(self) -> int:
        """Number of substitution actions, excluding the stop action."""
        return self._length * self._alphabet.size

    @property
    def n_actions(self) -> int:
        """Size of the action space, including the stop action."""
        return self.n_mutation_actions + 1

    @property
    def stop_action(self) -> int:
        """Index of the action that terminates a trajectory."""
        return self.n_mutation_actions

    @property
    def constrains_intermediates(self) -> bool:
        """Whether the adjacency rule is enforced at every state, not just the end.

        The third of the decisions a variant of this graph may change, alongside
        `_substitution_mask` and `_may_stop`, and the one that cannot be read off
        either of them cheaply. It exists because a *caller* -- a projection, an
        enumeration, an audit -- has to know which set it is targeting before it
        starts, and probing the masks cannot tell it: a mask only ever reports
        the rows and columns the states it was handed happen to touch, so an
        environment that constrains nothing and one whose anchor is far from
        every forbidden pair look identical from the outside.

        Returns:
            ``True`` when a transition matrix is set, since this class masks it
            at every step. A subclass that defers the rule to the terminal must
            say so here as well as in its masks; leaving this ``True`` would let
            a caller narrow itself to a subset of what that subclass can build,
            and the loss would look like a search failure rather than a bug.
        """
        return self._transitions is not None

    def reanchored(self, parent: Tokens) -> Self:
        """Return the same environment anchored at a new parent.

        A campaign round searches within ``max_mutations`` of the anchor. Keeping
        the anchor at the wild type for every round caps the whole campaign at
        one round's budget from it, which is why a planted optimum tens of
        mutations away is unreachable no matter how many rounds are run. Moving
        the anchor to the round's best variant is the mechanism that makes
        cumulative distance grow while the per-round budget does not.

        **A new object rather than a moved anchor.** Samplers, policies and
        replay buffers hold a reference to the environment they were built
        against, and every mask this class produces is read relative to
        ``self._parent``: which positions count as mutated, how much budget is
        left, how many parents a state has. Moving the anchor in place would
        change all of that underneath a live trajectory -- a half-built state
        would acquire a different mutation count and a mask forbidding what it
        had already done -- and nothing would raise. The failure mode of a moved
        anchor is wrong numbers, not an exception, so the anchor is immutable and
        the caller has to decide, explicitly, what to hand the new environment
        to.

        Args:
            parent: The design to anchor at, shape ``(length,)``. Ordinarily the
                best variant measured so far.

        Returns:
            A new environment of *this* class over the same alphabet, mutation
            budget, transition matrix and stopping rule, anchored at ``parent``.
            This environment is unchanged. The class is read off ``self`` rather
            than named, so a subclass that changes the masking rule keeps that
            rule across a move; naming the base class here would silently revert
            a campaign to intermediate-masked search the first time its anchor
            moved, and every later round would be a different method than the
            one the arm asked for.

        Raises:
            ValueError: If ``parent`` is not a single sequence of this
                environment's length, holds tokens outside the alphabet, or --
                under a transition constraint -- is itself infeasible. An
                infeasible anchor is refused rather than accepted because every
                state it can reach inherits its forbidden adjacencies except by
                accident: the environment's promise that everything it builds is
                constructible would be void, silently, from the round it was
                anchored onward.
        """
        candidate = np.asarray(parent)
        if candidate.ndim != 1 or candidate.shape[0] != self._length:
            raise ValueError(
                f"a new anchor must be one sequence of length {self._length}, "
                f"got shape {candidate.shape}"
            )
        if self._transitions is not None and not self._is_feasible(candidate):
            raise ValueError(
                "refusing to anchor at an infeasible design: it violates the "
                "transition constraint, so the environment could no longer "
                "guarantee that what it builds is constructible"
            )
        return type(self)(
            candidate,
            self._alphabet,
            max_mutations=self._max_mutations,
            transitions=self._transitions,
            allow_stop_before_max=self._allow_stop_before_max,
        )

    def _is_feasible(self, sequence: Tokens) -> bool:
        """Whether every adjacent token pair in one sequence is permitted.

        Args:
            sequence: A single sequence of this environment's length.

        Returns:
            ``True`` when no transition constraint is set, or when the sequence
            satisfies it.
        """
        if self._transitions is None or self._length < 2:  # noqa: PLR2004 - a pair needs two
            return True
        permitted = self._transitions > 0
        return bool(np.all(permitted[sequence[:-1], sequence[1:]]))

    def initial(self, n: int) -> State:
        """Create ``n`` trajectories sitting at the parent.

        Args:
            n: Number of trajectories.

        Returns:
            A state of size ``n``, none of them stopped.
        """
        return State(
            sequences=np.tile(self._parent, (n, 1)),
            stopped=np.zeros(n, dtype=np.bool_),
        )

    def n_mutations(self, state: State) -> npt.NDArray[np.integer]:
        """How many positions differ from the parent, per trajectory.

        This is the grade of the state in the lattice, and doubles as the number
        of parents it has.

        Args:
            state: The current state.

        Returns:
            An ``(n,)`` integer array.
        """
        counts: npt.NDArray[np.integer] = (state.sequences != self._parent[None, :]).sum(axis=1)
        return counts

    def forward_mask(self, state: State) -> npt.NDArray[np.bool_]:
        """Which mutations, and whether stopping, are available.

        A substitution is available when the position is still unmutated, the
        token differs from the parent's (substituting the same token would be a
        no-op that never changes the state, so it cannot be an edge), and the
        result keeps every adjacency permitted.

        Args:
            state: The current state.

        Returns:
            An ``(n, n_actions)`` boolean array.
        """
        n = len(state)
        capacity_left = self.n_mutations(state) < self._max_mutations

        mask = np.zeros((n, self.n_actions), dtype=np.bool_)
        mask[:, : self.n_mutation_actions] = self._substitution_mask(state, capacity_left).reshape(
            n, -1
        )

        # Stopping is available once the trajectory is entitled to stop, and is
        # forced when nothing else is possible -- a state with no legal action
        # would leave the policy with nothing to normalise over.
        may_stop = self._may_stop(state, capacity_left)
        mask[:, self.stop_action] = may_stop | ~mask[:, : self.n_mutation_actions].any(axis=1)

        # A stopped trajectory takes no further actions at all.
        mask[state.stopped] = False
        return mask

    def _substitution_mask(
        self, state: State, capacity_left: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.bool_]:
        """Which substitutions are edges out of each state.

        Split out of `forward_mask` so that *which constraints apply to an
        intermediate* is one overridable decision rather than something a
        subclass has to restate the whole mask to change. The stop rule is split
        out for the same reason, and the two together are the only places a
        variant of this graph differs.

        Args:
            state: The current state.
            capacity_left: ``(n,)``, whether the trajectory may still mutate.

        Returns:
            An ``(n, length, v)`` boolean array: allowed where the position is
            still untouched, the token is a genuine change, the trajectory has
            capacity left, and the result keeps every adjacency permitted.
        """
        untouched = state.sequences == self._parent[None, :]
        differs = self._parent[None, :, None] != np.arange(self._alphabet.size)[None, None, :]
        allowed: npt.NDArray[np.bool_] = (
            untouched[:, :, None] & differs & capacity_left[:, None, None]
        )
        if self._transitions is not None:
            allowed &= self._adjacency_allowed(state.sequences, self._transitions)
        return allowed

    def _may_stop(
        self, state: State, capacity_left: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.bool_]:
        """Which trajectories are *entitled* to stop where they are.

        Entitlement is not the same as availability: `forward_mask` also forces
        stopping on a trajectory with no other legal action, so a rule here that
        forbids stopping everywhere still yields a well-formed mask.

        Args:
            state: The current state.
            capacity_left: ``(n,)``, whether the trajectory may still mutate.

        Returns:
            An ``(n,)`` boolean array.
        """
        if self._allow_stop_before_max:
            return np.ones(len(state), dtype=np.bool_)
        return ~capacity_left

    def backward_mask(self, state: State) -> npt.NDArray[np.bool_]:
        """Which actions could have produced this state.

        For an unstopped state carrying ``k`` mutations these are exactly the
        ``k`` substitutions that introduced them, so a uniform distribution over
        this mask is the exact backward policy the lattice induces.

        A *stopped* state is different, and getting it wrong is easy. Its only
        parent is the same state unstopped: the stop action is the sole edge
        into it. Also marking its mutations would give a terminal ``k + 1``
        parents instead of one, making the uniform backward policy ``1/(k+1)``
        where it should be ``1``, and admitting paths that undo a mutation while
        stopped -- which is not an edge of this graph.

        Under a transition constraint there is a further condition: undoing a
        mutation must land on a state that is itself feasible. A parent that
        violates an adjacency is not in the graph, so the edge into it does not
        exist either -- and a backward walk that ignored this would reconstruct
        paths the forward direction refuses.

        Args:
            state: The current state.

        Returns:
            An ``(n, n_actions)`` boolean array.
        """
        n = len(state)
        mask = np.zeros((n, self.n_actions), dtype=np.bool_)

        running = ~np.asarray(state.stopped, dtype=np.bool_)
        mutated = (state.sequences != self._parent[None, :]) & running[:, None]
        mutated &= self._revertible(state.sequences)
        rows, positions = np.nonzero(mutated)
        tokens = state.sequences[rows, positions]
        mask[rows, positions * self._alphabet.size + tokens] = True

        mask[:, self.stop_action] = state.stopped
        return mask

    def is_reachable(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Report which sequences clear the budget and the endpoint constraint.

        A sequence outside the mutation budget, or infeasible under the
        transition constraint, is not in the space the policy is defined over.
        Scoring one is meaningless rather than merely inaccurate, so callers
        that accept sequences from elsewhere -- a replay buffer, a genetic
        algorithm, an assay -- should check first.

        **This is a necessary condition and not a sufficient one.** Where
        intermediates are constrained it admits designs that are feasible where
        they stand and that no trajectory can build, because every ordering of
        their substitutions passes through a forbidden state. A caller deciding
        what belongs to the search space -- rather than merely filtering out what
        obviously does not -- wants
        [is_constructible][evogfn.env.mutation.MutationEnvironment.is_constructible],
        which adds the ordering condition and costs the same.

        Args:
            sequences: An ``(n, length)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array.
        """
        array = np.asarray(sequences)
        within_budget = (array != self._parent[None, :]).sum(axis=1) <= self._max_mutations
        if self._transitions is None:
            return np.asarray(within_budget, dtype=np.bool_)
        permitted = self._transitions > 0
        feasible = np.all(permitted[array[:, :-1], array[:, 1:]], axis=1)
        return np.asarray(within_budget & feasible, dtype=np.bool_)

    def is_constructible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Report which sequences some trajectory can actually build.

        The condition
        [is_reachable][evogfn.env.mutation.MutationEnvironment.is_reachable]
        checks -- inside the budget, feasible at the endpoint -- is necessary and
        **not sufficient**. Mutations are applied one at a time, so a design is
        constructible only when some ordering of its substitutions keeps every
        intermediate feasible too, and a design can satisfy the transition rule
        perfectly while every order of writing it down passes through a state the
        masks refuse. Anything that decides membership of this environment's
        search space by endpoint feasibility alone is targeting a strictly larger
        set than the one the policy is defined over, and a method allowed to
        leave the construction graph is not being compared to methods confined to
        it.

        Why a local test is exact
        -------------------------

        The obvious test -- search the orderings -- is factorial in the number of
        substitutions. It is not needed, because the constraint decomposes:

        1. Every intermediate is *fully* feasible, not merely feasible at the
           adjacency just written. The anchor is feasible, and a substitution
           disturbs only the two pairs it sits between, which is exactly what
           `_substitution_mask` checks. Feasibility of a whole state is therefore
           the conjunction over its adjacent pairs.
        2. An adjacent pair ``(i, i + 1)`` changes value only when one of its two
           positions is substituted, so the value combinations it passes through
           are decided entirely by which of the two comes first. Three of them
           are fixed regardless of the order: both at the anchor, which is
           feasible, and both at the destination, which is feasible by
           assumption. Only the mixed combination depends on the ordering.
        3. So each pair imposes at most one condition, and only when *both* its
           positions are substituted: the mixed state reached by doing the left
           one first, or the one reached by doing the right one first, must be
           permitted. A pair with an unsubstituted position imposes nothing --
           its mixed state is the destination or the anchor.
        4. Those conditions are orderings of adjacent positions, so the graph
           they constrain is a path. Any orientation of a path is acyclic, hence
           has a topological order, and no condition couples two different pairs.
           Satisfying every pair separately therefore yields a genuine global
           ordering, and each maximal run of consecutive substituted positions
           can be constructed to completion before the next one is begun.

        The same argument read the other way is the run decomposition: two
        maximal runs are separated by at least one position that never changes,
        so no adjacent pair straddles them and no run constrains another.

        Args:
            sequences: An ``(n, length)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array. Where the environment does not constrain
            intermediates this is `is_reachable` unchanged, because then every
            ordering is legal and the two sets coincide.

        Note:
            This answers whether a trajectory can *reach* a design, which is
            what a search space is. Under ``allow_stop_before_max`` it is also
            whether the design can be a terminal state; without it, a reachable
            design short of the budget is a state the policy passes through
            rather than one it can emit.
        """
        array = np.asarray(sequences)
        reachable = self.is_reachable(array)
        transitions = self._transitions
        if (
            transitions is None or not self.constrains_intermediates or self._length < 2  # noqa: PLR2004 - a pair needs two
        ):
            return reachable

        permitted = transitions > 0
        substituted = array != self._parent[None, :]
        # The only pairs that can constrain an ordering, per step 3 above.
        coupled = substituted[:, :-1] & substituted[:, 1:]
        left_first = permitted[array[:, :-1], self._parent[None, 1:]]
        right_first = permitted[self._parent[None, :-1], array[:, 1:]]
        orderable = ~coupled | left_first | right_first
        return np.asarray(reachable & orderable.all(axis=1), dtype=np.bool_)

    def _revertible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Which mutated positions have the reversion as a genuine parent state.

        The mirror of `_substitution_mask`: an edge exists backward exactly where
        it exists forward, so whatever a subclass permits during construction it
        must also permit undoing. Overriding one without the other gives ``P_B``
        a different graph from ``P_F``, and trajectory balance is then being
        solved for a graph nobody walks.

        Args:
            sequences: An ``(n, length)`` array of token indices.

        Returns:
            An ``(n, length)`` boolean array, ``True`` where reverting that
            position lands on a state this environment admits.
        """
        if self._transitions is None:
            return np.ones(sequences.shape, dtype=np.bool_)
        return self._parent_would_be_feasible(sequences)

    def _parent_would_be_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Which single-mutation reversions land on a feasible state.

        Returns:
            An ``(n, length)`` boolean array, ``True`` where reverting that
            position keeps every adjacency permitted.
        """
        if self._transitions is None:  # pragma: no cover - guarded by the caller
            return np.ones(sequences.shape, dtype=np.bool_)
        permitted = self._transitions > 0
        reverted = np.broadcast_to(self._parent[None, :], sequences.shape)

        allowed = np.ones(sequences.shape, dtype=np.bool_)
        if self._length > 1:
            # Reverting position p only disturbs the (p-1, p) and (p, p+1) pairs.
            allowed[:, 1:] &= permitted[sequences[:, :-1], reverted[:, 1:]]
            allowed[:, :-1] &= permitted[reverted[:, :-1], sequences[:, 1:]]
        return allowed

    def step(self, state: State, actions: npt.NDArray[np.integer]) -> State:
        """Apply one action per trajectory.

        Args:
            state: The current state.
            actions: ``(n,)`` action indices.

        Returns:
            The resulting state.

        Raises:
            ValueError: If any action is masked out. Silently ignoring an
                illegal action would let a policy place probability on edges
                that do not exist, and the flow equations would be solved for the
                wrong graph.
        """
        actions = np.asarray(actions)
        self._check_permitted(self.forward_mask(state), actions, "forward")

        sequences = state.sequences.copy()
        stopped = state.stopped.copy()

        stopping = actions == self.stop_action
        stopped[stopping] = True

        mutating = ~stopping
        if mutating.any():
            rows = np.nonzero(mutating)[0]
            positions = actions[mutating] // self._alphabet.size
            tokens = actions[mutating] % self._alphabet.size
            sequences[rows, positions] = tokens

        return State(sequences=sequences, stopped=stopped)

    def backward_step(self, state: State, actions: npt.NDArray[np.integer]) -> State:
        """Undo one action per trajectory.

        Args:
            state: The current state.
            actions: ``(n,)`` action indices to reverse.

        Returns:
            The preceding state.

        Raises:
            ValueError: If any action could not have produced this state.
        """
        actions = np.asarray(actions)
        self._check_permitted(self.backward_mask(state), actions, "backward")

        sequences = state.sequences.copy()
        stopped = state.stopped.copy()

        unstopping = actions == self.stop_action
        stopped[unstopping] = False

        reverting = ~unstopping
        if reverting.any():
            rows = np.nonzero(reverting)[0]
            positions = actions[reverting] // self._alphabet.size
            sequences[rows, positions] = self._parent[positions]

        return State(sequences=sequences, stopped=stopped)

    def enumerate_terminal_states(self) -> Tokens:
        """Every sequence within the mutation budget of the parent.

        This is the Hamming ball of radius ``max_mutations``. It is the set the
        environment can construct only when nothing constrains the edges -- no
        transition matrix, and early stopping allowed. Otherwise it overstates the
        support in two separate ways:

        * A transition matrix masks substitutions out, so the ball contains
          sequences no trajectory can build. Not only infeasible ones: also
          *feasible* sequences whose every construction order passes through an
          infeasible intermediate.
        * With ``allow_stop_before_max`` false, the ball contains partial
          constructions that are states of the graph but never terminal states.

        Either way, normalising a target distribution over the ball puts mass on
        sequences of probability zero, and the L1 that results measures the
        mis-specified support rather than the policy -- loudly enough to read as a
        broken sampler. Use
        [reachable_terminal_states][evogfn.env.mutation.MutationEnvironment.reachable_terminal_states]
        whenever a
        transition matrix is set or early stopping is forbidden; this method is
        the cheap closed-form answer for the unconstrained case, and an upper
        bound otherwise.

        Returns:
            An ``(m, length)`` array of distinct sequences, the parent first.

        Raises:
            ValueError: If the ball exceeds
                `MAX_ENUMERABLE_SIZE`.
        """
        from itertools import combinations, product  # noqa: PLC0415 - only needed here

        from evogfn.landscapes.base import MAX_ENUMERABLE_SIZE  # noqa: PLC0415

        alternatives = [
            [t for t in range(self._alphabet.size) if t != int(self._parent[position])]
            for position in range(self._length)
        ]
        size = self._hamming_ball_size()
        if size > MAX_ENUMERABLE_SIZE:
            raise ValueError(
                f"{size:,} sequences are reachable, above the "
                f"{MAX_ENUMERABLE_SIZE:,} enumeration limit"
            )

        reachable = [self._parent.copy()]
        for k in range(1, self._max_mutations + 1):
            for positions in combinations(range(self._length), k):
                for tokens in product(*(alternatives[p] for p in positions)):
                    variant = self._parent.copy()
                    variant[list(positions)] = tokens
                    reachable.append(variant)
        return np.stack(reachable)

    def reachable_terminal_states(self) -> Tokens:
        """Every terminal state a trajectory can actually end in.

        This is the support of the policy: the set an exact distributional
        comparison must be normalised over. It is found by walking the graph
        forward from the parent using this environment's own `forward_mask`
        and `step`, which makes it the reachable set by construction. A
        second implementation that re-derived reachability from ``transitions``
        could disagree with the masks, and then the measurement would be wrong in
        a way no test of the constraint alone would catch.

        Filtering `enumerate_terminal_states` through `is_reachable`
        is *not* equivalent, and that is the whole reason this method exists.
        Mutations are applied one at a time, so a variant carrying ``k`` of them
        is constructible only if some ordering exists along which all ``k``
        intermediates are feasible too. When every ordering passes through a
        forbidden adjacency, masking refuses that step in all of them and no path
        to the destination exists -- even though the destination itself is
        perfectly feasible and satisfies `is_reachable`. On a length-8
        Ehrlich toy with a transition matrix and a budget of two mutations, the
        Hamming ball holds 277 sequences, 26 of them feasible, and 18 reachable:
        eight feasible designs the policy can never emit.

        Filtering it through
        [is_constructible][evogfn.env.mutation.MutationEnvironment.is_constructible]
        *is* equivalent under ``allow_stop_before_max``, and that predicate is
        local and costs nothing, so anything asking whether a design it already
        holds is in the search space should ask there rather than enumerate. This
        method remains the definition, and the one that stays affordable on an
        instance too large to enumerate is the one that has to be checked against
        it.

        Terminality is read from the mask rather than from the mutation count, so
        the result also respects ``allow_stop_before_max``: where stopping early
        is forbidden, partial constructions are visited during the search but only
        states that may stop are returned.

        Returns:
            An ``(m, length)`` array of distinct terminal sequences in
            breadth-first order, so the parent comes first where it is terminal.

        Raises:
            ValueError: If the Hamming ball exceeds
                `MAX_ENUMERABLE_SIZE`. The search visits a subset of the ball,
                so refusing up front on that bound means the walk itself can
                never exhaust memory, at the cost of refusing some heavily
                constrained graphs that would in fact have been small.
        """
        from evogfn.landscapes.base import MAX_ENUMERABLE_SIZE  # noqa: PLC0415

        size = self._hamming_ball_size()
        if size > MAX_ENUMERABLE_SIZE:
            raise ValueError(
                f"the search would visit up to {size:,} sequences, above the "
                f"{MAX_ENUMERABLE_SIZE:,} enumeration limit"
            )

        terminal: list[Tokens] = []
        visited = {self._parent.tobytes()}
        frontier = self._parent[None, :].copy()
        while frontier.shape[0]:
            mask = self.forward_mask(
                State(sequences=frontier, stopped=np.zeros(frontier.shape[0], dtype=np.bool_))
            )
            terminal.extend(frontier[np.nonzero(mask[:, self.stop_action])[0]])

            # Every forward substitution raises the mutation count by one, so a
            # state first seen in this layer is never reached again from a later
            # one and dropping duplicates here loses no edges worth walking.
            rows, actions = np.nonzero(mask[:, : self.n_mutation_actions])
            discovered: list[Tokens] = []
            for start in range(0, rows.size, _SEARCH_BATCH):
                block = slice(start, start + _SEARCH_BATCH)
                sources = State(
                    sequences=frontier[rows[block]],
                    stopped=np.zeros(rows[block].size, dtype=np.bool_),
                )
                for child in self.step(sources, actions[block]).sequences:
                    key = child.tobytes()
                    if key not in visited:
                        visited.add(key)
                        discovered.append(child)
            frontier = (
                np.stack(discovered)
                if discovered
                else np.empty((0, self._length), dtype=self._parent.dtype)
            )

        return np.stack(terminal)

    def _hamming_ball_size(self) -> int:
        """How many sequences lie within the mutation budget of the parent.

        Returns:
            The size of the Hamming ball of radius ``max_mutations``, counted in
            closed form rather than by enumeration so it can gate enumeration.
        """
        return sum(
            math.comb(self._length, k) * (self._alphabet.size - 1) ** k
            for k in range(self._max_mutations + 1)
        )

    def log_n_trajectories(self, state: State) -> npt.NDArray[np.float64]:
        """Log of how many distinct paths reach each state from the parent.

        A state with ``k`` mutations is reached by ``k!`` orderings. Returned in
        log space because ``k!`` overflows quickly, and because every use of it
        is inside a log-domain balance computation anyway.

        Args:
            state: The current state.

        Returns:
            An ``(n,)`` array of ``log(k!)``.
        """
        counts: npt.NDArray[np.float64] = self._log_factorial[self.n_mutations(state)]
        return counts

    def _adjacency_allowed(
        self, sequences: Tokens, transitions: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.bool_]:
        """Which substitutions keep every adjacency permitted.

        Returns:
            An ``(n, length, v)`` boolean array.
        """
        permitted = transitions > 0
        n = sequences.shape[0]
        candidates = np.arange(self._alphabet.size)

        allowed = np.ones((n, self._length, self._alphabet.size), dtype=np.bool_)
        if self._length > 1:
            # Substituting at position p constrains the (p-1, p) and (p, p+1)
            # pairs only; every other adjacency is untouched.
            left = sequences[:, :-1]  # token before positions 1..L-1
            allowed[:, 1:, :] &= permitted[left[:, :, None], candidates[None, None, :]]
            right = sequences[:, 1:]  # token after positions 0..L-2
            allowed[:, :-1, :] &= permitted[candidates[None, None, :], right[:, :, None]]
        return allowed

    def _check_permitted(
        self,
        mask: npt.NDArray[np.bool_],
        actions: npt.NDArray[np.integer],
        direction: str,
    ) -> None:
        """Raise if any action is not permitted by ``mask``."""
        if actions.shape != (mask.shape[0],):
            raise ValueError(
                f"expected one action per trajectory, got {actions.shape} for "
                f"{mask.shape[0]} trajectories"
            )
        if actions.size == 0:
            return
        if actions.min() < 0 or actions.max() >= self.n_actions:
            raise ValueError(
                f"action indices must lie in [0, {self.n_actions}), got "
                f"[{actions.min()}, {actions.max()}]"
            )
        permitted = mask[np.arange(mask.shape[0]), actions]
        if not permitted.all():
            offenders = np.nonzero(~permitted)[0]
            raise ValueError(
                f"{direction} action not permitted for trajectories {offenders.tolist()}: "
                f"{actions[offenders].tolist()}"
            )


class TerminalFeasibilityEnvironment(MutationEnvironment):
    """The same lattice, with feasibility required of terminals and not of steps.

    Why the intermediate constraint is an artifact
    ----------------------------------------------

    `MutationEnvironment` enforces the transition constraint at *every* state a
    trajectory passes through, because that is what masking a construction graph
    does. The consequence is that the set it can build is not the set of feasible
    designs within the mutation budget: it is the smaller set of feasible designs
    that have *some ordering of their substitutions along which every prefix is
    also feasible*. A design can be perfectly feasible, sit well inside the
    budget, and still be unreachable because every order of writing its
    substitutions down passes through a forbidden adjacency.

    The claim this class makes is that the excluded designs are excluded for no
    reason a biologist would recognise. An intermediate here has no physical
    referent. It is never synthesised, never assayed, and never exists in a tube;
    it is the state of the policy's notepad after some of the substitutions have
    been written down. Feasibility is a constraint on a **sequence** -- on what
    gets built -- not on the order in which its differences from the parent are
    enumerated. Requiring it of intermediates is a property of masking-on-
    intermediates, not of the biology, and it silently deletes designs from the
    search space.

    So this environment masks substitutions on the mutation budget alone, leaves
    adjacency unenforced during construction, and applies the constraint where it
    means something: at the terminal.

    What that buys
    --------------

    Every feasible design within the budget becomes constructible, by any
    ordering at all. Two things that were approximations in the base class become
    exact as a result:
    [is_reachable][evogfn.env.mutation.MutationEnvironment.is_reachable] --
    within budget and feasible -- now describes the constructible set rather than
    over-stating it, and ``log k!`` really is the number of trajectories reaching
    a ``k``-mutation state, since no ordering is refused.

    How the stop action carries the constraint, and what it costs
    -------------------------------------------------------------

    The constraint has to be applied *somewhere*, and the only edge whose
    destination is a terminal is the stop action. So stopping is an entitlement
    of feasible states only: a trajectory sitting on an infeasible sequence must
    keep mutating.

    That is what creates the dead-end. Substitutions still consume budget, each
    position may still be mutated at most once, and reverting is not an edge of
    an acyclic graph -- so a trajectory can spend its last unit of budget on a
    substitution that breaks an adjacency and then have nothing legal left: no
    capacity to repair, and no entitlement to stop. **The decision taken here is
    to force the stop action in that case**, which is exactly the rule the base
    class already applies when a state has no other move, and the trajectory
    terminates on an infeasible design.

    The alternatives were all worse. Permitting a reverting move would make the
    graph cyclic and the flow equations unsolvable. Refusing any substitution
    whose result cannot still be repaired within the remaining budget is a
    reachability lookahead over the rest of the ball, which is the expensive
    computation the masking was supposed to avoid, and it re-imports the
    intermediate constraint in a weaker disguise. Raising would turn a legal walk
    of the graph into a crash.

    What the decision costs is real and should be reported, not hidden:

    * A share of trajectories terminate on infeasible designs. The oracle scores
      those ``-inf``, so each one occupies a well of the plate and returns no
      measurement -- a direct loss of oracle budget, in the currency the whole
      benchmark is indexed by.
    * Training does not push back on them. The policy is trained against a
      surrogate proxy, which has no notion of feasibility and will happily assign
      a dead-end design a high reward; the correction only arrives when the
      design reaches the assay.
    * The dead-end rate is a property of the transition matrix and the budget,
      not of the method, so it is a confound between arms whenever the two
      differ. It goes to zero as the alphabet's permitted adjacencies get denser
      and rises as the budget gets tighter, since a tight budget leaves no room
      to mutate out of a broken adjacency.

    `is_reachable` deliberately keeps refusing those designs. It is read by the
    replay buffer and the genetic teacher to decide what is worth constructing a
    path to, and an infeasible design is worth none: a dead-end is something a
    trajectory can stumble into, not something anything should aim at.
    [reachable_terminal_states][evogfn.env.mutation.MutationEnvironment.reachable_terminal_states]
    reports them, because it walks this environment's own masks and its contract
    is what a trajectory can *actually* end on -- which is the support a
    distributional comparison must normalise over.

    Nothing changes without a transition matrix. With ``transitions=None`` there
    is no adjacency to enforce or defer, so this class and its base describe the
    identical graph, and the flag that selects it is a no-op on an unconstrained
    landscape rather than a silent second effect.
    """

    @property
    def constrains_intermediates(self) -> bool:
        """Never: the rule is deferred to the terminal.

        Returns:
            ``False``, whether or not a transition matrix is set. This is the
            declaration that makes
            [is_constructible][evogfn.env.mutation.MutationEnvironment.is_constructible]
            collapse onto `is_reachable` here, which is the whole content of this
            class: no ordering is refused, so no ordering has to be found.
        """
        return False

    def _substitution_mask(
        self, state: State, capacity_left: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.bool_]:
        """Which substitutions are edges out of each state, adjacency ignored.

        Args:
            state: The current state.
            capacity_left: ``(n,)``, whether the trajectory may still mutate.

        Returns:
            An ``(n, length, v)`` boolean array, allowed where the position is
            untouched, the token is a genuine change, and budget remains. The
            transition matrix is not consulted: an intermediate is a notepad
            entry rather than a molecule.
        """
        untouched = state.sequences == self._parent[None, :]
        differs = self._parent[None, :, None] != np.arange(self._alphabet.size)[None, None, :]
        allowed: npt.NDArray[np.bool_] = (
            untouched[:, :, None] & differs & capacity_left[:, None, None]
        )
        return allowed

    def _may_stop(
        self, state: State, capacity_left: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.bool_]:
        """Which trajectories are entitled to stop: the feasible ones.

        This is where the deferred constraint lands. A trajectory on an
        infeasible sequence is not entitled to stop and must mutate on; one that
        has also run out of budget has no legal action at all, and `forward_mask`
        forces the stop, terminating it on an infeasible design.

        Args:
            state: The current state.
            capacity_left: ``(n,)``, whether the trajectory may still mutate.

        Returns:
            An ``(n,)`` boolean array.
        """
        return super()._may_stop(state, capacity_left) & self._feasible_rows(state.sequences)

    def _revertible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Every mutated position, since no intermediate is refused.

        The mirror of `_substitution_mask`. Keeping the base class's check --
        that reverting lands on a feasible state -- would make ``P_B`` describe a
        graph with fewer edges than ``P_F`` walks, and the balance condition
        would be fitted to a graph that is not the one being sampled.

        Args:
            sequences: An ``(n, length)`` array of token indices.

        Returns:
            An ``(n, length)`` all-true boolean array.
        """
        return np.ones(sequences.shape, dtype=np.bool_)

    def _feasible_rows(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Whether each sequence satisfies the transition constraint.

        Args:
            sequences: An ``(n, length)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array, all true where no constraint is set or
            where a sequence is too short to have an adjacency.
        """
        if self._transitions is None or self._length < 2:  # noqa: PLR2004 - a pair needs two
            return np.ones(sequences.shape[0], dtype=np.bool_)
        permitted = self._transitions > 0
        return np.asarray(
            np.all(permitted[sequences[:, :-1], sequences[:, 1:]], axis=1), dtype=np.bool_
        )
