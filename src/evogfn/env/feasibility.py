"""What makes a sequence legal, as an object the environment can be handed.

Before this module the answer was one thing and only one thing: a transition
matrix over adjacent token pairs, reached by duck-typing
``landscape.transition_matrix``. That predicate is, by construction, the *easy*
case -- its constraint graph over positions is a path, so the completion oracle
is a dynamic program over ``(position, token, counter)`` and an exact projection
exists. The hard case, where no tractable sound mask exists, could not be
expressed at all.

A predicate here is **three vectorised tests**, and it is three rather than two
on purpose:

``is_feasible``
    The endpoint test, ``phi(x)``. What the feasible set is defined by.
``permits_substitution``
    The **forward** local mask: for every ``(position, token)``, may this
    substitution be made here?
``permits_reversion``
    The **backward** local mask: for every position, may this substitution be
    *undone* here?

An edge exists backward exactly where it exists forward, so a predicate that
answers the first and not the third gives ``P_B`` a different graph from
``P_F`` -- and trajectory balance is then being solved for a graph nobody walks
(`MutationEnvironment._revertible`). The third test is not a convenience.

**The invariant every local mask here is written against.** A trajectory starts
at a feasible anchor and every admitted move lands on a feasible state, so a
mask may assume the *current* state is feasible and check only what its own move
disturbs. `AdjacencyPredicate` checks two pairs rather than ``L-1``;
`ContactPredicate` checks the contacts incident to the moved position.
`BudgetBandPredicate` needs no such assumption -- it recomputes the total
exactly -- and is written that way regardless, so the three agree on what a mask
means.

**What `factorises` is for.** It records whether a *tractable* completion oracle
exists for this predicate, which is the condition the soundness argument turns
on. It is a property of the predicate and of the variable set -- the induced
width of the constraint graph over positions -- and **not** of the action space.
A predicate can factorise and still admit a local mask that loses support,
because the mask tests ``phi(s')`` where the oracle tests "does a feasible
completion exist"; those coincide for prefixes and diverge for edits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evogfn.core.types import Tokens

__all__ = [
    "AdjacencyPredicate",
    "BudgetBandPredicate",
    "CodebookPredicate",
    "ConjunctionPredicate",
    "ContactPredicate",
    "FeasibilityPredicate",
]


@runtime_checkable
class FeasibilityPredicate(Protocol):
    """A rule saying which complete sequences are legal, and which moves keep them so."""

    @property
    def factorises(self) -> bool:
        """Whether a completion oracle for this predicate is tractable.

        Returns:
            ``True`` where the constraint graph over positions has bounded
            induced width, so the oracle is a dynamic program and an exact
            projection exists; ``False`` where no such algorithm is known.
        """
        ...

    @property
    def induced_width(self) -> int:
        """Induced width of the constraint graph over positions.

        Returns:
            The width. ``1`` for a chain, and the sequence length where every
            position is coupled to every other, which is the case a completion
            oracle cannot be built for.
        """
        ...

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Whether each complete sequence satisfies the predicate.

        Args:
            sequences: An ``(n, length)`` array of token indices.

        Returns:
            An ``(n,)`` boolean array.
        """
        ...

    def permits_substitution(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Which single substitutions keep the sequence legal.

        Args:
            sequences: An ``(n, length)`` array of token indices, assumed
                feasible (see the module docstring).

        Returns:
            An ``(n, length, alphabet)`` boolean array.
        """
        ...

    def permits_reversion(self, sequences: Tokens, parent: Tokens) -> npt.NDArray[np.bool_]:
        """Which single reversions to the anchor keep the sequence legal.

        Args:
            sequences: An ``(n, length)`` array of token indices.
            parent: The anchor, of shape ``(length,)``.

        Returns:
            An ``(n, length)`` boolean array.
        """
        ...


class AdjacencyPredicate:
    """Every adjacent token pair must be permitted by a transition matrix.

    The predicate this project shipped, extracted unchanged. Its constraint
    graph over positions is a path, so `induced_width` is 1 and the completion
    oracle is a Viterbi-style dynamic program -- which is why an exact
    projection exists for it and a learned sampler buys no support advantage
    where the score is additive.

    Attributes:
        transitions: The ``(v, v)`` matrix whose zeros mark forbidden pairs.
    """

    def __init__(self, transitions: npt.NDArray[np.floating], *, length: int) -> None:
        """Store the matrix and the shape the masks are built at.

        Args:
            transitions: A ``(v, v)`` array; entries above zero are permitted.
            length: Sequence length.

        Raises:
            ValueError: If the matrix is not square.
        """
        matrix = np.asarray(transitions)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:  # noqa: PLR2004 - a matrix is 2-D
            raise ValueError(f"transitions must be a square matrix, got {matrix.shape}")
        self.transitions = matrix
        self._permitted = matrix > 0
        self._size = int(matrix.shape[0])
        self._length = int(length)

    @property
    def factorises(self) -> bool:
        """A chain always admits a dynamic-programming completion oracle."""
        return True

    @property
    def induced_width(self) -> int:
        """One: each position is coupled only to its two neighbours."""
        return 1

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Whether every adjacent pair is permitted, per sequence."""
        array = np.atleast_2d(np.asarray(sequences))
        if self._length < 2:  # noqa: PLR2004 - a pair needs two positions
            return np.ones(array.shape[0], dtype=np.bool_)
        return np.asarray(self._permitted[array[:, :-1], array[:, 1:]].all(axis=1), dtype=np.bool_)

    def permits_substitution(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Which substitutions keep both disturbed adjacencies permitted."""
        array = np.asarray(sequences)
        n = array.shape[0]
        candidates = np.arange(self._size)
        allowed = np.ones((n, self._length, self._size), dtype=np.bool_)
        if self._length > 1:
            # Substituting at position p constrains the (p-1, p) and (p, p+1)
            # pairs only; every other adjacency is untouched.
            left = array[:, :-1]
            allowed[:, 1:, :] &= self._permitted[left[:, :, None], candidates[None, None, :]]
            right = array[:, 1:]
            allowed[:, :-1, :] &= self._permitted[candidates[None, None, :], right[:, :, None]]
        return allowed

    def permits_reversion(self, sequences: Tokens, parent: Tokens) -> npt.NDArray[np.bool_]:
        """Which reversions keep both disturbed adjacencies permitted."""
        array = np.asarray(sequences)
        anchor = np.asarray(parent)
        reverted = np.broadcast_to(anchor[None, :], array.shape)
        allowed = np.ones(array.shape, dtype=np.bool_)
        if self._length > 1:
            # Reverting position p only disturbs the (p-1, p) and (p, p+1) pairs.
            allowed[:, 1:] &= self._permitted[array[:, :-1], reverted[:, 1:]]
            allowed[:, :-1] &= self._permitted[reverted[:, :-1], array[:, 1:]]
        return allowed

    def orderable(self, sequences: Tokens, parent: Tokens) -> npt.NDArray[np.bool_]:
        """Whether some ordering of each design's substitutions stays legal throughout.

        Exact, and linear rather than factorial, because the conditions this
        predicate imposes on an ordering constrain a **path**: an adjacent pair
        changes value only when one of its two positions is substituted, three of
        its four value combinations are fixed by the anchor and the destination
        both being feasible, and only the mixed one depends on which position
        moves first. Any orientation of a path is acyclic, so satisfying each
        pair separately yields a genuine global order.

        This method exists on this class and not on the protocol precisely
        because that argument is a property of the *chain*. A predicate whose
        constraint graph is not a path has no such closed form, and
        `MutationEnvironment.is_constructible` refuses rather than guessing.

        Args:
            sequences: An ``(n, length)`` array of token indices.
            parent: The anchor, of shape ``(length,)``.

        Returns:
            An ``(n,)`` boolean array.
        """
        array = np.asarray(sequences)
        anchor = np.asarray(parent)
        if self._length < 2:  # noqa: PLR2004 - a pair needs two positions
            return np.ones(array.shape[0], dtype=np.bool_)
        substituted = array != anchor[None, :]
        # The only pairs that can constrain an ordering: both ends substituted.
        coupled = substituted[:, :-1] & substituted[:, 1:]
        left_first = self._permitted[array[:, :-1], anchor[None, 1:]]
        right_first = self._permitted[anchor[None, :-1], array[:, 1:]]
        orderable = ~coupled | left_first | right_first
        return np.asarray(orderable.all(axis=1), dtype=np.bool_)


class ContactPredicate:
    """Named position pairs must carry a permitted token pair.

    The generalisation of `AdjacencyPredicate` off the diagonal, and the reason
    it exists: the contacts can be arbitrarily long-range, so the induced width
    of the constraint graph is a **dial** rather than a constant. The completion
    oracle costs ``O(L * v**(w+1) * B)`` at width ``w``, so this class spans the
    tractable and intractable regimes continuously, which a single hard instance
    cannot.

    `AdjacencyPredicate` is the width-1 special case. It is kept separate rather
    than reduced to this one because its two-pair form is the shipped code and a
    faster path, and because a behaviour-preserving extraction is what licenses
    the bless of every stored record.

    Attributes:
        pairs: An ``(m, 2)`` integer array of coupled positions.
        permitted: An ``(m, v, v)`` boolean array, one matrix per contact.
    """

    #: Above this width the oracle is treated as intractable. Chosen as the
    #: point where ``v**(w+1)`` leaves the range a benchmark can enumerate at
    #: the alphabet sizes here; it is a declaration, not a theorem.
    TRACTABLE_WIDTH = 2

    def __init__(
        self,
        pairs: npt.NDArray[np.integer],
        permitted: npt.NDArray[np.bool_],
        *,
        length: int,
        alphabet_size: int,
    ) -> None:
        """Store the contact list and its per-contact token matrices.

        Args:
            pairs: An ``(m, 2)`` array of position indices.
            permitted: An ``(m, v, v)`` boolean array.
            length: Sequence length.
            alphabet_size: Tokens available at each position.

        Raises:
            ValueError: If the shapes disagree or a position is out of range.
        """
        self.pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self.permitted = np.asarray(permitted, dtype=np.bool_)
        self._length = int(length)
        self._size = int(alphabet_size)
        if self.permitted.shape != (self.pairs.shape[0], self._size, self._size):
            raise ValueError(
                f"permitted must be {(self.pairs.shape[0], self._size, self._size)}, "
                f"got {self.permitted.shape}"
            )
        if self.pairs.size and (self.pairs.min() < 0 or self.pairs.max() >= self._length):
            raise ValueError(f"contact positions must lie in [0, {self._length})")
        if np.any(self.pairs[:, 0] == self.pairs[:, 1]):
            raise ValueError("a contact must couple two distinct positions")

    @property
    def factorises(self) -> bool:
        """Whether the induced width is small enough for the oracle to be built."""
        return self.induced_width <= self.TRACTABLE_WIDTH

    @property
    def induced_width(self) -> int:
        """An upper bound on the treewidth of the contact graph, by min-degree elimination.

        Eliminate the lowest-degree position, connect its neighbours to each
        other, and record the degree it had; the largest such degree bounds the
        treewidth from above. Exact treewidth is NP-hard and nothing here needs
        it exactly -- but the bound has to be *tight enough to be right about the
        easy end*, which maximum degree is not: a path has maximum degree 2 and
        treewidth 1, so reading degree would have this class report width 2 for
        the very constraint graph `AdjacencyPredicate` correctly calls width 1,
        and `factorises` reads this number.

        Returns:
            The bound, or 0 where there are no contacts.
        """
        if not self.pairs.size:
            return 0
        neighbours: dict[int, set[int]] = {p: set() for p in range(self._length)}
        for left, right in self.pairs:
            neighbours[int(left)].add(int(right))
            neighbours[int(right)].add(int(left))

        width = 0
        remaining = {p for p, n in neighbours.items() if n}
        while remaining:
            position = min(remaining, key=lambda p: len(neighbours[p]))
            clique = neighbours[position]
            width = max(width, len(clique))
            for other in clique:
                neighbours[other].discard(position)
                neighbours[other] |= clique - {other}
            del neighbours[position]
            remaining.remove(position)
        return width

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Whether every contact carries a permitted token pair."""
        array = np.atleast_2d(np.asarray(sequences))
        if not self.pairs.size:
            return np.ones(array.shape[0], dtype=np.bool_)
        left = array[:, self.pairs[:, 0]]
        right = array[:, self.pairs[:, 1]]
        contact = np.arange(self.pairs.shape[0])
        return np.asarray(self.permitted[contact[None, :], left, right].all(axis=1), dtype=np.bool_)

    def permits_substitution(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Which substitutions keep every contact incident to the moved position legal."""
        array = np.asarray(sequences)
        n = array.shape[0]
        allowed = np.ones((n, self._length, self._size), dtype=np.bool_)
        # A loop over contacts, not over positions: the contact list is sparse
        # by construction, and each iteration is a vectorised gather over the
        # whole batch and the whole alphabet.
        for k, (i, j) in enumerate(self.pairs):
            allowed[:, i, :] &= self.permitted[k][:, array[:, j]].T
            allowed[:, j, :] &= self.permitted[k][array[:, i], :]
        return allowed

    def permits_reversion(self, sequences: Tokens, parent: Tokens) -> npt.NDArray[np.bool_]:
        """Which reversions keep every contact incident to the moved position legal."""
        array = np.asarray(sequences)
        anchor = np.asarray(parent)
        allowed = np.ones(array.shape, dtype=np.bool_)
        for k, (i, j) in enumerate(self.pairs):
            allowed[:, i] &= self.permitted[k][anchor[i], array[:, j]]
            allowed[:, j] &= self.permitted[k][array[:, i], anchor[j]]
        return allowed


class BudgetBandPredicate:
    """A weighted running total over the whole sequence must stay inside a band.

    Every position is coupled to every other, so `induced_width` is the sequence
    length and no completion oracle of the usual shape exists. Deciding whether a
    legal construction order exists is **NP-complete**, by reduction from
    PARTITION: give ``n`` positions the weights of a PARTITION instance summing
    to ``2S`` and one further position the weight ``-S``, with the band
    ``[0, S]``. Since the positive weights only ever raise the running total, the
    total immediately before the negative position must be at most ``S``; and
    since the total after it must be at least ``0``, that same total must be at
    least ``S``. So it is exactly ``S``, and the positions preceding the negative
    one are a PARTITION half. Conversely any half orders as: the half, then the
    negative position, then the rest.

    This is the *hard endpoint* of the width dial `ContactPredicate` spans.
    Unit weights recover a counter with at most ``L`` states -- GC content, and
    the other predicates a small automaton recognises -- where the oracle is a
    polynomial dynamic program again. The hardness needs the weights to carry
    range, so it is **weak** NP-hardness: a pseudo-polynomial dynamic program
    over the total exists, and instances with small weights are easy. Two
    independent bands, or weights at a realistic magnitude and precision, are
    what make it bite.

    Attributes:
        weights: An ``(length, v)`` integer array; the total of a sequence is the
            sum of the weight of the token at each position.
        low: Inclusive lower bound on the total.
        high: Inclusive upper bound on the total.
    """

    def __init__(
        self, weights: npt.NDArray[np.integer], *, low: int, high: int, length: int
    ) -> None:
        """Store the weights and the band.

        Args:
            weights: An ``(length, v)`` integer array.
            low: Inclusive lower bound.
            high: Inclusive upper bound.
            length: Sequence length.

        Raises:
            ValueError: If the weights are the wrong shape or the band is empty.
        """
        self.weights = np.asarray(weights, dtype=np.int64)
        if self.weights.ndim != 2 or self.weights.shape[0] != length:  # noqa: PLR2004 - 2-D
            raise ValueError(f"weights must be (length, v) with length {length}")
        if low > high:
            raise ValueError(f"band [{low}, {high}] is empty")
        self.low = int(low)
        self.high = int(high)
        self._length = int(length)
        self._positions = np.arange(self._length)

    @property
    def factorises(self) -> bool:
        """No: the completion oracle here is NP-complete (see the class docstring)."""
        return False

    @property
    def induced_width(self) -> int:
        """The sequence length: the running total couples every position."""
        return self._length

    def totals(self, sequences: Tokens) -> npt.NDArray[np.int64]:
        """The weighted total of each sequence.

        Args:
            sequences: An ``(n, length)`` array of token indices.

        Returns:
            An ``(n,)`` integer array.
        """
        array = np.atleast_2d(np.asarray(sequences))
        return np.asarray(self.weights[self._positions[None, :], array].sum(axis=1))

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Whether each sequence's total lies inside the band."""
        total = self.totals(sequences)
        return np.asarray((total >= self.low) & (total <= self.high))

    def permits_substitution(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Which substitutions leave the total inside the band.

        Exact rather than assumed: the total after substituting ``t`` at ``p`` is
        the current total less the weight of the token being replaced plus the
        weight of ``t``, so no invariant about the current state is needed.
        """
        array = np.asarray(sequences)
        current = self.totals(array)[:, None, None]
        removed = self.weights[self._positions[None, :], array][:, :, None]
        added = self.weights[None, :, :]
        total = current - removed + added
        return np.asarray((total >= self.low) & (total <= self.high))

    def permits_reversion(self, sequences: Tokens, parent: Tokens) -> npt.NDArray[np.bool_]:
        """Which reversions leave the total inside the band."""
        array = np.asarray(sequences)
        anchor = np.asarray(parent)
        current = self.totals(array)[:, None]
        removed = self.weights[self._positions[None, :], array]
        added = self.weights[self._positions, anchor][None, :]
        total = current - removed + added
        return np.asarray((total >= self.low) & (total <= self.high))


class ConjunctionPredicate:
    """Several predicates at once; a design is legal when all of them admit it.

    The construction that makes a local mask genuinely insufficient. A single
    band is not enough on its own: from a state whose running total has left the
    band, one substitution of large enough weight moves the total back inside it,
    so a mask that looks one move ahead recovers the whole feasible set and
    nothing has to be learned. Measured on an enumerable instance, depth-one
    lookahead took the reachable share from 0.0005 to 1.000.

    Two totals whose weights are drawn independently do not admit that repair.
    A substitution changes both at once, and a move that carries the first total
    back into its band generally carries the second out of its own, so no single
    move restores feasibility and the depth of lookahead a sound mask needs grows
    with how many constraints have to be satisfied simultaneously. That is also
    the setting a design campaign actually poses -- net charge *and* hydrophobic
    fraction, expression *and* stability -- rather than a single scalar budget.

    Attributes:
        parts: The predicates conjoined, in the order given.
    """

    def __init__(self, parts: Sequence[FeasibilityPredicate]) -> None:
        """Store the conjuncts.

        Args:
            parts: Two or more predicates over the same sequence shape.

        Raises:
            ValueError: If fewer than two predicates are given, which would be a
                conjunction in name only and would hide the single-predicate
                case behind a wrapper.
        """
        if len(parts) < 2:  # noqa: PLR2004 - a conjunction needs two
            raise ValueError(f"a conjunction needs at least two predicates, got {len(parts)}")
        self.parts = tuple(parts)

    @property
    def factorises(self) -> bool:
        """Whether every conjunct admits a tractable completion oracle.

        Returns:
            ``True`` only when all of them do. A conjunction inherits the
            hardest conjunct: an oracle for the whole must answer for each part
            simultaneously, so one intractable part is enough to lose the
            guarantee, and satisfying the parts separately does not satisfy them
            together.
        """
        return all(part.factorises for part in self.parts)

    @property
    def induced_width(self) -> int:
        """The largest width among the conjuncts.

        Returns:
            The maximum. This is a *lower* bound on the width of the union of
            the conjuncts' constraint graphs, and it is exact where one
            conjunct's graph contains the others' -- which is the case that
            matters here, since two running totals over every position are each
            already complete. Reported rather than computed on the union so the
            number is cheap and never overstates tractability.
        """
        return max(part.induced_width for part in self.parts)

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Whether every conjunct admits each sequence."""
        array = np.atleast_2d(np.asarray(sequences))
        admitted = np.ones(array.shape[0], dtype=np.bool_)
        for part in self.parts:
            admitted &= part.is_feasible(array)
        return admitted

    def permits_substitution(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Which substitutions every conjunct permits."""
        array = np.asarray(sequences)
        allowed = self.parts[0].permits_substitution(array).copy()
        for part in self.parts[1:]:
            allowed &= part.permits_substitution(array)
        return allowed

    def permits_reversion(self, sequences: Tokens, parent: Tokens) -> npt.NDArray[np.bool_]:
        """Which reversions every conjunct permits."""
        array = np.asarray(sequences)
        allowed = self.parts[0].permits_reversion(array, parent).copy()
        for part in self.parts[1:]:
            allowed &= part.permits_reversion(array, parent)
        return allowed


class CodebookPredicate:
    """Legal designs are an explicit set whose members are mutually well separated.

    The construction that makes the *depth* of lookahead a sound mask needs into
    a controlled quantity, and it exists because the two earlier attempts did
    not. A single weighted band, and then two independent bands, both admitted
    repair by one substitution: from a state whose totals had left their bands,
    some single move among the ``length * alphabet`` available carried them back,
    so a mask looking one move ahead recovered essentially the whole feasible
    set -- measured, from a reachable share of 0.002 to 0.998. The reason is
    counting rather than structure. With far more moves available than
    constraints to violate, a repairing move almost always exists.

    Separation removes it by construction. Where every pair of legal designs
    differs in at least ``separation`` positions, moving between two of them
    requires that many substitutions and every intermediate is illegal, so a
    mask that looks fewer than ``separation - 1`` moves ahead admits nothing at
    all and the reachable set collapses to the anchor. The depth at which the
    support reappears is then a property of the predicate that can be dialled,
    and it is exactly the cost of soundness: a mask certain to be sound must
    search ``alphabet**depth`` continuations per move.

    Attributes:
        separation: Least Hamming distance between two legal designs.
        designs: The codebook itself. Exposed because it *is* the feasible set,
            stated rather than searched for -- which is what lets a measurement
            taken against that set run at lengths where the Hamming ball is far
            beyond enumeration.
    """

    def __init__(
        self, designs: Tokens, *, length: int, alphabet_size: int, separation: int
    ) -> None:
        """Store the codebook, encoded for constant-time membership.

        Sequences are encoded as integers in base ``alphabet_size`` so that
        membership is a sorted-array lookup rather than a per-row comparison
        against the whole book; the masks below ask ``length * alphabet``
        membership questions per state, and a linear scan would dominate.

        Args:
            designs: An ``(m, length)`` array of legal designs.
            length: Sequence length.
            alphabet_size: Tokens per position.
            separation: Least Hamming distance between distinct designs, which
                the caller guarantees and this class records.

        Raises:
            ValueError: If the codebook is empty or the wrong shape.
        """
        book = np.atleast_2d(np.asarray(designs))
        if book.size == 0 or book.shape[1] != length:
            raise ValueError(f"designs must be (m, {length}), got {book.shape}")
        self._length = int(length)
        self._size = int(alphabet_size)
        self.separation = int(separation)
        self._place = self._size ** np.arange(self._length)
        self._codes = np.sort(book @ self._place)
        self.designs = book

    def _encode(self, sequences: np.ndarray) -> np.ndarray:
        """Base-``alphabet`` codes for a batch of sequences."""
        return np.asarray(sequences) @ self._place

    @property
    def factorises(self) -> bool:
        """No: an explicit set carries no structure a dynamic program can exploit."""
        return False

    @property
    def induced_width(self) -> int:
        """The sequence length: membership couples every position at once."""
        return self._length

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Whether each sequence is in the codebook."""
        array = np.atleast_2d(np.asarray(sequences))
        return np.isin(self._encode(array), self._codes)

    def permits_substitution(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Which substitutions land on another codeword.

        With ``separation`` at two or more this admits nothing from a legal
        state, which is the point: the local mask cannot move at all, and every
        design beyond the anchor is reached only by a mask that looks ahead.
        """
        array = np.asarray(sequences)
        codes = self._encode(array)
        tokens = np.arange(self._size)
        # Substituting position p replaces its contribution to the code.
        delta = (tokens[None, None, :] - array[:, :, None]) * self._place[None, :, None]
        return np.isin(codes[:, None, None] + delta, self._codes)

    def permits_reversion(self, sequences: Tokens, parent: Tokens) -> npt.NDArray[np.bool_]:
        """Which reversions to the anchor land on another codeword."""
        array = np.asarray(sequences)
        anchor = np.asarray(parent)
        codes = self._encode(array)
        delta = (anchor[None, :] - array) * self._place[None, :]
        return np.isin(codes[:, None] + delta, self._codes)
