r"""A de-novo autoregressive construction graph over fixed-length sequences.

Where [MutationEnvironment][evogfn.env.mutation.MutationEnvironment] edits an
anchor, this environment builds a sequence left to right from nothing, appending
one token per step until all ``length`` positions are filled. It carries no
feasibility predicate: every sequence of the right length is reachable, which is
the setting the GFlowNet sequence-design literature trains in
\\citep{jain2022biological} and the setting GFNSeqEditor's flow is trained in
before its editing procedure is run (see
[GFNSeqEditorSampler][evogfn.algorithms.baselines.gfnseqeditor.GFNSeqEditorSampler]).

The construction graph is a tree: a partial sequence has exactly one parent, the
sequence with its last-filled position removed, so the backward policy is trivial
(``P_B = 1`` on the one legal reverse edge). Acyclicity holds because the number
of filled positions strictly increases along every forward edge. Both properties
are what the trajectory balance objective requires, and they are exercised by the
environment invariants in the test-suite rather than assumed here.

A partial construction is stored as a full ``(n, length)`` token array whose
positions from a left-to-right cursor onward hold a **pad** token, ``alphabet.size``.
Consumers that embed the sequence therefore see ``alphabet.size + 1`` distinct
tokens; the policy built for this environment sizes its embedding table
accordingly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.env.base import SequenceEnvironment, State

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Alphabet, Tokens


class DeNovoEnvironment(SequenceEnvironment):
    """Left-to-right construction of fixed-length sequences, unconstrained.

    Actions ``0 .. v-1`` append that token at the cursor; action ``v`` (the stop
    action) terminates and is available only once all ``length`` positions are
    filled. The pad token is ``v = alphabet.size``.
    """

    def __init__(self, alphabet: Alphabet, length: int) -> None:
        """Store the alphabet and length and fix the pad and stop indices.

        Args:
            alphabet: The alphabet sequences are written in.
            length: The fixed length every constructed sequence has.
        """
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        self._alphabet = alphabet
        self._length = int(length)
        self._v = int(alphabet.size)
        self._pad = self._v  # token index reserved for an unfilled position

    @property
    def alphabet(self) -> Alphabet:
        """The alphabet sequences are written in."""
        return self._alphabet

    @property
    def sequence_length(self) -> int:
        """Length of the sequences this environment constructs."""
        return self._length

    @property
    def n_actions(self) -> int:
        """The ``v`` append actions plus one stop action."""
        return self._v + 1

    @property
    def stop_action(self) -> int:
        """The index of the terminating action."""
        return self._v

    @property
    def n_tokens(self) -> int:
        """Distinct token indices a consumer must embed: the alphabet plus pad."""
        return self._v + 1

    def _cursor(self, sequences: Tokens) -> npt.NDArray[np.int64]:
        """Number of filled (non-pad) positions per row.

        Positions are filled left to right, so the non-pad positions are a prefix
        and their count is the cursor.
        """
        return np.asarray(sequences != self._pad, dtype=np.int64).sum(axis=1)

    def initial(self, n: int) -> State:
        """Create ``n`` all-pad trajectories at the source of the graph."""
        sequences = np.full((n, self._length), self._pad, dtype=np.int32)
        stopped = np.zeros(n, dtype=np.bool_)
        return State(sequences=sequences, stopped=stopped)

    def forward_mask(self, state: State) -> npt.NDArray[np.bool_]:
        """Append any token while positions remain; stop only when the sequence is full."""
        n = len(state)
        mask = np.zeros((n, self.n_actions), dtype=np.bool_)
        cursor = self._cursor(state.sequences)
        active = ~np.asarray(state.stopped, dtype=np.bool_)
        filling = active & (cursor < self._length)
        full = active & (cursor >= self._length)
        mask[filling, : self._v] = True
        mask[full, self.stop_action] = True
        return mask

    def backward_mask(self, state: State) -> npt.NDArray[np.bool_]:
        """The single reverse edge: undo the stop, else remove the last-filled token."""
        n = len(state)
        mask = np.zeros((n, self.n_actions), dtype=np.bool_)
        stopped = np.asarray(state.stopped, dtype=np.bool_)
        mask[stopped, self.stop_action] = True
        cursor = self._cursor(state.sequences)
        reverting = (~stopped) & (cursor > 0)
        rows = np.nonzero(reverting)[0]
        last = cursor[rows] - 1
        tokens = state.sequences[rows, last]
        mask[rows, tokens] = True
        return mask

    def step(self, state: State, actions: npt.NDArray[np.integer]) -> State:
        """Append a token at the cursor, or take the stop action."""
        actions = np.asarray(actions)
        sequences = np.array(state.sequences, dtype=np.int32, copy=True)
        stopped = np.array(state.stopped, dtype=np.bool_, copy=True)
        stopping = actions == self.stop_action
        appending = ~stopping
        rows = np.nonzero(appending)[0]
        cursor = self._cursor(state.sequences)
        sequences[rows, cursor[rows]] = actions[rows].astype(np.int32)
        stopped[stopping] = True
        return State(sequences=sequences, stopped=stopped)

    def backward_step(self, state: State, actions: npt.NDArray[np.integer]) -> State:
        """Undo the stop, or clear the last-filled position back to pad."""
        actions = np.asarray(actions)
        sequences = np.array(state.sequences, dtype=np.int32, copy=True)
        stopped = np.array(state.stopped, dtype=np.bool_, copy=True)
        unstopping = actions == self.stop_action
        stopped[unstopping] = False
        reverting = ~unstopping
        rows = np.nonzero(reverting)[0]
        cursor = self._cursor(state.sequences)
        last = cursor[rows] - 1
        sequences[rows, last] = self._pad
        return State(sequences=sequences, stopped=stopped)
