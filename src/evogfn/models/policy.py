"""The policy network: forward and backward action distributions over a state.

Trajectory balance needs three things from a model -- ``P_F(s'|s)``,
``P_B(s|s')`` and a scalar ``log Z`` -- and each has a detail that matters more
than the architecture does.

**Masking is applied inside the model, not by the caller.** A caller who forgets
to mask gets a policy placing probability on edges that do not exist, and the
symptom is not a crash but a slightly wrong distribution.
[log_probs][evogfn.models.policy.SequencePolicy.log_probs] therefore takes the
mask and there is no way to obtain unmasked log probabilities by accident.

**The heads share a trunk except for their final layer**, following Malkin et
al. The two policies describe the same graph from opposite directions, so most
of what they need to compute is the same.

**``log Z`` is a bare parameter, not a head.** It does not depend on the state --
it is the total flow through the whole DAG -- and Malkin et al. report it needs a
learning rate roughly an order of magnitude above the policy's, which is why it
is exposed separately for the optimiser to put in its own parameter group.

**The backward policy is uniform by default and not learned.** On the mutation
environment's subset lattice a state with ``k`` mutations has exactly ``k``
parents, so uniform ``P_B`` is ``1/k`` in closed form; it is also the maximum
entropy choice there, and ``log P_B(τ|x) = -log k!`` is constant across
trajectories reaching the same terminal state, so it adds no path-dependent
variance to the loss. Learning it is still supported, because any valid ``P_B``
induces the same optimal ``P_F`` and Malkin et al. report a learned one
converging faster on some tasks -- so which is better here is an empirical
question, not a settled one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from evogfn.core.types import Tokens

#: Logit assigned to a masked action. Finite rather than ``-inf`` so that a row
#: which is entirely masked -- a stopped trajectory -- yields zeros instead of
#: ``nan`` and poisons nothing downstream. Large enough that a masked action's
#: probability is zero to within float32 precision.
MASKED_LOGIT = -1e9


class SequencePolicy(nn.Module):
    """Forward and backward action distributions over fixed-length sequences.

    An embedding of each position is concatenated and passed through an MLP
    trunk. That is adequate while sequences are short -- the benchmarks here run
    from 4 positions (GB1) to a few dozen -- and deliberately simple, since the
    claim being tested is about the training objective rather than the
    architecture. Longer sequences will want a different trunk; the interface
    does not change.

    Args:
        n_tokens: Alphabet size.
        sequence_length: Number of positions.
        n_actions: Size of the action space, including any stop action.
        embedding_dim: Width of the per-position embedding.
        hidden_dim: Width of the trunk.
        n_layers: Number of hidden layers in the trunk. Must be at least 1.
        learn_backward: Whether to learn ``P_B``. When ``False`` the backward
            policy is uniform over the parents permitted by the backward mask.
        seed: Seeds every parameter. ``None`` draws from global torch state,
            which means two policies built from the same configuration differ
            -- and in a paired benchmark that silently breaks the pairing the
            statistics rest on, because "same seed" then no longer means "same
            starting network".
        learn_flow: Whether to estimate a state flow ``log F(s)``. Needed by
            the detailed-balance family, which constrains flow through each
            *state*; trajectory balance constrains whole trajectories and does
            not use it.

    Raises:
        ValueError: If any size is not positive.
    """

    def __init__(  # noqa: PLR0913 - the network's shape is its definition
        self,
        *,
        n_tokens: int,
        sequence_length: int,
        n_actions: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        learn_backward: bool = False,
        learn_flow: bool = False,
        seed: int | None = None,
    ) -> None:
        """Build the trunk and heads."""
        super().__init__()
        for name, value in [
            ("n_tokens", n_tokens),
            ("sequence_length", sequence_length),
            ("n_actions", n_actions),
            ("embedding_dim", embedding_dim),
            ("hidden_dim", hidden_dim),
            ("n_layers", n_layers),
        ]:
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

        self._n_actions = n_actions
        self._learn_backward = learn_backward
        self._learn_flow = learn_flow

        # fork_rng isolates the draw: construction is reproducible without
        # the side effect of reseeding global state for everything after it.
        with torch.random.fork_rng(enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
            self._build_layers(
                n_tokens=n_tokens,
                sequence_length=sequence_length,
                n_actions=n_actions,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                learn_backward=learn_backward,
                learn_flow=learn_flow,
            )

    def _build_layers(  # noqa: PLR0913 - mirrors the constructor
        self,
        *,
        n_tokens: int,
        sequence_length: int,
        n_actions: int,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        learn_backward: bool,
        learn_flow: bool,
    ) -> None:
        """Create the trunk and heads, under whatever RNG state the caller set."""
        self.embedding = nn.Embedding(n_tokens, embedding_dim)

        layers: list[nn.Module] = []
        width = self._trunk_input_dim(sequence_length=sequence_length, embedding_dim=embedding_dim)
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Separate final layers only, so the shared representation is genuinely
        # shared.
        self.forward_head = nn.Linear(hidden_dim, n_actions)
        self.backward_head = nn.Linear(hidden_dim, n_actions) if learn_backward else None
        # One scalar per state. Unlike log Z this is a genuine head: the flow
        # through a state depends on the state.
        self.flow_head = nn.Linear(hidden_dim, 1) if learn_flow else None

        # Not a head: log Z is state-independent, being the total flow through
        # the DAG. Initialised at 0 (Z = 1) and expected to move a long way, so
        # the trainer gives it its own learning rate.
        self.log_z = nn.Parameter(torch.zeros(()))

    def _trunk_input_dim(self, *, sequence_length: int, embedding_dim: int) -> int:
        """Width of the vector the trunk's first layer reads.

        A seam rather than an expression, so that a subclass which feeds the
        trunk something extra states the width in the same place it states the
        content. Sizing the trunk from one and building the input from the other
        is the failure this prevents, and it presents as a shape error at the
        first forward pass rather than at construction.

        Args:
            sequence_length: Number of positions.
            embedding_dim: Width of the per-position embedding.

        Returns:
            The flattened per-position embedding, one embedding per position.
        """
        return sequence_length * embedding_dim

    def _trunk_input(self, sequences: torch.Tensor) -> torch.Tensor:
        """Build the trunk's input from a batch of states.

        Args:
            sequences: An ``(n, sequence_length)`` integer tensor.

        Returns:
            An ``(n, _trunk_input_dim)`` float tensor.
        """
        embedded: torch.Tensor = self.embedding(sequences)
        return embedded.flatten(start_dim=1)

    @property
    def n_actions(self) -> int:
        """Size of the action space."""
        return self._n_actions

    @property
    def learns_backward(self) -> bool:
        """Whether ``P_B`` is learned rather than uniform."""
        return self._learn_backward

    @property
    def learns_flow(self) -> bool:
        """Whether a state flow estimate is available."""
        return self._learn_flow

    def log_flow(self, sequences: torch.Tensor) -> torch.Tensor:
        """Estimate ``log F(s)`` for a batch of states.

        Args:
            sequences: An ``(n, length)`` tensor of token indices.

        Returns:
            An ``(n,)`` tensor of log flows.

        Raises:
            RuntimeError: If the policy was built without a flow head. Falling
                back to a constant would silently turn detailed balance into a
                worse trajectory balance.
        """
        if self.flow_head is None:
            raise RuntimeError(
                "this policy has no flow head; build it with learn_flow=True to "
                "use a detailed-balance objective"
            )
        flow: torch.Tensor = self.flow_head(self.forward(sequences))
        return flow.squeeze(-1)

    def policy_parameters(self) -> list[nn.Parameter]:
        """Every parameter except ``log Z``.

        Returns:
            Parameters for the optimiser's main group.
        """
        return [p for name, p in self.named_parameters() if name != "log_z"]

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Encode a batch of states.

        Args:
            sequences: An ``(n, sequence_length)`` integer tensor.

        Returns:
            An ``(n, hidden_dim)`` representation.
        """
        hidden: torch.Tensor = self.trunk(self._trunk_input(sequences))
        return hidden

    def log_probs(
        self,
        sequences: torch.Tensor,
        forward_mask: torch.Tensor,
        backward_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked log probabilities over forward and backward actions.

        Args:
            sequences: An ``(n, sequence_length)`` integer tensor of states.
            forward_mask: An ``(n, n_actions)`` boolean tensor of legal forward
                actions.
            backward_mask: An ``(n, n_actions)`` boolean tensor of actions that
                could have produced each state.

        Returns:
            Two ``(n, n_actions)`` tensors of log probabilities. Masked entries
            are ``-inf``. A fully masked row -- a stopped trajectory, which has
            no actions in either direction -- is all ``-inf`` rather than
            ``nan``.

        Raises:
            ValueError: If a mask does not match the action space.
        """
        self._check_mask(forward_mask, "forward_mask")
        self._check_mask(backward_mask, "backward_mask")

        hidden = self(sequences)
        forward = _masked_log_softmax(self.forward_head(hidden), forward_mask)

        if self.backward_head is not None:
            backward = _masked_log_softmax(self.backward_head(hidden), backward_mask)
        else:
            # Uniform over parents: exact on the subset lattice, and it needs no
            # network evaluation at all.
            backward = _uniform_log_probs(backward_mask)
        return forward, backward

    def _check_mask(self, mask: torch.Tensor, name: str) -> None:
        """Raise if a mask is not ``(n, n_actions)`` boolean."""
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean, got {mask.dtype}")
        if mask.ndim != 2 or mask.shape[1] != self._n_actions:  # noqa: PLR2004
            raise ValueError(
                f"{name} must have shape (n, {self._n_actions}), got {tuple(mask.shape)}"
            )


class AnchorConditionedPolicy(SequencePolicy):
    """A policy that is told which parent the design is being evolved *from*.

    What the unconditioned policy cannot see
    ----------------------------------------

    [SequencePolicy][evogfn.models.policy.SequencePolicy] reads the state
    sequence and nothing else. The anchor is therefore invisible to it: given the
    sequence ``ACGT`` it has no way to know whether that is the parent untouched,
    or the parent with three substitutions already written into it. Every
    question a directed-evolution policy would want to ask about its own position
    -- how far have I moved, which of these residues are mine, which are
    inherited -- is unanswerable from its input. What it can learn is a policy
    *for the parent it happens to be sitting at*, encoded implicitly in the
    weights, and it has to relearn that whenever the parent changes.

    What conditioning buys
    ----------------------

    Given the anchor alongside the state, the policy is learning a function *over
    anchors*: ``P_F(a | s, x_0)`` rather than ``P_F(a | s)`` with ``x_0`` baked
    into the weights. The object it fits is then defined at parents it has never
    been anchored at, so in principle it transfers to one -- a campaign's next
    round, or a different starting variant on the same landscape -- rather than
    having to be retrained into it. That is a claim about generalisation across
    parents, and it is worth making precisely because nothing else in this
    benchmark can make it: a genetic algorithm has no representation of "where am
    I" beyond the contents of its population, and an unconditioned policy has no
    input that distinguishes one anchor from another. It is available only to an
    arm that has a policy at all, which makes it structurally a GFlowNet result.

    How the anchor reaches the trunk
    --------------------------------

    Two channels, concatenated per position to the state embedding:

    * **The anchor's tokens, through the same embedding table as the state.** The
      anchor is a sequence in the same alphabet, so a second table would learn a
      second and unaligned representation of the same symbols. Concatenated
      rather than added, because addition loses the distinction between "state
      ``A``, anchor ``B``" and "state ``B``, anchor ``A``", which is exactly the
      direction of a substitution.
    * **One indicator per position, marking where the state differs from the
      anchor** -- the mutation mask. It is redundant in the information-theoretic
      sense and not in the practical one: equality of two embedding vectors is
      not an affine function of them, so the trunk's first layer cannot form it,
      and "which positions are mine" is the single most load-bearing thing the
      conditioning is for.

    [conditioning][evogfn.models.conditioning] is deliberately *not* used here,
    and the reason is that it solves a different problem. Thermometer encoding
    exists to widen a handful of continuous scalars so their gradient signal
    competes with a sequence embedding, and to keep the encoding continuous so
    nearby preferences share a representation. An anchor is neither a scalar nor
    continuous: it is a sequence of categorical tokens, already the same width as
    the state, and already served by the embedding table the state uses. Applying
    a thermometer to the binary mutation mask would produce ``n_bins`` copies of
    one bit -- there is nothing between 0 and 1 for the encoding to interpolate.

    Keeping the anchor in step with the environment
    -----------------------------------------------

    The anchor is state on the module, so it can go stale, and a stale anchor is
    silent: the conditioning is simply wrong and the loss still looks fine. The
    invariant is established at one seam --
    [GFlowNetSampler][evogfn.algorithms.gflownet.sampler.GFlowNetSampler] binds
    the policy to its environment's parent at construction, and a re-anchored
    campaign constructs a new sampler -- so a policy walking a graph always
    carries that graph's anchor. The consequence, which is the same one the
    sampler's own re-anchoring contract already carries, is that a policy shared
    between two samplers at different anchors belongs to the one constructed
    last.

    Args:
        anchor: The parent to condition on, shape ``(sequence_length,)``. Taken
            at construction rather than defaulted, because there is no anchor a
            policy could sensibly assume.
        n_tokens: Alphabet size.
        sequence_length: Number of positions.
        n_actions: Size of the action space, including any stop action.
        embedding_dim: Width of the per-position embedding.
        hidden_dim: Width of the trunk.
        n_layers: Number of hidden layers in the trunk.
        learn_backward: Whether to learn ``P_B``.
        learn_flow: Whether to estimate a state flow.
        seed: Seeds every parameter.

    Raises:
        ValueError: If any size is not positive, or if the anchor is not one
            sequence of ``sequence_length`` tokens drawn from the alphabet.
    """

    #: Declared so the buffer registered below is typed rather than reached
    #: through ``nn.Module``'s catch-all attribute lookup.
    _anchor: torch.Tensor

    def __init__(  # noqa: PLR0913 - mirrors the policy it extends
        self,
        *,
        anchor: Tokens,
        n_tokens: int,
        sequence_length: int,
        n_actions: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        learn_backward: bool = False,
        learn_flow: bool = False,
        seed: int | None = None,
    ) -> None:
        """Build the trunk at the widened input, then bind the anchor."""
        super().__init__(
            n_tokens=n_tokens,
            sequence_length=sequence_length,
            n_actions=n_actions,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            learn_backward=learn_backward,
            learn_flow=learn_flow,
            seed=seed,
        )
        self._n_tokens = n_tokens
        self._sequence_length = sequence_length
        # A buffer rather than a plain attribute so it follows the module to
        # another device and appears in the state dict: a policy restored
        # without its anchor would condition on whatever it was built with.
        self.register_buffer("_anchor", self._checked(anchor))

    @property
    def anchor(self) -> torch.Tensor:
        """The parent currently being conditioned on, as a ``(length,)`` tensor."""
        return self._anchor

    def set_anchor(self, anchor: Tokens) -> None:
        """Point the policy at a new parent, keeping every weight.

        The weights are anchor-free by construction -- the anchor enters as
        input, not as a parameter -- so moving it is exactly the operation this
        class exists to make possible, and it is what the amortisation claim
        rests on.

        Args:
            anchor: The parent to condition on, shape ``(sequence_length,)``.

        Raises:
            ValueError: If the anchor is not one sequence of this policy's
                length holding tokens the embedding table has rows for. Refused
                rather than reshaped: an anchor of the wrong length would index
                the embedding out of range or condition on a truncated parent,
                and only one of those raises.
        """
        self._anchor = self._checked(anchor)

    def _checked(self, anchor: Tokens) -> torch.Tensor:
        """Validate an anchor and return it as an ``int64`` tensor.

        Returns:
            A ``(sequence_length,)`` tensor of token indices.

        Raises:
            ValueError: If the anchor has the wrong shape or holds tokens
                outside the alphabet.
        """
        tokens = torch.as_tensor(anchor, dtype=torch.long)
        if tokens.ndim != 1 or int(tokens.shape[0]) != self._sequence_length:
            raise ValueError(
                f"anchor must be one sequence of length {self._sequence_length}, "
                f"got shape {tuple(tokens.shape)}"
            )
        if tokens.numel() and (int(tokens.min()) < 0 or int(tokens.max()) >= self._n_tokens):
            raise ValueError(
                f"anchor tokens must lie in [0, {self._n_tokens}), got "
                f"[{int(tokens.min())}, {int(tokens.max())}]"
            )
        return tokens

    def _trunk_input_dim(self, *, sequence_length: int, embedding_dim: int) -> int:
        """Width of the trunk input: state, anchor and one difference indicator.

        Args:
            sequence_length: Number of positions.
            embedding_dim: Width of the per-position embedding.

        Returns:
            ``sequence_length * (2 * embedding_dim + 1)``.
        """
        return sequence_length * (2 * embedding_dim + 1)

    def _trunk_input(self, sequences: torch.Tensor) -> torch.Tensor:
        """Concatenate the state, the anchor and where the two differ.

        Args:
            sequences: An ``(n, sequence_length)`` integer tensor of states.

        Returns:
            An ``(n, sequence_length * (2 * embedding_dim + 1))`` float tensor.
        """
        anchor = self._anchor.to(sequences.device).expand(sequences.shape[0], -1)
        state_embedded: torch.Tensor = self.embedding(sequences)
        anchor_embedded: torch.Tensor = self.embedding(anchor)
        differs = (sequences != anchor).to(state_embedded.dtype).unsqueeze(-1)
        joined = torch.cat([state_embedded, anchor_embedded, differs], dim=-1)
        return joined.flatten(start_dim=1)


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax restricted to the permitted actions.

    Masked logits are set to a large negative constant rather than ``-inf``,
    because a row with every action masked would otherwise softmax to ``nan``
    and contaminate the whole batch through the loss. Stopped trajectories are
    exactly that case.
    """
    restricted = logits.masked_fill(~mask, MASKED_LOGIT)
    log_probabilities = torch.log_softmax(restricted, dim=-1)
    # Report masked actions as -inf: their probability really is zero, and any
    # arithmetic that touches one should be visibly wrong rather than merely
    # very small.
    return log_probabilities.masked_fill(~mask, float("-inf"))


def _uniform_log_probs(mask: torch.Tensor) -> torch.Tensor:
    """Uniform distribution over the permitted actions in each row."""
    counts = mask.sum(dim=-1, keepdim=True)
    # A row with no permitted action contributes nothing; guard the division so
    # it yields -inf everywhere rather than nan.
    safe = counts.clamp(min=1)
    uniform = -torch.log(safe.to(mask.device, dtype=torch.float32)).expand_as(mask)
    return uniform.masked_fill(~mask, float("-inf"))


def to_tensor(sequences: Tokens, device: torch.device | str = "cpu") -> torch.Tensor:
    """Convert token indices to the integer tensor the policy expects.

    Args:
        sequences: An ``(n, sequence_length)`` array of token indices.
        device: Where to place the tensor.

    Returns:
        A ``(n, sequence_length)`` ``int64`` tensor.
    """
    return torch.as_tensor(sequences, dtype=torch.long, device=device)
