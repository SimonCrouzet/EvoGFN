r"""A policy that is told which trade-off it is being asked for.

MOGFN-PC (Jain et al., 2023, *Multi-Objective GFlowNets*, ICML) trains a single
model over ``p(x|ω) ∝ R(x|ω)^β``. Three things make that one model rather than
``N``: the preference reaches the *policy* as an input, the preference reaches
the *reward*, and the partition function becomes a function of the preference.
This module holds the first and the third. The second lives in
[ScalarizedReward][evogfn.rewards.scalarization.ScalarizedReward] and is applied
by the training loop.

Why this is a separate module and not a sibling of ``AnchorConditionedPolicy``
------------------------------------------------------------------------------

Convention says it belongs in [evogfn.models.policy][], next to the other
conditioned policy and next to the seam it overrides. It is here instead for a
reason that has nothing to do with design: ``policy.py`` is in the dependency
closure of *both* benchmark suites, and
[ResultStore][evogfn.benchmark.store.ResultStore] fingerprints raw file bytes
rather than a normalised AST -- so adding a class to it, or even a comment,
invalidates every stored single-objective record. A new module joins the
multi-objective closure only, where nothing is stored yet. Moving this class
into ``policy.py`` is the right end state and is itself an invalidating edit, so
it should be batched with whatever next has to touch that file.

How ``ω`` reaches the trunk, and why it is not tiled
----------------------------------------------------

[preference_encoding][evogfn.models.conditioning.preference_encoding] widens each
simplex weight into ``n_bins`` monotonically-filling coordinates, and the result
is concatenated **once** to the flattened per-position embedding:

    trunk input width = sequence_length * embedding_dim + encoding_dim(k, n_bins)

[AnchorConditionedPolicy][evogfn.models.policy.AnchorConditionedPolicy] tiles its
conditioning per position because an anchor genuinely *is* per-position. A
preference is not: tiling it would multiply the conditioning width by the
sequence length and add no information at all.

The cost of not tiling is stated rather than hidden, because it is the failure
[evogfn.models.conditioning][] was written about. A once-concatenated ``ω`` is a
smaller share of the trunk input than a tiled one would be -- at ``L = 10``,
``embedding_dim = 32``, two objectives and 16 bins it is 32 dimensions against
320, or 9% -- and a conditioning signal that is a whisper next to a shout is
fitted around rather than conditioned on. `n_bins` is the dial for that share and
it is the first thing to raise if a front collapses to a point; tiling is the
second, and would be a change to the method.

``Z_θ(ω)``, and why the scalar was not kept
--------------------------------------------

``log Z(ω) = log Σ_x R(x|ω)^β`` genuinely varies with ``ω``: at one corner of the
simplex the reward is one objective's, at another it is a different objective's,
and the two sums are unrelated. Keeping
[SequencePolicy][evogfn.models.policy.SequencePolicy]'s single scalar would leave
the trajectory-balance residual absorbing that variation, and the policy is then
pushed to compensate -- over-sampling wherever the fitted scalar happens to sit
and under-sampling the rest of the simplex.

That failure matters more than its size suggests. Its symptom is a Pareto front
collapsing toward one trade-off, which is *exactly* what "preference conditioning
does not help in this regime" would look like. Getting it wrong does not produce
a worse result, it produces the wrong answer to the question wearing the right
answer's clothes. So the head is built, and
[ConditionalTrajectoryBalance][evogfn.algorithms.gflownet.preference_conditioned.ConditionalTrajectoryBalance]
refuses any policy that cannot supply it.

The inherited scalar ``log_z`` still exists, because the base class builds it. It
is trained by nothing, read by nothing here, and deliberately placed in no
optimiser group -- see
[conditional_parameter_groups][evogfn.models.preference_policy.conditional_parameter_groups].

Statefulness, and the contract it inherits
-------------------------------------------

The preference is held on the module, so it can go stale exactly as a held anchor
can, and a stale preference is silent: the policy conditions on a trade-off
nobody asked for, the loss stays finite, and the front is simply wrong. The
invariant is that ``ω`` is set immediately before every sampling call and every
loss computation and never between them, and it is owned by
[PreferenceConditionedSampler][evogfn.algorithms.gflownet.preference_sampler.PreferenceConditionedSampler]
in the same way ``_bind_anchor`` owns the anchor's.

This is fragile, and it is fragile for a reason worth recording rather than
excusing: [Trajectories][evogfn.algorithms.gflownet.sampling.Trajectories] has no
field for the preference that produced it, and adding one would edit a module in
both closures. With such a field the objective could read ``ω`` off the batch and
the invariant would be checkable rather than maintained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from evogfn.models.conditioning import MIN_BINS, encoding_dim, preference_encoding
from evogfn.models.policy import SequencePolicy

if TYPE_CHECKING:
    import numpy.typing as npt

#: Bins per objective in the thermometer encoding. Sixteen, following the MOGFN
#: reference implementation. It is also the dial on how loud the conditioning is
#: relative to the state embedding: see the module docstring.
DEFAULT_PREFERENCE_BINS = 16

#: Width of the hidden layer in ``Z_θ(ω)``. Small deliberately -- the head maps a
#: low-dimensional encoding to one number, and a wide one would be capacity spent
#: where there is nothing to fit.
DEFAULT_LOG_Z_HIDDEN_DIM = 32

#: Attribute name of the conditional partition function, and the prefix
#: `conditional_parameter_groups` selects on. A constant rather than a literal at
#: two call sites, because the selection is by *name* and a rename that moved only
#: one of them would put the head at the policy's learning rate silently.
LOG_Z_HEAD_PREFIX = "log_z_head"

#: Fewest objectives a preference can be over. With one the simplex is the single
#: point ``[1.0]``: the conditioning is a constant, ``Z_θ(ω)`` is a scalar with
#: extra steps, and the arm is an unconditioned GFlowNet under a name that claims
#: a trade-off nothing varies.
MIN_OBJECTIVES = 2


class PreferenceConditionedPolicy(SequencePolicy):
    """A policy conditioned on the trade-off ``ω`` it is being asked to serve.

    See the module docstring for how ``ω`` reaches the trunk, why it is
    concatenated once rather than tiled, and why the partition function is a head
    rather than the inherited scalar.

    Args:
        n_objectives: How many objectives the preference trades off. Must be at
            least `MIN_OBJECTIVES`.
        n_bins: Bins per objective in the thermometer encoding. Must be at least
            [MIN_BINS][evogfn.models.conditioning.MIN_BINS].
        n_tokens: Alphabet size.
        sequence_length: Number of positions.
        n_actions: Size of the action space, including any stop action.
        embedding_dim: Width of the per-position embedding.
        hidden_dim: Width of the trunk.
        n_layers: Number of hidden layers in the trunk.
        log_z_hidden_dim: Width of the hidden layer in ``Z_θ(ω)``.
        learn_backward: Whether to learn ``P_B``.
        learn_flow: Whether to estimate a state flow.
        seed: Seeds every parameter, the conditional partition function included.

    Raises:
        ValueError: If any size is not positive, if fewer than
            `MIN_OBJECTIVES` objectives are given, or if ``n_bins`` is below
            `MIN_BINS`.
    """

    #: Declared so the buffers registered below are typed rather than reached
    #: through ``nn.Module``'s catch-all attribute lookup.
    _preference: torch.Tensor
    _preference_encoded: torch.Tensor

    #: ``Z_θ(ω)``. Named to match `LOG_Z_HEAD_PREFIX`, which is what
    #: `conditional_parameter_groups` selects on; the two are kept in step by a
    #: test rather than by convention, because the selection is by name and a
    #: rename that moved only one of them is silent.
    log_z_head: nn.Sequential

    def __init__(  # noqa: PLR0913 - mirrors the policy it extends
        self,
        *,
        n_objectives: int,
        n_bins: int = DEFAULT_PREFERENCE_BINS,
        n_tokens: int,
        sequence_length: int,
        n_actions: int,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        n_layers: int = 2,
        log_z_hidden_dim: int = DEFAULT_LOG_Z_HIDDEN_DIM,
        learn_backward: bool = False,
        learn_flow: bool = False,
        seed: int | None = None,
    ) -> None:
        """Size the trunk from the conditioning, then bind the neutral preference."""
        # Set *before* `super().__init__()`, and this is the trap the seam's own
        # docstring warns about: `SequencePolicy.__init__` calls `_build_layers`,
        # which calls `self._trunk_input_dim(...)` with only the two arguments it
        # knows about. A subclass needing anything else at that moment must have
        # it already. Getting this wrong is an AttributeError at construction,
        # which is the good failure; the bad one is sizing the trunk from one
        # expression and feeding it from another.
        if n_objectives < MIN_OBJECTIVES:
            raise ValueError(
                f"a preference-conditioned policy needs at least {MIN_OBJECTIVES} objectives, "
                f"got {n_objectives}; over one objective the simplex is the single point [1.0], "
                f"so the conditioning is a constant and this is an unconditioned policy"
            )
        if n_bins < MIN_BINS:
            raise ValueError(
                f"n_bins must be at least {MIN_BINS}, got {n_bins}; a single bin encodes every "
                f"preference identically and conditions the policy on nothing"
            )
        if log_z_hidden_dim < 1:
            raise ValueError(f"log_z_hidden_dim must be at least 1, got {log_z_hidden_dim}")
        self._n_objectives = n_objectives
        self._n_bins = n_bins
        self._log_z_hidden_dim = log_z_hidden_dim

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
        # Buffers rather than plain attributes so the preference follows the
        # module to another device and appears in the state dict: a policy
        # restored without it would condition on whatever it was built with.
        #
        # The neutral trade-off, which is what `preference_vectors(k, 1)` returns.
        # Anything else would be this class claiming which objective matters.
        neutral = np.full(n_objectives, 1.0 / n_objectives, dtype=np.float64)
        self.register_buffer("_preference", torch.as_tensor(neutral, dtype=torch.float32))
        self.register_buffer(
            "_preference_encoded",
            torch.as_tensor(self._encode(neutral), dtype=torch.float32),
        )

    def _build_layers(  # noqa: PLR0913 - mirrors the base class it forwards to
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
        """Build the trunk and heads, then the conditional partition function.

        Called from inside
        [SequencePolicy][evogfn.models.policy.SequencePolicy]'s seeded
        ``fork_rng`` block, which is the whole reason the head is created here
        rather than after ``super().__init__()`` returns: built outside it, it
        would draw from global torch state and two policies at the same seed
        would differ in their ``Z_θ(ω)`` alone. In a paired benchmark that is the
        kind of irreproducibility that surfaces as unexplained run-to-run
        variance rather than as an error.

        Args:
            n_tokens: Alphabet size.
            sequence_length: Number of positions.
            n_actions: Size of the action space.
            embedding_dim: Width of the per-position embedding.
            hidden_dim: Width of the trunk.
            n_layers: Hidden layers in the trunk.
            learn_backward: Whether to learn ``P_B``.
            learn_flow: Whether to estimate a state flow.
        """
        super()._build_layers(
            n_tokens=n_tokens,
            sequence_length=sequence_length,
            n_actions=n_actions,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            learn_backward=learn_backward,
            learn_flow=learn_flow,
        )
        # A genuine head, unlike `log_z`: the partition function depends on the
        # preference. Two layers rather than one because `log Z(ω)` is a
        # log-sum-exp of the reward over the reachable set and is not linear in
        # the thermometer coordinates.
        width = encoding_dim(self._n_objectives, n_bins=self._n_bins)
        self.log_z_head = nn.Sequential(
            nn.Linear(width, self._log_z_hidden_dim),
            nn.ReLU(),
            nn.Linear(self._log_z_hidden_dim, 1),
        )

    @property
    def n_objectives(self) -> int:
        """How many objectives the preference trades off."""
        return self._n_objectives

    @property
    def n_bins(self) -> int:
        """Bins per objective in the thermometer encoding."""
        return self._n_bins

    @property
    def preference(self) -> npt.NDArray[np.float64]:
        """The trade-off currently being conditioned on, as a copy."""
        return np.asarray(self._preference.detach().cpu().numpy(), dtype=np.float64)

    def set_preference(self, preference: npt.ArrayLike) -> None:
        """Point the policy at a new trade-off, keeping every weight.

        The weights are preference-free by construction -- ``ω`` enters as input,
        not as a parameter -- so moving it is exactly the operation this class
        exists to make possible, and it is what the amortisation claim rests on:
        a new trade-off at inference costs a forward pass rather than a
        retraining run.

        Args:
            preference: An ``(n_objectives,)`` vector, non-negative and summing
                to one.

        Raises:
            ValueError: If the preference is not a single vector of this policy's
                width on the simplex. Refused rather than renormalised: an
                unnormalised preference encodes to a conditioning vector the
                policy was never trained on, and a preference of the wrong width
                encodes to a vector the trunk was not sized for -- which is a
                shape error one forward pass later rather than here.
        """
        weights = np.asarray(preference, dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != self._n_objectives:
            raise ValueError(
                f"expected one preference over {self._n_objectives} objectives, got shape "
                f"{weights.shape}"
            )
        encoded = self._encode(weights)
        device = self._preference.device
        self._preference = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self._preference_encoded = torch.as_tensor(encoded, dtype=torch.float32, device=device)

    def _encode(self, preference: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Thermometer-encode a preference, raising if it is off the simplex."""
        return preference_encoding(preference, n_bins=self._n_bins)

    def conditional_log_z(self) -> torch.Tensor:
        """``log Z_θ(ω)`` at the preference currently held.

        Returns:
            A scalar tensor carrying gradients to the head's parameters.
        """
        value: torch.Tensor = self.log_z_head(self._preference_encoded)
        return value.squeeze(-1)

    def _trunk_input_dim(self, *, sequence_length: int, embedding_dim: int) -> int:
        """Width of the trunk input: the flattened state, plus the encoded preference.

        Args:
            sequence_length: Number of positions.
            embedding_dim: Width of the per-position embedding.

        Returns:
            ``sequence_length * embedding_dim + encoding_dim(n_objectives, n_bins)``.
        """
        return sequence_length * embedding_dim + encoding_dim(
            self._n_objectives, n_bins=self._n_bins
        )

    def _trunk_input(self, sequences: torch.Tensor) -> torch.Tensor:
        """Concatenate the state with the encoded preference, once per design.

        Args:
            sequences: An ``(n, sequence_length)`` integer tensor of states.

        Returns:
            An ``(n, _trunk_input_dim)`` float tensor.
        """
        embedded: torch.Tensor = self.embedding(sequences)
        flat = embedded.flatten(start_dim=1)
        conditioning = self._preference_encoded.to(device=flat.device, dtype=flat.dtype).expand(
            flat.shape[0], -1
        )
        return torch.cat([flat, conditioning], dim=-1)


def conditional_parameter_groups(
    policy: PreferenceConditionedPolicy,
    *,
    learning_rate: float = 1e-3,
    log_z_multiplier: float = 10.0,
) -> list[dict[str, Any]]:
    """Optimiser groups giving ``Z_θ(ω)`` its own learning rate.

    Written here rather than as a branch in
    [parameter_groups][evogfn.algorithms.gflownet.objectives.parameter_groups]
    for the reason the module docstring gives: that module is in both dependency
    closures. The consequence is a real one and is stated so it is not
    rediscovered -- ``parameter_groups`` selects the elevated group by the exact
    name ``log_z``, so a head named anything else lands in the policy group at the
    policy's rate. That is not an error anything raises: it is a partition
    function converging an order of magnitude too slowly, which reads as a policy
    that has not finished training.

    The elevated rate is kept, and the argument for it is weaker here than for
    one scalar. Malkin et al.'s reasoning is that ``log Z`` is a single parameter
    that must travel to ``log Σ R(x)``; this head is many parameters and only its
    *output* has to travel that far. Keep the multiplier, and record it as a
    hyperparameter nobody has screened.

    Two parameters are deliberately in no group at all. The inherited scalar
    ``log_z`` is one of them: it exists because the base class builds it, this
    policy's objective never reads it, and leaving it in a group would let
    momentum or weight decay drift it into a plausible-looking number that
    trained against nothing.

    Args:
        policy: The policy whose parameters to group.
        learning_rate: Rate for the policy parameters.
        log_z_multiplier: How much faster ``Z_θ(ω)`` should learn.

    Returns:
        Groups suitable for a torch optimiser.

    Raises:
        ValueError: If either rate is not positive, or if the policy carries no
            conditional partition function to elevate -- which would mean the
            groups silently trained the same thing the base class does.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if log_z_multiplier <= 0:
        raise ValueError(f"log_z_multiplier must be positive, got {log_z_multiplier}")

    head = [
        parameter
        for name, parameter in policy.named_parameters()
        if name.startswith(f"{LOG_Z_HEAD_PREFIX}.")
    ]
    if not head:
        raise ValueError(
            f"{type(policy).__name__} has no {LOG_Z_HEAD_PREFIX}; a preference-conditioned "
            f"policy without one measures balance against a scalar log Z fitted across every "
            f"preference, which biases it toward whichever trade-off that scalar suited"
        )
    rest = [
        parameter
        for name, parameter in policy.named_parameters()
        if name != "log_z" and not name.startswith(f"{LOG_Z_HEAD_PREFIX}.")
    ]
    return [
        {"params": rest, "lr": learning_rate},
        {"params": head, "lr": learning_rate * log_z_multiplier},
    ]
