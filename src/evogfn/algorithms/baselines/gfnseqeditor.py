r"""GFNSeqEditor as a benchmark arm.

A faithful port of GFNSeqEditor \\citep{ghari2024gfnseqeditor}, the prior
GFlowNet-based sequence editor for biological sequence design, run at the
authors' reported hyperparameters. GFNSeqEditor trains a de-novo GFlowNet on the
property reward, then edits a seed sequence position by position: at each
position it decides whether keeping the seed token is *sub-optimal* under the
flow, and if so resamples the token from the flow-proportional policy.

Two facts make the port exact on our stack. First, GFNSeqEditor trains with
trajectory balance, so its flow satisfies ``F(s + a) = F(s) P_F(a | s)`` and the
ratio its gate uses, ``F(x + a) / max_a F(x + a)``, equals
``P_F(a | x) / max_a P_F(a | x)`` -- the parent flow cancels, so their eqs. (4)
and (6) are computed from the trained forward policy with no separate flow head.
Second, their construction is de-novo, so the flow trains in a
[DeNovoEnvironment][evogfn.env.denovo.DeNovoEnvironment] rather than the edit
lattice, which is why this is their method and not our mask handed a new head.

The arm carries no feasibility mask: GFNSeqEditor's setting has no hard
feasibility predicate, so on a constrained task it optimises the property and
leaves legality to chance. That is the comparison -- their editor ranks edits by
property, feasibility by construction gates them by legality.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.gflownet.training import TrainingConfig, train_trajectory_balance
from evogfn.env.base import State

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Tokens
    from evogfn.env.denovo import DeNovoEnvironment
    from evogfn.models.policy import SequencePolicy
    from evogfn.rewards.base import Reward
    from evogfn.surrogate.proxy import ProxyLandscape


class GFNSeqEditorSampler(Sampler):
    """Train a de-novo GFlowNet on the proxy, then edit the anchor by its flow.

    Args:
        env: The de-novo construction graph the flow is trained in.
        policy: The forward policy to train, modified in place across rounds.
        anchor: The seed sequence GFNSeqEditor edits, of shape ``(length,)``.
        proxy: The surrogate the policy is trained against.
        reward: Transforms proxy values into log rewards.
        delta: Sub-optimality threshold in eq. (4); larger flags more positions.
        lam: Edit-count regularizer in eq. (6); larger keeps more of the seed.
        sigma: Standard deviation of the exploration noise added in eq. (4).
        config: Training run settings.
        train: Whether to retrain each round.
        seed: Seed for the editing noise and any sampling.
    """

    def __init__(  # noqa: PLR0913 - the arm is defined by its parts
        self,
        env: DeNovoEnvironment,
        policy: SequencePolicy,
        anchor: Tokens,
        *,
        proxy: ProxyLandscape,
        reward: Reward,
        delta: float = 0.4,
        lam: float = 0.1,
        sigma: float = 0.001,
        config: TrainingConfig | None = None,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        """Store the training and editing setup without running it."""
        super().__init__()
        self._env = env
        self._policy = policy
        self._anchor = np.asarray(anchor).astype(np.int64).ravel()
        self._proxy = proxy
        self._reward = reward
        self._delta = float(delta)
        self._lam = float(lam)
        self._sigma = float(sigma)
        self._config = config or TrainingConfig()
        self._train = train
        self._seed = seed
        self._rounds_trained = 0
        self._proxy_calls = 0
        self._noise = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        """Short label naming the arm and its threshold."""
        return f"GFNSeqEditor (delta={self._delta:g})"

    @property
    def rounds_trained(self) -> int:
        """How many times the de-novo policy has been retrained."""
        return self._rounds_trained

    @property
    def proxy_calls(self) -> int:
        """Surrogate evaluations spent training the flow, a campaign total."""
        return self._proxy_calls

    def propose(self, n: int) -> Tokens:
        """Retrain the de-novo flow against the proxy, then edit the anchor ``n`` times.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, length)`` array of edited designs.
        """
        if self._train and self._proxy.is_ready:
            config = replace(self._config, seed=self._config.seed + self._rounds_trained)
            result = train_trajectory_balance(
                self._env, self._policy, self._proxy, self._reward, config
            )
            self._proxy_calls += result.oracle_calls
            self._rounds_trained += 1
        designs = self._edit(n)
        self._count(n)
        return designs

    def _forward_probs(
        self, prefix: npt.NDArray[np.int64], t: int, n: int
    ) -> npt.NDArray[np.float64]:
        """``P_F(a | x_{:t})`` over the ``v`` append tokens for a batch of prefixes.

        Args:
            prefix: The edited sequence so far, ``(n, length)``; positions ``t``
                onward are ignored.
            t: The current cursor position.
            n: Batch size.

        Returns:
            An ``(n, v)`` array of forward probabilities over the append tokens.
        """
        v = self._env.n_tokens - 1
        pad = self._env.n_tokens - 1
        seqs = np.full((n, self._env.sequence_length), pad, dtype=np.int32)
        if t > 0:
            seqs[:, :t] = prefix[:, :t]
        state = State(sequences=seqs, stopped=np.zeros(n, dtype=np.bool_))
        fmask = self._env.forward_mask(state)
        bmask = self._env.backward_mask(state)
        with torch.no_grad():
            log_forward, _ = self._policy.log_probs(
                torch.as_tensor(seqs, dtype=torch.long),
                torch.as_tensor(fmask),
                torch.as_tensor(bmask),
            )
        probs = torch.exp(log_forward).cpu().numpy()[:, :v]
        return np.asarray(probs, dtype=np.float64)

    def _edit(self, n: int) -> Tokens:
        """Run GFNSeqEditor's Algorithm 1 on the anchor, ``n`` times in parallel."""
        length = self._env.sequence_length
        edited = np.tile(self._anchor, (n, 1)).astype(np.int64)
        for t in range(length):
            probs = self._forward_probs(edited, t, n)  # (n, v)
            keep = int(self._anchor[t])
            prob_keep = probs[:, keep]
            prob_max = probs.max(axis=1)
            noise = self._noise.normal(0.0, self._sigma, size=n)
            # eq. (4): position is sub-optimal when keeping the seed token has
            # far less flow than the best action.
            sub_optimal = prob_keep < self._delta * prob_max + noise
            # eq. (6): a mixture of the flow-proportional policy and keeping the
            # seed token, so lambda trades property gain against edit count.
            policy = (1.0 - self._lam) * probs
            policy[:, keep] += self._lam
            policy /= policy.sum(axis=1, keepdims=True)
            cumulative = np.cumsum(policy, axis=1)
            draws = self._noise.random(n)[:, None]
            sampled = (draws < cumulative).argmax(axis=1)
            edited[:, t] = np.where(sub_optimal, sampled, keep)
        return edited.astype(np.int32)
