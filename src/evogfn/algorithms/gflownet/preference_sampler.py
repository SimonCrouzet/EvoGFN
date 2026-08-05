r"""MOGFN-PC behind the same sampler interface as everything else in the suite.

One policy, conditioned on the trade-off. Trained once per round over the whole
simplex, then queried at each point of an evaluation grid, so a front is covered
by a single model rather than by one model per preference. That is Jain et al.'s
amortisation claim, and this class is what makes it runnable at plate scale.

Where ``ω`` enters, and why no campaign edit was needed
--------------------------------------------------------

The seam is [observe][evogfn.algorithms.base.Sampler.observe]. A campaign calls
it with the landscape's raw ``(n, n_objectives)`` matrix, *before* any
scalarisation -- the reduction to one number happens later and separately, when
the campaign fits its own surrogate. A sampler that fits a **multi-output**
surrogate at that call, and trains against a scalarised reward over a proxy
wrapping it, has applied the preference at reward time with ``loop/campaign.py``
untouched.

That is the structural difference from every other arm here. F3's reduction --
"at a fixed preference a scalarised arm faces exactly the single-objective
problem" -- rests on the campaign's proxy having no objective vector left in it.
This one carries its own, so it is genuinely outside the reduction. It is also
the reason the reduction still holds for the arms it was written about: nothing
here changes what a scalarised hill-climber is.

How the plate is spent, and the collapse this is built to avoid
----------------------------------------------------------------

The easy half is training. The half that is easy to get subtly wrong is
*evaluation*, and getting it wrong destroys the arm's advantage at the last step
with every column still looking healthy.

If this sampler proposed a preference-diverse pool and the campaign then screened
it with a [ScalarizedAcquisition][evogfn.acquisition.rules.ScalarizedAcquisition]
at one uniform preference and took the top ``k``, the whole pool would collapse
onto one trade-off. The indicators would come back flat and would read as
"conditioning does not help", which is a claim about the method rather than about
the wiring.

So [propose][evogfn.algorithms.gflownet.preference_sampler.PreferenceConditionedSampler.propose]
does the screening itself, per preference:

1. take the evaluation grid -- deliberately the same
   [preference_vectors][evogfn.benchmark.multi_objective.preference_vectors] the
   ``gfn-tb-pref{N}`` ablation is trained at, so both rows of the comparison are
   pointed at the same trade-offs;
2. sample a slice at each ``ω``;
3. rank each slice under **its own** ``ω``, using this sampler's surrogate;
4. return the slices interleaved round-robin, so that any prefix holds within one
   of an equal share from each preference and is locally best-first within each.

The campaign is then run with ``surrogate=None`` and takes the prefix, which is
what it already does for a sampler that ranks its own output. What that costs is
two columns: ``surrogate_correlation`` is ``nan`` for this arm and ``screened``
equals the pool. That is honest -- there is no campaign-level surrogate -- but it
is a column every other arm populates, so it belongs in the report rather than
being read as a defect. The alternative, a scalarising adapter passed as the
campaign's surrogate, would restore those columns *and* reintroduce the collapse
above, because the campaign would then re-rank the pool under one preference.

**The balance is exact in what this sampler returns and approximate in what gets
measured**, and the gap is the campaign's cross-round memory rather than
anything here. A round drops designs measured earlier and asks again, so the pool
the plate is taken from is one or more interleaved blocks with an arbitrary
subset removed. A preference whose slice has already been measured contributes
less to that plate than its share. The effect grows with saturation -- these
tasks are saturated by design -- so on a long campaign the *measured* plate can
drift away from the *proposed* balance. It is bounded by the pool being twenty
times the plate, it is visible in
[last_preference_index][evogfn.algorithms.gflownet.preference_sampler.PreferenceConditionedSampler.last_preference_index]
against the round's batch, and it is a caveat rather than a bug: charging a
method for re-proposing what it has already measured is the protocol every arm
here runs under.

What is stateful, and what that costs
--------------------------------------

The preference is held on the policy, so it can go stale, exactly as an anchor
can. This class owns the invariant: ``ω`` is set immediately before every
sampling call and every loss computation and never between them. It is fragile,
and it is fragile because
[Trajectories][evogfn.algorithms.gflownet.sampling.Trajectories] cannot be given
an ``ω`` field without invalidating every stored record. Recorded as a cost.

Unlike [GFlowNetSampler][evogfn.algorithms.gflownet.sampler.GFlowNetSampler],
this sampler *holds measurements*: they are what its own surrogate is fitted to.
They are accumulated across rounds, because
[Surrogate.fit][evogfn.surrogate.base.Surrogate.fit] is specified as taking the
full dataset while the campaign hands ``observe`` one round at a time -- and they
are carried across a re-anchoring, because a model refitted on the last round
alone is a different model from the one the ledger records.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.gflownet.preference_conditioned import (
    ConditionalTrajectoryBalance,
    _check_objective,
    train_preference_conditioned,
)
from evogfn.algorithms.gflownet.sampling import sample_trajectories
from evogfn.algorithms.gflownet.training import TrainingConfig

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.algorithms.gflownet.objectives import GFlowNetObjective
    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.base import SequenceEnvironment
    from evogfn.env.mutation import MutationEnvironment
    from evogfn.models.preference_policy import PreferenceConditionedPolicy
    from evogfn.rewards.scalarization import ScalarizedReward
    from evogfn.surrogate.multi_output import MultiObjectiveProxy, MultiOutputEnsemble

# An objective matrix is two-dimensional by definition.
_MATRIX_NDIM = 2


class PreferenceConditionedSampler(Sampler):
    """One preference-conditioned policy, queried across an evaluation grid.

    See the module docstring for where ``ω`` enters, why the plate is interleaved
    rather than screened by the campaign, and what holding measurements costs.

    Args:
        env: The construction graph. Feasibility lives here, as masks.
        policy: The policy to train, modified in place across rounds. Its
            preference is set by this sampler and should not be set from outside.
        surrogate: The multi-output model this sampler fits at ``observe`` and
            ranks with. Never the campaign's -- that one is fitted to an
            already-scalarised target.
        proxy: The landscape view of ``surrogate``, which training optimises
            against. Held by reference, so a refit is seen without anyone handing
            over a new one.
        preferences: The ``(k, n_objectives)`` evaluation grid. What the arm is
            *queried* at; training draws its own preferences from the whole
            simplex.
        reward: The template the per-step reward is rebuilt from -- it carries
            the scalarisation and ``β``. Its own preference is unused.
        config: Training settings applied on every retrain.
        objective: How balance violation is measured. Defaults to
            [ConditionalTrajectoryBalance][evogfn.algorithms.gflownet.preference_conditioned.ConditionalTrajectoryBalance].
        alpha: Dirichlet concentration for the training draw.
        seed: Seeds proposal sampling, independently of training.

    Raises:
        ValueError: If the grid does not match the policy's objectives, or if the
            objective cannot be run by the preference-conditioned loop. Both are
            refused here rather than at the first gradient step, which is a round
            of oracle calls later.
    """

    def __init__(  # noqa: PLR0913 - the sampler is defined by its parts
        self,
        env: SequenceEnvironment,
        policy: PreferenceConditionedPolicy,
        *,
        surrogate: MultiOutputEnsemble,
        proxy: MultiObjectiveProxy,
        preferences: npt.ArrayLike,
        reward: ScalarizedReward,
        config: TrainingConfig | None = None,
        objective: GFlowNetObjective | None = None,
        alpha: npt.ArrayLike = 1.0,
        seed: int = 0,
    ) -> None:
        """Store the training setup without running it."""
        super().__init__()
        grid = np.asarray(preferences, dtype=np.float64)
        if grid.ndim != _MATRIX_NDIM or grid.shape[0] < 1:
            raise ValueError(
                f"expected an evaluation grid of shape (k, n_objectives), got {grid.shape}"
            )
        if grid.shape[1] != policy.n_objectives:
            raise ValueError(
                f"the evaluation grid covers {grid.shape[1]} objectives but the policy is "
                f"conditioned on {policy.n_objectives}"
            )
        if surrogate.n_objectives != policy.n_objectives:
            raise ValueError(
                f"the surrogate predicts {surrogate.n_objectives} objectives but the policy is "
                f"conditioned on {policy.n_objectives}"
            )
        _check_objective(objective if objective is not None else ConditionalTrajectoryBalance())

        self._env = env
        self._policy = policy
        self._surrogate = surrogate
        self._proxy = proxy
        self._preferences = grid
        self._reward = reward
        self._config = config or TrainingConfig()
        self._objective = objective
        self._alpha = alpha
        self._seed = seed
        self._generator: torch.Generator | None = None
        self._rounds_trained = 0
        self._proxy_calls = 0
        self._sequences: list[Tokens] = []
        self._values: list[Fitness] = []
        self._last_origin: npt.NDArray[np.intp] | None = None

    @property
    def name(self) -> str:
        """Short label naming the method, the objective and the grid size."""
        objective = (
            type(self._objective).__name__ if self._objective else "ConditionalTrajectoryBalance"
        )
        return f"MOGFN-PC ({objective}) x{self._preferences.shape[0]} preferences"

    @property
    def policy(self) -> PreferenceConditionedPolicy:
        """The one policy this arm trains, for inspection."""
        return self._policy

    @property
    def surrogate(self) -> MultiOutputEnsemble:
        """The one multi-output model this arm fits, for inspection."""
        return self._surrogate

    @property
    def reward(self) -> ScalarizedReward:
        """The reward template, which carries the scalarisation and ``β``."""
        return self._reward

    @property
    def preferences(self) -> npt.NDArray[np.float64]:
        """The evaluation grid, as a copy so a caller cannot move it."""
        return self._preferences.copy()

    @property
    def rounds_trained(self) -> int:
        """How many times the policy has been retrained."""
        return self._rounds_trained

    @property
    def proxy_calls(self) -> int:
        """Reward evaluations spent on the proxy.

        Free in budget terms and expensive in compute terms, and here it carries
        half of the claim under test: one conditioned policy trained once should
        cost roughly ``1/N`` of ``N`` replicated policies at the same oracle
        budget. If it does not, the amortisation claim is half-refuted before any
        indicator is read.

        Ranking predictions are deliberately **not** counted, matching
        [GFlowNetSampler][evogfn.algorithms.gflownet.sampler.GFlowNetSampler]:
        the ablation's ranking happens inside its campaign and is not counted
        either, and a column that counted one and not the other would make the
        comparison a comparison of accounting.
        """
        return self._proxy_calls

    @property
    def observed_sequences(self) -> Tokens:
        """Every design this sampler has been told the value of."""
        if not self._sequences:
            return np.zeros((0, self._env.sequence_length), dtype=np.int64)
        return np.concatenate(self._sequences)

    @property
    def observed_values(self) -> Fitness:
        """Their objective values, as vectors rather than as a scalarisation."""
        if not self._values:
            return np.zeros((0, self._policy.n_objectives), dtype=np.float64)
        return np.concatenate(self._values)

    @property
    def last_preference_index(self) -> npt.NDArray[np.intp] | None:
        """Which grid entry produced each design of the last proposal, or ``None``.

        Exposed because the balance of the plate across preferences cannot be
        recovered from the designs themselves and cannot be inferred from any
        indicator, yet it is the property the whole evaluation half turns on. A
        prefix that is all one slice is the collapse this class is built to
        prevent, and this is what makes it visible.
        """
        return None if self._last_origin is None else self._last_origin.copy()

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Accumulate the measurements and refit the multi-output surrogate.

        This is the seam the whole arm hangs on: the campaign calls it with the
        landscape's raw objective matrix, before anything has scalarised it.

        Args:
            sequences: The candidates that were measured.
            values: An ``(n, n_objectives)`` array of their objective values.

        Raises:
            ValueError: If the values are not an objective matrix of the policy's
                width. An ``(n, 1)`` here means something upstream reduced them,
                and the model would then be fitted to a trade-off this arm exists
                to keep out of its surrogate.
        """
        matrix = np.asarray(values, dtype=np.float64)
        expected = self._policy.n_objectives
        if matrix.ndim != _MATRIX_NDIM or matrix.shape[1] != expected:
            raise ValueError(
                f"expected measurements of shape (n, {expected}) covering {expected} objectives, "
                f"got {matrix.shape}; a reduced column here would put the preference upstream of "
                f"the reward, which is what this arm exists not to do"
            )
        self._sequences.append(np.asarray(sequences))
        self._values.append(matrix)

        accumulated = self.observed_values
        # A round can legitimately produce nothing finite on some objective -- on
        # a masked landscape an infeasible design is -inf everywhere. There is
        # then nothing to fit, and that is a result about the method rather than
        # an error: the campaign proceeds unassisted and the ledger records it.
        # Raising here would turn the finding into a traceback and lose the rest
        # of a campaign whose budget is already partly spent.
        if np.isfinite(accumulated).any(axis=0).all():
            self._surrogate.fit(self.observed_sequences, accumulated)

    def propose(self, n: int) -> Tokens:
        """Retrain against the current proxy, then spend ``n`` across the grid.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array, the grid's slices interleaved
            round-robin so every prefix is balanced across preferences.
        """
        if self._proxy.is_ready:
            self._retrain()

        k = self._preferences.shape[0]
        # Ceiling, then truncate: rounding down would leave the pool short of
        # what the campaign asked for, and the campaign would spend proposal
        # calls papering over an arithmetic decision made here.
        per_slice = -(-n // k)
        slices = [
            self.designs_at(preference, per_slice, generator=self._sampling_generator())
            for preference in self._preferences
        ]
        pool, origin = _interleave(slices)
        self._last_origin = origin[:n]
        self._count(n)
        return np.asarray(pool[:n])

    def designs_at(
        self,
        preference: npt.ArrayLike,
        n: int,
        *,
        generator: torch.Generator | None = None,
    ) -> Tokens:
        """Sample ``n`` designs at one trade-off, best-first under that trade-off.

        The amortisation claim at its smallest testable size: a new preference
        costs a forward pass rather than a retraining run. It is public because
        that is the operation the claim *is*, not merely an internal step of
        `propose`.

        Args:
            preference: The ``(n_objectives,)`` trade-off to condition on and to
                rank under.
            n: How many designs to sample.
            generator: Torch generator for the rollout. ``None`` uses this
                sampler's own stream.

        Returns:
            An ``(n, sequence_length)`` array, ordered best-first under the
            surrogate's prediction scalarised at ``preference``. Unordered while
            the surrogate is unfitted, which is round zero: ranking against an
            initialisation would order the plate by noise.

        Raises:
            ValueError: If the preference is not on the simplex or does not cover
                this policy's objectives.
        """
        self._policy.set_preference(preference)
        trajectories = sample_trajectories(
            self._env,
            self._policy,
            n,
            epsilon=0.0,
            generator=generator if generator is not None else self._sampling_generator(),
        )
        designs = np.asarray(trajectories.terminal)
        if not self._surrogate.is_fitted:
            return designs
        mean, _ = self._surrogate.predict(designs)
        scored = self._reward.scalarization.scalarize(mean, self._policy.preference)
        # Descending, and stable, so two designs the model cannot separate keep
        # the order the policy produced them in rather than an arbitrary one.
        order = np.argsort(-scored, kind="stable")
        return designs[order]

    def reanchored(self, env: MutationEnvironment) -> PreferenceConditionedSampler:
        """Carry the policy, the model, the measurements and the accounting to ``env``.

        The policy's weights transfer for the reason
        [GFlowNetSampler.reanchored][evogfn.algorithms.gflownet.sampler.GFlowNetSampler.reanchored]
        gives: the action layout and the network's input describe the space
        rather than the parent, and what is anchored is only the masks, which are
        read from ``env`` on every step. The preference is not anchored at all --
        it is a direction in objective space, not in sequence space.

        ``Z_θ(ω)`` is carried and is stale by construction, exactly as the scalar
        ``log Z`` is in the unconditioned sampler: it estimates the partition
        function over the reachable set, and the reachable set is precisely what
        moved. It is kept for the same three reasons -- it enters the loss and
        never the forward distribution, it carries the elevated learning rate
        that re-converges it inside the next retrain, and the reward *scale*
        largely transfers across a move.

        **The measurements are carried, and this is where this sampler differs
        from its unconditioned sibling.** That one holds none -- everything it
        learns arrives through a proxy the campaign refits. This one fits its own
        model, from data the campaign hands it one round at a time, so restarting
        the accumulation at each anchor would refit the surrogate on a single
        round while the ledger recorded four.

        Args:
            env: The re-anchored environment. Must describe the same space.

        Returns:
            A sampler over ``env``, sharing this one's policy, surrogate, proxy
            and random stream and continuing its counts.

        Raises:
            ValueError: If ``env`` changes the sequence length or the size of the
                action space, which would leave the policy emitting logits for
                actions that no longer exist.
        """
        if env.sequence_length != self._env.sequence_length or env.n_actions != self._env.n_actions:
            raise ValueError(
                f"cannot carry a policy over {self._env.n_actions} actions on sequences of "
                f"length {self._env.sequence_length} into an environment of {env.n_actions} "
                f"actions on length {env.sequence_length}; the anchor may move but the "
                f"sequence length and alphabet may not"
            )
        moved = PreferenceConditionedSampler(
            env,
            self._policy,
            surrogate=self._surrogate,
            proxy=self._proxy,
            preferences=self._preferences,
            reward=self._reward,
            config=self._config,
            objective=self._objective,
            alpha=self._alpha,
            seed=self._seed,
        )
        moved._generator = self._sampling_generator()
        moved._rounds_trained = self._rounds_trained
        moved._proxy_calls = self._proxy_calls
        moved._proposals_made = self._proposals_made
        moved._sequences = list(self._sequences)
        moved._values = list(self._values)
        return moved

    def _retrain(self) -> None:
        """One round of preference-conditioned training against the current proxy."""
        # A distinct seed per round, or every round replays the same trajectories
        # *and* the same preference schedule, and the later rounds teach nothing.
        config = replace(self._config, seed=self._config.seed + self._rounds_trained)
        result = train_preference_conditioned(
            self._env,
            self._policy,
            self._proxy,
            self._reward,
            config,
            alpha=self._alpha,
            objective=self._objective,
        )
        self._proxy_calls += result.oracle_calls
        self._rounds_trained += 1

    def _sampling_generator(self) -> torch.Generator:
        """One generator across rounds, so proposals do not repeat themselves."""
        if self._generator is None:
            self._generator = torch.Generator().manual_seed(self._seed)
        return self._generator

    def __repr__(self) -> str:
        """Name the sampler, its grid and how much training it has had."""
        return (
            f"PreferenceConditionedSampler(preferences={self._preferences.shape[0]}, "
            f"rounds_trained={self._rounds_trained})"
        )


def _interleave(slices: list[Tokens]) -> tuple[Tokens, npt.NDArray[np.intp]]:
    """Round-robin the per-preference slices, keeping each one's order.

    The property this exists for: a prefix of length ``m`` holds between
    ``floor(m/k)`` and ``ceil(m/k)`` designs from each of the ``k`` preferences,
    and within each preference they are in the order that preference ranked them.
    A campaign takes a prefix, so concatenating the slices instead would hand the
    whole plate to the first preference and the pool's diversity would be spent
    on designs nobody measures.

    Args:
        slices: One array of designs per preference, all the same length.

    Returns:
        The interleaved designs, and which slice each of them came from.
    """
    stacked = np.stack(slices, axis=1)
    designs = stacked.reshape(-1, stacked.shape[-1])
    origin = np.tile(np.arange(len(slices), dtype=np.intp), stacked.shape[0])
    return designs, origin
