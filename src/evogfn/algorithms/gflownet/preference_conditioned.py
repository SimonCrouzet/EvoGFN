r"""Training a GFlowNet over the whole simplex, one preference per gradient step.

MOGFN-PC's training loop, and the objective that goes with it. The difference
from [train_trajectory_balance][evogfn.algorithms.gflownet.training.train_trajectory_balance]
is one thing repeated every step: draw ``ω``, condition the policy on it, rebuild
the reward at it. Everything else -- the annealed exploration, the optimiser
groups, the reporting -- is the same job and is written the same way.

This is a copy, and which one is canonical
-------------------------------------------

[train_trajectory_balance][evogfn.algorithms.gflownet.training.train_trajectory_balance]
is canonical. It takes one fixed reward and cannot be given a per-step one
without changing its signature, and it lives in a module inside the dependency
closure of both benchmark suites -- where the fingerprint is over raw file bytes,
so any edit invalidates every stored single-objective record. So the loop is
duplicated rather than parameterised, and the duplication will drift. When it
does, the single-objective loop is the one that is right, and this one should be
brought back to it rather than the reverse.

One ``ω`` per batch, not one per trajectory
--------------------------------------------

The repository says both things in different files. This loop draws **one per
gradient step**, and the reasons are internal to this codebase rather than read
off the paper, which is worth saying plainly because the paper is the authority
and nobody has checked it:

* It makes [ContrastiveBalance][evogfn.algorithms.gflownet.objectives.ContrastiveBalance]
  a legal alternative rather than a subtly wrong one. What cancels in a
  contrasted pair is ``-log Z(ω)``, and it only cancels when both members of the
  pair share their ``ω``. Under a per-trajectory draw that objective would run
  and would be measuring nothing.
* It lets the reward be a [ScalarizedReward][evogfn.rewards.scalarization.ScalarizedReward]
  rebuilt with ``with_preference``, which is the documented per-batch path. A
  per-trajectory draw needs an ``(n, k)`` preference matrix, which
  ``Scalarization.scalarize`` accepts but ``ScalarizedReward`` does not -- so it
  would mean reaching past the reward layer.
* The alignment between "the ``ω`` the policy was conditioned on" and "the ``ω``
  the reward used" becomes a single identity per step, which a test can check
  exactly. Per trajectory it is a row-order invariant, and
  [Trajectories][evogfn.algorithms.gflownet.sampling.Trajectories] has no field
  to carry ``ω`` in -- so the invariant would live nowhere and be checkable
  nowhere. Training a policy conditioned on ``ω₁`` against ``R(·|ω₂)`` is the
  worst bug available here: the loss converges, nothing raises, and the arm
  learns nothing about the trade-off it was asked for.

What it costs, and it is a real cost: at 300 steps of batch 64 a round sees 300
distinct preferences rather than 19,200, so each step's gradient is a
single-``ω`` estimate of an expectation over the simplex and the ``ω`` direction
is higher-variance than a per-trajectory scheme at the same number of
trajectories. If the front comes back thin, this is the second thing to suspect
after `n_bins`.

Which objectives this loop will run
------------------------------------

Two are refused, both at the top rather than at the first gradient step.

[TrajectoryBalance][evogfn.algorithms.gflownet.objectives.TrajectoryBalance]
reads ``policy.log_z``, the scalar this policy never trains. It would run to
completion against a constant partition function, and the residual would absorb
the variation in ``log Z(ω)`` by pushing the policy toward whichever trade-off
that constant suited. The symptom is a front collapsed to a point, which is
indistinguishable from the null the experiment is testing for.

The detailed-balance family asks for a reward at every *visited state*, which
this loop does not compute. That is a gap rather than a decision -- the scoring
helper it would need is private to the canonical loop, and copying it would
double the drift this module already carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from evogfn.algorithms.gflownet.objectives import GFlowNetObjective, balance_violation
from evogfn.algorithms.gflownet.sampling import sample_trajectories
from evogfn.models.conditioning import sample_preferences
from evogfn.models.preference_policy import (
    PreferenceConditionedPolicy,
    conditional_parameter_groups,
)
from evogfn.rewards.scalarization import ScalarizedReward
from evogfn.tracking.base import NoOpTracker

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.algorithms.gflownet.sampling import Trajectories
    from evogfn.algorithms.gflownet.training import TrainingConfig
    from evogfn.env.base import SequenceEnvironment
    from evogfn.landscapes.base import FitnessLandscape
    from evogfn.models.policy import SequencePolicy
    from evogfn.tracking.base import Tracker


class ConditionalTrajectoryBalance(GFlowNetObjective):
    r"""Trajectory balance measured against ``Z_θ(ω)`` rather than a scalar.

    Malkin et al.'s residual with Jain et al.'s conditional partition function:

    $$
    \left(\log Z_\theta(\omega) + \log P_F(\tau) - \log R(x|\omega)
    - \log P_B(\tau|x)\right)^2
    $$

    The preference is read off the *policy* rather than off the batch, because
    [Trajectories][evogfn.algorithms.gflownet.sampling.Trajectories] carries no
    field for it and adding one would invalidate every stored record. The
    invariant that makes that sound -- the policy is conditioned on the ``ω`` the
    rewards were computed at, and nothing changes it in between -- belongs to the
    training loop, which is where it is established and where it is tested.
    """

    def loss(
        self,
        trajectories: Trajectories,
        log_rewards: torch.Tensor,
        policy: SequencePolicy,
    ) -> torch.Tensor:
        """Mean squared deviation of the violation from ``-log Z_θ(ω)``.

        Args:
            trajectories: Completed trajectories, all sampled at one preference.
            log_rewards: An ``(n,)`` tensor of ``log R(x|ω)`` at that same
                preference.
            policy: Supplies ``Z_θ(ω)`` at the preference it currently holds.

        Returns:
            A scalar loss.

        Raises:
            TypeError: If the policy has no conditional partition function.
                Falling back to ``policy.log_z`` would be the failure this class
                exists to remove, and it would run to completion.
        """
        if not isinstance(policy, PreferenceConditionedPolicy):
            raise TypeError(
                f"conditional trajectory balance needs a preference-conditioned policy with a "
                f"Z_theta(omega) head, got {type(policy).__name__}; measuring against the "
                f"scalar log Z instead would fit one partition function across every preference "
                f"and bias the policy toward whichever trade-off it suited"
            )
        violation = policy.conditional_log_z() + balance_violation(trajectories, log_rewards)
        return violation.pow(2).mean()

    @property
    def uses_log_z(self) -> bool:
        """``False``, and it is a statement about the *scalar*.

        This objective does train a learned partition function; what it does not
        train is ``policy.log_z``, which is the only thing this flag causes a
        trainer to log. Reporting the base class's untrained zero under the name
        ``log_z`` would be worse than reporting nothing.
        """
        return False


@dataclass(slots=True)
class ConditionalTrainingResult:
    """What a completed preference-conditioned run produced.

    A separate type from
    [TrainingResult][evogfn.algorithms.gflownet.training.TrainingResult] because
    it carries one field that has no meaning there and that two of this module's
    failure modes are only visible through: the preferences actually trained at.
    A fixed ``ω`` and an unseeded draw both produce a plausible loss curve, and
    both turn the arm into something other than what its name says.

    Attributes:
        losses: Loss at every step.
        preferences: The ``ω`` drawn at every step, in order.
        final_log_z: ``log Z_θ(ω)`` at the **last** preference trained, which is
            one point of a function and not the run's partition function. Kept
            for parity with the canonical loop's reporting and not to be read as
            that loop's field is.
        objective: Name of the objective the run was trained under, so a stored
            or logged result says which balance condition produced it.
        oracle_calls: Landscape evaluations. In a campaign these land on the
            proxy rather than the assay, which is what makes them free in budget
            terms and expensive in compute terms.
    """

    losses: list[float] = field(default_factory=list)
    preferences: list[npt.NDArray[np.float64]] = field(default_factory=list)
    final_log_z: float = 0.0
    objective: str = ""
    oracle_calls: int = 0


def train_preference_conditioned(  # noqa: PLR0913 - the run is defined by its parts
    env: SequenceEnvironment,
    policy: PreferenceConditionedPolicy,
    landscape: FitnessLandscape,
    reward: ScalarizedReward,
    config: TrainingConfig,
    *,
    alpha: npt.ArrayLike = 1.0,
    objective: GFlowNetObjective | None = None,
    tracker: Tracker | None = None,
) -> ConditionalTrainingResult:
    """Train one policy to sample proportionally to ``R(x|ω)^β`` over the simplex.

    The whole simplex is drawn up front rather than one vector at a time, which
    is not a micro-optimisation: it makes the schedule a pure function of
    ``config.seed`` in a single call, so "same seed" means "same preferences" and
    the benchmark's paired comparisons pair what they claim to. Drawing per step
    from ``config.seed`` would give every step the *same* ``ω`` -- a fixed
    preference dressed as a conditioned one, which is the quietest way to turn
    this arm back into the one it is being compared against.

    Args:
        env: The construction graph.
        policy: The policy to train, modified in place. Its preference is set at
            every step and is left at the last one drawn.
        landscape: What terminal sequences are scored against -- in a campaign,
            a [MultiObjectiveProxy][evogfn.surrogate.multi_output.MultiObjectiveProxy]
            over the sampler's own surrogate, never the oracle.
        reward: The template the per-step reward is rebuilt from. Its own
            preference is never used: it supplies the scalarisation and the
            scalar reward -- ``β`` and the floor -- and
            [with_preference][evogfn.rewards.scalarization.ScalarizedReward.with_preference]
            supplies the rest.
        config: Run settings. ``seed`` fixes both the rollout stream and the
            preference schedule.
        alpha: Dirichlet concentration for the draw. One is uniform over the
            simplex; below one puts mass at the corners, so most batches ask for
            a near single-objective specialist and the extremes of the front get
            covered.
        objective: How balance violation is measured. Defaults to
            `ConditionalTrajectoryBalance`.
        tracker: Where to report. Defaults to discarding.

    Returns:
        The losses, the preferences trained at, and the proxy budget consumed.

    Raises:
        TypeError: If ``reward`` is not a
            [ScalarizedReward][evogfn.rewards.scalarization.ScalarizedReward].
            A scalar reward would ignore every preference drawn and the run would
            complete looking exactly like a successful one.
        ValueError: If the objective reads the scalar ``log Z``, or asks for a
            reward at every visited state. Both are refused before the first
            gradient step; see the module docstring for why each is wrong here
            rather than merely unsupported.
    """
    if not isinstance(reward, ScalarizedReward):
        raise TypeError(
            f"preference-conditioned training needs a ScalarizedReward to rebuild per step, "
            f"got {type(reward).__name__}; a scalar reward has nowhere to apply omega and "
            f"would train every step against the same target"
        )
    loss_fn = objective if objective is not None else ConditionalTrajectoryBalance()
    _check_objective(loss_fn)

    recorder = tracker if tracker is not None else NoOpTracker()
    optimiser = torch.optim.Adam(
        conditional_parameter_groups(
            policy,
            learning_rate=config.learning_rate,
            log_z_multiplier=config.log_z_multiplier,
        )
    )
    generator = torch.Generator().manual_seed(config.seed)
    # One call, so the schedule is a pure function of the seed. See the note in
    # the docstring on what per-step seeding would have done instead.
    schedule = sample_preferences(policy.n_objectives, config.steps, alpha=alpha, seed=config.seed)
    result = ConditionalTrainingResult(objective=type(loss_fn).__name__)

    for step in range(config.steps):
        fraction = step / max(config.steps - 1, 1)
        epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)

        # The invariant the whole arm rests on, and it lives in these two lines:
        # the policy is conditioned on `omega` and the reward is rebuilt at the
        # *same* `omega`, with nothing between them that could move either. A
        # mismatch here converges, raises nothing, and teaches the policy about a
        # trade-off nobody asked for.
        omega = schedule[step]
        policy.set_preference(omega)
        step_reward = reward.with_preference(omega)

        trajectories = sample_trajectories(
            env, policy, config.batch_size, epsilon=epsilon, generator=generator
        )
        values = landscape.evaluate(trajectories.terminal)
        result.oracle_calls += int(trajectories.terminal.shape[0])
        log_rewards = torch.as_tensor(step_reward.log_reward(values), dtype=torch.float32)

        loss = loss_fn.loss(trajectories, log_rewards, policy)
        optimiser.zero_grad()
        # torch ships stubs but leaves Tensor.backward unannotated.
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()

        loss_value = float(loss.detach().item())
        result.losses.append(loss_value)
        result.preferences.append(np.asarray(omega, dtype=np.float64))
        if step % config.log_every == 0 or step == config.steps - 1:
            recorder.log_metrics(
                {
                    "loss": loss_value,
                    # Named for what it is: one point of `Z_theta(omega)`, not a
                    # run-level scalar. Logging it as `log_z` would invite the
                    # reader to watch it converge, which it has no reason to do.
                    "log_z_at_omega": float(policy.conditional_log_z().detach().item()),
                    "epsilon": epsilon,
                    "mean_reward": float(np.exp(log_rewards.numpy()).mean()),
                    "oracle_calls": float(result.oracle_calls),
                },
                step=step,
            )

    result.final_log_z = float(policy.conditional_log_z().detach().item())
    return result


def _check_objective(objective: GFlowNetObjective) -> None:
    """Refuse an objective this loop would run wrongly or not at all.

    Args:
        objective: The objective the caller asked for.

    Raises:
        ValueError: If it measures against the scalar ``log Z``, or needs a
            reward at every visited state.
    """
    if objective.uses_log_z:
        raise ValueError(
            f"{type(objective).__name__} measures balance against the scalar log Z, which a "
            f"preference-conditioned policy never trains; one partition function fitted across "
            f"every preference makes the residual absorb the variation in log Z(omega) and "
            f"pushes the policy toward a single trade-off, which looks exactly like preference "
            f"conditioning failing"
        )
    if objective.needs_state_rewards:
        raise ValueError(
            f"{type(objective).__name__} needs a reward at every visited state and this loop "
            f"does not score intermediate states; the detailed-balance family is a known gap "
            f"here rather than a supported configuration"
        )
