"""Whether a learned flow recovers the support a local mask cannot reach.

The support study measures two masks exactly. A local mask admits a move when
the state it lands on is legal, and on a codebook predicate that reaches one
design out of eight hundred. A sound mask admits a move when a legal design
still lies ahead, and reaches all eight hundred. Between them sits the lookahead
family, which prices the difference: a mask seeing `k` moves ahead must search
``alphabet**k`` continuations per move, and at a separation of four nothing
below `k = 3` recovers anything at all.

A GFlowNet's flow is the third possibility. Where the reward is zero on illegal
designs, the flow into a state is the total reward of the legal designs
reachable from it, so a state from which nothing legal survives carries zero
flow -- and a policy trained to that reward has, implicitly, an estimate of the
completion question the local mask cannot answer. This script asks whether the
estimate is good enough to be useful, on an instance where the true answer is
enumerable and the comparison is therefore exact rather than inferred.

The environment used here defers feasibility to the terminal, so the policy is
**not** locally masked: it must learn which moves lead somewhere legal rather
than being told. That is the setting Silva et al. identify as terminally
unrestricted and therefore not guaranteed to sample the target, which is why
both halves of the trade are reported -- what share of the legal set the sampler
covers, and what share of what it emits is legal at all.

Nothing here touches the fingerprinted closure, so no stored record is staled.

Usage:
    uv run python experiments/support_flowmask.py --task support-sep4
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from evogfn.algorithms.gflownet.flow_objectives import (
    ForwardLookingDetailedBalance,
    SubTrajectoryBalance,
)
from evogfn.algorithms.gflownet.objectives import TrajectoryBalance
from evogfn.algorithms.gflownet.sampling import sample_trajectories
from evogfn.algorithms.gflownet.training import TrainingConfig, train_trajectory_balance
from evogfn.benchmark.determinism import configure_determinism
from evogfn.benchmark.methods import _environment
from evogfn.benchmark.suite import support_tasks
from evogfn.landscapes.base import FitnessLandscape
from evogfn.models.policy import SequencePolicy
from evogfn.rewards.base import TemperedReward

#: Designs drawn from the trained policy when measuring coverage.
DEFAULT_DRAWS = 20_000

#: Gradient steps. The instance is small; this is well past where the loss flattens.
DEFAULT_STEPS = 3_000


class FeasibilityLandscape(FitnessLandscape):
    """Scores a design by whether the predicate admits it, and by nothing else.

    Using the predicate itself as the reward is what makes the measurement
    clean: the target distribution is then uniform over the legal set, so
    coverage of that set is exactly what a correctly-trained sampler would
    achieve, and any shortfall is the learning problem rather than a fitness
    landscape the sampler also has to climb.
    """

    def __init__(
        self, predicate: object, alphabet: object, length: int, anchor: np.ndarray
    ) -> None:
        """Wrap a predicate as a landscape, scoring the anchor at zero.

        Excluding the anchor is not a convenience. Scored as legal it is a
        reward of one collected by stopping immediately at no risk, while every
        other legal design lies several substitutions away across states that
        score nothing -- and a policy trained against that reward learns to stop
        where it started. Measured, coverage stayed at one design in eight
        hundred after three thousand gradient steps. Since a campaign earns
        nothing for returning the sequence it began with, scoring the anchor at
        zero states the problem rather than altering it.
        """
        self._predicate = predicate
        self._alphabet = alphabet
        self._length = length
        self._anchor = np.asarray(anchor)

    @property
    def alphabet(self) -> object:
        """The alphabet designs are written in."""
        return self._alphabet

    @property
    def sequence_length(self) -> int:
        """Positions per design."""
        return self._length

    def _evaluate(self, sequences: object) -> np.ndarray:
        """One where the predicate admits the design, zero otherwise.

        The public `evaluate` validates shape and range before delegating here,
        which is why this is the hook a subclass implements.
        """
        array = np.atleast_2d(np.asarray(sequences))
        admitted = self._predicate.is_feasible(array)  # type: ignore[attr-defined]
        admitted = admitted & (array != self._anchor[None, :]).any(axis=1)
        return admitted.astype(np.float64)[:, None]


def measure(
    task: object, *, steps: int, draws: int, seed: int, objective: str = "fldb"
) -> dict[str, float]:
    """Train a policy without a local mask and measure what it covers.

    Args:
        task: A support task.
        steps: Gradient steps.
        draws: Designs sampled from the trained policy.
        seed: Seeds the policy and the sampling.
        objective: Which training objective to use; `fldb` assigns credit along
            the trajectory rather than only at the terminal, which is what the
            flow-as-oracle premise actually requires.

    Returns:
        Coverage of the legal set, the legal share of what was emitted, and the
        sizes both are taken against.
    """
    landscape = task.landscape()  # type: ignore[attr-defined]
    anchor = np.asarray(task.parent(landscape))  # type: ignore[attr-defined]
    local = _environment(task, landscape)  # type: ignore[arg-type]
    predicate = local._feasibility
    legal = predicate.is_feasible(local.enumerate_terminal_states())
    legal_designs = {
        tuple(int(t) for t in design) for design in local.enumerate_terminal_states()[legal]
    } - {tuple(int(t) for t in anchor)}

    # Terminal-only feasibility: the policy is not told which moves are safe.
    env = _environment(task, landscape, terminal_feasibility=True)  # type: ignore[arg-type]
    policy = SequencePolicy(
        n_tokens=env.alphabet.size,
        sequence_length=env.sequence_length,
        n_actions=env.n_actions,
        hidden_dim=64,
        seed=seed,
        learn_flow=True,
    )
    train_trajectory_balance(
        env,
        policy,
        FeasibilityLandscape(predicate, env.alphabet, env.sequence_length, anchor),
        TemperedReward(beta=1.0),
        TrainingConfig(steps=steps, batch_size=64, seed=seed),
        objective={
            "fldb": ForwardLookingDetailedBalance(),
            "subtb": SubTrajectoryBalance(),
            "tb": TrajectoryBalance(),
        }[objective],
    )

    generator = torch.Generator().manual_seed(seed)
    drawn = sample_trajectories(env, policy, draws, generator=generator).terminal
    admitted = predicate.is_feasible(drawn)
    covered = {tuple(int(t) for t in design) for design in drawn[admitted]} & legal_designs
    return {
        "legal": float(len(legal_designs)),
        "coverage": len(covered) / max(1, len(legal_designs)),
        "precision": float(admitted.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    """Train and measure on the support tasks named.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="Support tasks to measure.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--objective",
        default="fldb",
        choices=("fldb", "subtb", "tb"),
        help="Which objective to train against. `fldb` is the premise: a "
        "forward-looking estimate assigns credit along the trajectory, where "
        "trajectory balance sees a reward that is zero almost everywhere and "
        "only at the terminal.",
    )
    args = parser.parse_args(argv)

    configure_determinism()
    tasks = [t for t in support_tasks() if not args.task or t.name in set(args.task)]
    if not tasks:
        print(f"nothing matched task={args.task}", file=sys.stderr)
        return 2

    print(f"objective: {args.objective}")
    print(f"{'task':<16}{'|legal|':>9}{'coverage':>11}{'precision':>11}")
    for task in tasks:
        got = measure(
            task, steps=args.steps, draws=args.draws, seed=args.seed, objective=args.objective
        )
        print(
            f"{task.name:<16}{got['legal']:>9,.0f}{got['coverage']:>11.3f}"
            f"{got['precision']:>11.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
