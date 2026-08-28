"""Whether a learned flow, used to order candidates, buys depth it does not pay for.

The exact lookahead family prices soundness directly: a mask seeing `k` moves
ahead searches ``alphabet**k`` continuations per move, and on a codebook
predicate at separation four nothing below `k = 3` recovers anything at all.
The cost is therefore exponential in the depth, and the depth is set by the
predicate rather than by the budget.

Between paying that cost and not paying it lies a third option. A mask may
examine only the `m` continuations a learned flow ranks highest, at any depth,
which costs `m` evaluations per move instead of ``alphabet**k``. If the flow
carries any information about which branches survive, a shallow search over the
branches it favours should reach more of the legal set than a shallow search
over all of them -- and that is measurable exactly here, because the legal set is
enumerable and the exact depth ladder is already known.

The comparison is against two references rather than one, since either alone can
mislead: the exhaustive search at the same depth, which is what the flow must
beat to be worth consulting, and the exhaustive search one depth deeper, which
is what it would have to match to have bought a depth for free.

Nothing here touches the fingerprinted closure.

Usage:
    uv run python experiments/support_hybrid.py --task support-sep4
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations, product

import numpy as np
import torch
from support_flowmask import FeasibilityLandscape  # type: ignore[import-not-found]

from evogfn.algorithms.gflownet.flow_objectives import ForwardLookingDetailedBalance
from evogfn.algorithms.gflownet.training import TrainingConfig, train_trajectory_balance
from evogfn.benchmark.determinism import configure_determinism
from evogfn.benchmark.methods import _environment
from evogfn.benchmark.suite import support_tasks
from evogfn.models.policy import SequencePolicy
from evogfn.rewards.base import TemperedReward

#: Continuations the flow-ranked search is allowed to examine per move. Small on
#: purpose: the whole question is whether ranking substitutes for breadth.
DEFAULT_BEAM = 8

#: Gradient steps for the flow that does the ranking.
DEFAULT_STEPS = 8_000


def _train_flow(task: object, steps: int, seed: int) -> tuple[SequencePolicy, object]:
    """Train a forward-looking policy whose flow head ranks continuations.

    Args:
        task: A support task.
        steps: Gradient steps.
        seed: Seeds the policy.

    Returns:
        The trained policy and the environment it was trained on.
    """
    landscape = task.landscape()  # type: ignore[attr-defined]
    anchor = np.asarray(task.parent(landscape))  # type: ignore[attr-defined]
    predicate = _environment(task, landscape)._feasibility  # type: ignore[arg-type]
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
        objective=ForwardLookingDetailedBalance(),
    )
    return policy, env


def _ranked(policy: SequencePolicy, env: object, states: np.ndarray, beam: int) -> np.ndarray:
    """The ``beam`` actions the policy scores highest at each state.

    Ranked under the *structural* mask of the terminal-feasibility environment,
    so the ordering reflects what the policy learned rather than what a
    feasibility mask would already have refused -- the point being to test the
    ranking, not to reintroduce the mask through it.

    Args:
        policy: The trained policy.
        env: The environment supplying the structural mask.
        states: An ``(n, length)`` array.
        beam: How many actions to keep per state.

    Returns:
        An ``(n, beam)`` array of action indices.
    """
    from evogfn.env.base import State  # noqa: PLC0415 - only needed here

    state = State(sequences=np.asarray(states), stopped=np.zeros(len(states), dtype=bool))
    forward = env.forward_mask(state)  # type: ignore[attr-defined]
    backward = env.backward_mask(state)  # type: ignore[attr-defined]
    with torch.no_grad():
        log_forward, _ = policy.log_probs(
            torch.as_tensor(np.asarray(states), dtype=torch.long),
            torch.as_tensor(forward),
            torch.as_tensor(backward),
        )
    return np.asarray(torch.topk(log_forward, k=beam, dim=-1).indices)


def reachable_flow_ranked(task: object, policy: SequencePolicy, beam: int, depth: int = 2) -> set:
    """Designs a depth-one search restricted to the flow's top ``beam`` moves reaches.

    Args:
        task: A support task.
        policy: The trained policy supplying the ranking.
        beam: First moves examined per state, taken in the policy's order.
        depth: Depth of the exhaustive test applied to each of them. Ranking
            only pays where the test it gates is deeper than what exhaustive
            search affords at the same cost, so a depth of one -- already at the
            floor on every rung -- can say nothing.

    Returns:
        The legal designs the restricted search can build.
    """
    landscape = task.landscape()  # type: ignore[attr-defined]
    local = _environment(task, landscape)  # type: ignore[arg-type]
    predicate = local._feasibility
    anchor = np.asarray(task.parent(landscape))  # type: ignore[attr-defined]
    length, tokens = len(anchor), landscape.alphabet.size
    budget = task.max_mutations  # type: ignore[attr-defined]
    term_env = _environment(task, landscape, terminal_feasibility=True)  # type: ignore[arg-type]

    start = tuple(int(t) for t in anchor)
    seen, frontier = {start}, [start]
    terminals: set = set()
    while frontier:
        nxt = []
        batch = np.asarray(frontier)
        ranked = _ranked(policy, term_env, batch, min(beam, length * tokens))
        for row, state in enumerate(frontier):
            array = np.asarray(state)
            if predicate.is_feasible(array[None, :])[0]:
                terminals.add(state)
            if int((array != anchor).sum()) >= budget:
                continue
            for action in ranked[row]:
                position, token = divmod(int(action), tokens)
                if position >= length or array[position] != anchor[position]:
                    continue
                if token == anchor[position]:
                    continue
                child = array.copy()
                child[position] = token
                key = tuple(int(t) for t in child)
                if key in seen:
                    continue
                untouched = tuple(q for q in range(length) if child[q] == anchor[q])
                candidates = [child]
                for extra in range(1, depth + 1):
                    for positions in combinations(untouched, extra):
                        for assignment in product(range(tokens), repeat=extra):
                            nxt_state = child.copy()
                            nxt_state[list(positions)] = assignment
                            candidates.append(nxt_state)
                if predicate.is_feasible(np.asarray(candidates)).any():
                    seen.add(key)
                    nxt.append(key)
        frontier = nxt
    return terminals


def main(argv: list[str] | None = None) -> int:
    """Compare a flow-ranked shallow search against the exhaustive depth ladder.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="Support tasks to measure.")
    parser.add_argument("--beam", type=int, default=DEFAULT_BEAM)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    configure_determinism()
    tasks = [t for t in support_tasks() if not args.task or t.name in set(args.task)]
    if not tasks:
        print(f"nothing matched task={args.task}", file=sys.stderr)
        return 2

    print(f"beam {args.beam}, depth-{args.depth} test over the ranked first moves")
    print(f"{'task':<16}{'|legal|':>9}{'flow-ranked':>14}")
    for task in tasks:
        landscape = task.landscape()
        local = _environment(task, landscape)
        predicate = local._feasibility
        legal = int(predicate.is_feasible(local.enumerate_terminal_states()).sum())
        policy, _ = _train_flow(task, args.steps, args.seed)
        reached = len(reachable_flow_ranked(task, policy, args.beam, args.depth))
        print(
            f"{task.name:<16}{legal:>9,}{f'{reached:,} ({reached / legal:.3f})':>14}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
