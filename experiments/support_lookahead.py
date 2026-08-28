"""How much lookahead a local mask would need to recover the feasible set.

The support study measures what a *local* mask reaches: a move is admitted when
the state it lands on satisfies the predicate, and nothing asks whether a
feasible design still lies ahead. A sound mask asks exactly that question, and
where the predicate's constraint graph is a path the two coincide. Where it is
not, they need not, and on the budget band they come apart on the first move.

Between those two masks lies a family. A **depth-`k` lookahead** mask admits a
move when the resulting state is feasible *or* some sequence of at most `k`
further substitutions reaches a feasible design. Depth 0 is the local mask;
depth `L` is the completion oracle. Sweeping `k` therefore prices the
approximation: it says how far ahead a mask has to see before the support it
reaches resembles the support that exists, and it does so without training
anything, which is what makes it a control rather than a method.

Nothing here touches the fingerprinted closure -- the walk is written against
the predicate interface rather than against an environment -- so this script
stales no stored record and needs no bless.

Usage:
    uv run python experiments/support_lookahead.py
    uv run python experiments/support_lookahead.py --task support-band --depth 2
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations, product

import numpy as np

from evogfn.benchmark.methods import _environment
from evogfn.benchmark.suite import support_tasks

#: Depths swept when none are named. Zero is the local mask and is always run,
#: because every other number is read against it.
DEFAULT_DEPTHS = (0, 1, 2)


def _feasible_within(
    state: np.ndarray, untouched: tuple[int, ...], predicate: object, depth: int, tokens: int
) -> bool:
    """Whether some assignment of at most ``depth`` untouched positions is feasible.

    Args:
        state: One complete sequence.
        untouched: Positions still at the anchor's token, and therefore still
            substitutable under the once-per-position rule.
        predicate: The feasibility predicate.
        depth: Most further substitutions to consider.
        tokens: Alphabet size.

    Returns:
        Whether a feasible design lies within ``depth`` substitutions.
    """
    if predicate.is_feasible(state[None, :])[0]:  # type: ignore[attr-defined]
        return True
    for extra in range(1, depth + 1):
        for positions in combinations(untouched, extra):
            candidates = []
            for assignment in product(range(tokens), repeat=extra):
                candidate = state.copy()
                candidate[list(positions)] = assignment
                candidates.append(candidate)
            if predicate.is_feasible(np.asarray(candidates)).any():  # type: ignore[attr-defined]
                return True
    return False


def reachable_under_lookahead(task: object, depth: int) -> set[tuple[int, ...]]:
    """Terminal states a depth-``k`` lookahead mask can build.

    A breadth-first walk from the anchor, admitting a substitution when the
    lookahead test passes. Written against the predicate rather than against an
    environment so that the mask can be varied without defining a new
    environment class for each depth.

    Args:
        task: A support task.
        depth: Lookahead depth; 0 is the local mask.

    Returns:
        Every feasible design the walk can end on.
    """
    landscape = task.landscape()  # type: ignore[attr-defined]
    environment = _environment(task, landscape)  # type: ignore[arg-type]
    predicate = environment._feasibility
    anchor = np.asarray(task.parent(landscape))  # type: ignore[attr-defined]
    length, tokens = len(anchor), landscape.alphabet.size
    budget = task.max_mutations  # type: ignore[attr-defined]

    start = tuple(int(t) for t in anchor)
    seen, frontier = {start}, [start]
    terminals: set[tuple[int, ...]] = set()
    while frontier:
        nxt = []
        for state in frontier:
            array = np.asarray(state)
            if predicate.is_feasible(array[None, :])[0]:
                terminals.add(state)
            touched = tuple(p for p in range(length) if array[p] != anchor[p])
            if len(touched) >= budget:
                continue
            untouched = tuple(p for p in range(length) if array[p] == anchor[p])
            for position in untouched:
                for token in range(tokens):
                    if token == anchor[position]:
                        continue
                    child = array.copy()
                    child[position] = token
                    key = tuple(int(t) for t in child)
                    if key in seen:
                        continue
                    remaining = tuple(q for q in untouched if q != position)
                    if _feasible_within(child, remaining, predicate, depth, tokens):
                        seen.add(key)
                        nxt.append(key)
        frontier = nxt
    return terminals


def main(argv: list[str] | None = None) -> int:
    """Sweep lookahead depth on the support tasks and report the support reached.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="Run only these support tasks.")
    parser.add_argument("--depth", action="append", type=int, help="Depths to sweep.")
    args = parser.parse_args(argv)

    depths = tuple(args.depth) if args.depth else DEFAULT_DEPTHS
    tasks = [t for t in support_tasks() if not args.task or t.name in set(args.task)]
    if not tasks:
        print(f"nothing matched task={args.task}", file=sys.stderr)
        return 2

    header = "".join(f"{f'depth {d}':>14}" for d in depths)
    print(f"{'task':<16}{'|feasible|':>12}{header}")
    for task in tasks:
        landscape = task.landscape()
        environment = _environment(task, landscape)
        predicate = environment._feasibility
        feasible = int(predicate.is_feasible(environment.enumerate_terminal_states()).sum())
        cells = ""
        for depth in depths:
            reached = len(reachable_under_lookahead(task, depth))
            cells += f"{f'{reached:,} ({reached / feasible:.3f})':>14}"
        print(f"{task.name:<16}{feasible:>12,}{cells}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
