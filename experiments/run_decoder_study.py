"""Run the decoder study across the tiers the reported comparison is drawn on.

`DECODER_STUDY` is deliberately not in `BASELINES`
(`docs/benchmark.md`, "`cmaes+dp` is deliberately not a baseline"): a stronger
decoder than any published account of CMA-ES uses is our engineering, not
Hansen's method, so it must never reach a baseline table by resolving inside a
tier's roster. The consequence is that **no tier runs it**, and the arm
therefore exists only where somebody asked for it by hand.

That produced a gap with a bearing on what can be claimed. `cmaes+dp` is held on
three of the five `main` tasks and on none of the six `replication` tasks, while
the comparison it anchors -- an exact projection tying a learned constructive
sampler -- is read across tasks. `protocol-alde` and `protocol-evolvepro` are the
same Ehrlich instance and `feasibility` is a constant, so the effective number of
independent landscapes behind that tie is about one. The `replicate-*` tasks are
the six *other* instances, and they are the only cheap way to raise it: the arm
costs on the order of three CPU-seconds a seed and calls no surrogate at all.

This script is the missing runner and nothing more. It defines no task, no tier
and no arm: it takes the tiers `run_suite` already declares, keeps the tasks
asked for, and runs `DECODER_STUDY` over them through the same
[run_tier][evogfn.benchmark.suite.run_tier] every other arm goes through, so a
record it writes is the same kind of object -- same budget, same protocol, same
fingerprinted closure -- as one written by the suite. Restricting to a task list
is what makes it usable for the gap rather than for a re-run: the default target
is exactly the cells that are missing.

Usage:
    uv run python experiments/run_decoder_study.py            # the missing cells
    uv run python experiments/run_decoder_study.py --report   # say what is held
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_suite import tiers  # type: ignore[import-not-found]

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.methods import DECODER_STUDY
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import Tier, run_tier

#: Tiers this study is drawn across. `main` carries the reported comparison and
#: `replication` carries the six independent instances that say whether it is a
#: property of the method or of one draw.
STUDY_TIERS = ("main", "replication")

#: Seeds, matching `run_suite.MAIN_SEEDS`. The arm is cheap enough that a
#: reduced count would be a decision about nothing.
STUDY_SEEDS = 100


def main(argv: list[str] | None = None) -> int:
    """Run or report on the decoder study.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code: 0 on success, 2 when nothing matched the selection,
        3 when threading is not pinned and a run was asked for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", action="append", help=f"Run only these tiers of {STUDY_TIERS}.")
    parser.add_argument("--task", action="append", help="Run only these tasks.")
    parser.add_argument("--method", action="append", help="Run only these decoder-study arms.")
    parser.add_argument("--seeds", type=int, default=STUDY_SEEDS, help="Seeds per cell.")
    parser.add_argument("--results", default="results", help="Where to store results.")
    parser.add_argument("--report", action="store_true", help="Say what is held, run nothing.")
    args = parser.parse_args(argv)

    configure_determinism()
    if not args.report and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    arms = {
        name: arm
        for name, arm in DECODER_STUDY.items()
        if not args.method or name in set(args.method)
    }
    wanted_tiers = set(args.tier) if args.tier else set(STUDY_TIERS)
    selected = [t for t in tiers(args.seeds, args.seeds) if t.name in wanted_tiers]
    if args.task:
        wanted = set(args.task)
        selected = [
            Tier(t.name, tuple(k for k in t.tasks if k.name in wanted), t.seeds, t.purpose)
            for t in selected
        ]
        selected = [t for t in selected if t.tasks]
    if not selected or not arms:
        print(
            f"nothing matched tier={args.tier} task={args.task} method={args.method}",
            file=sys.stderr,
        )
        return 2

    store = ResultStore(Path(args.results))
    if args.report:
        for tier in selected:
            for task in tier.tasks:
                for name in arms:
                    held = len(store.usable(task.name, name))
                    stale = len(store.stale(task.name, name))
                    print(f"{task.name:<24} {name:<10} held {held:>3}  stale {stale:>3}")
        return 0

    started = time.perf_counter()
    for tier in selected:
        ran = run_tier(tier, arms, store, report=_flush)
        _flush(f"{tier.name}: ran {ran} campaigns")
    _flush(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


def _flush(message: str) -> None:
    """Print immediately, so a long run can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
