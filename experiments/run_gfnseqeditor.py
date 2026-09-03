"""Run the GFNSeqEditor baseline arm on a constrained task, into the store.

The arm is registered in `evogfn.benchmark.methods.BASELINES` as ``gfnseqeditor``
at the authors' reported ``delta=0.4, lambda=0.1, sigma=0.001``. This runs it
through the same `run_task` path every other arm takes, so its records land
keyed by ``(task, "gfnseqeditor", seed)`` beside the rest and pair against them.

Usage, from the repo root:
    uv run python experiments/run_gfnseqeditor.py --task protocol-alde --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys

from evogfn.benchmark.methods import BASELINES
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import MAIN, run_task


def main() -> int:
    """Run the missing GFNSeqEditor seeds for one task."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=[t.name for t in MAIN])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    task = next(t for t in MAIN if t.name == args.task)
    store = ResultStore("results")
    run_task(task, {"gfnseqeditor": BASELINES["gfnseqeditor"]}, store, args.seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
