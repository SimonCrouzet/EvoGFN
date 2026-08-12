"""Sweep the reference generator, including the re-anchored recovery.

The port answered its questions at one instance per shape and then swept all
but one of them. The exception was the re-anchored audit -- half of the
headline sentence, and the half that says what a campaign which *moves* its
anchor recovers from a planted optimum no fixed-anchor ordering can build.
`sweep_instance` now answers it, and this is the runner.

Nothing here is campaign compute: no sampler trains and no oracle budget is
spent. The cost is holo's constructor plus, where the audit runs, a beam
search -- and the re-anchored audit chains one beam per round, so it is the
expensive part and rides on the same per-shape audit gate.

Usage:
    uv run python experiments/run_holo_sweep.py            # every shape
    uv run python experiments/run_holo_sweep.py --shape diagnostic
"""

from __future__ import annotations

import argparse
import sys
import time

from evogfn.benchmark.holo_port import REFERENCE_SHAPES, summarise, sweep_report, sweep_shape


def main(argv: list[str] | None = None) -> int:
    """Sweep the shapes asked for and print the report.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code: 0 on success, 2 when no shape matched.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", help="Sweep only these shapes.")
    parser.add_argument("--workers", type=int, default=12, help="Parallel instances.")
    args = parser.parse_args(argv)

    shapes = [s for s in REFERENCE_SHAPES if not args.shape or s.name in set(args.shape)]
    if not shapes:
        print(f"nothing matched shape={args.shape}", file=sys.stderr)
        return 2

    summaries = []
    for shape in shapes:
        started = time.perf_counter()
        outcomes = sweep_shape(shape, workers=args.workers)
        summaries.append(summarise(shape.name, outcomes))
        print(
            f"{shape.name}: {len(outcomes)} instances in {time.perf_counter() - started:.0f}s",
            flush=True,
        )
        print(summaries[-1], flush=True)
    print(sweep_report(summaries), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
