"""Surrogate-quality intervention: does the GFlowNet's margin move with it.

Companion to the on-disk correlation in `evogfn-constraint-learning-dissociation`
(surrogate-oracle Pearson r predicts whether masking or learning carries the
GFlowNet's margin over baselines, measured across six landscape instances with
no intervention). This script is the causal half: `label_noise_std` on
`DeepEnsemble` (see `evogfn.surrogate.ensemble`) injects Gaussian noise onto
the fit targets, scaled to each fit's own target spread, and this runs the
shipped GFlowNet arm at a noise level on top of that knob against `protocol-
alde` (transition_density=0.5, a real but moderate constraint) and `gb1-anchor`
(no feasibility constraint at all -- the empirical landscape the dissociation
says learning does all the work on).

Deliberately outside `run_suite.py`'s tiers, on the precedent `proxy_saturation.py`
and `select_configuration.py` set: a diagnostic sweep over a knob no shipped
arm sets does not belong in the same `--report` output as the headline tables,
and nothing under `experiments/` is in the store's fingerprinted closure -- so
running this costs nothing against the tiers that are.

Reads the shipped arm's already-stored `label_noise_std=0.0` campaigns
straight from the store rather than re-running them (identical arm, identical
key -- `shipped_base().arm` is exactly what `run_suite.py` puts in the
headline table). Only the noisy rung is new compute.

Usage, from the repo root:
    uv run python experiments/surrogate_noise_intervention.py --task protocol-alde
        --noise 1.0 --seeds 90001
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.methods import shipped_base
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import MAIN

if TYPE_CHECKING:
    from evogfn.benchmark.store import RunRecord


def arm_name(noise: float) -> str:
    """Store-safe name for the noise rung, mirroring `+untrained@pN`'s style."""
    return f"{shipped_base().name}+noise{noise:g}"


def surrogate_r(record: RunRecord) -> float:
    """Mean per-round surrogate-oracle Pearson r, ignoring rounds with none."""
    values: list[float] = []
    for r in record.rounds:
        raw = r.get("surrogate_correlation")
        if isinstance(raw, (int, float)) and not np.isnan(raw):
            values.append(float(raw))
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    """Run the noise rung's missing seeds, then print its stored summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=[t.name for t in MAIN])
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    task = next(t for t in MAIN if t.name == args.task)
    store = ResultStore("results")
    name = arm_name(args.noise)

    if not args.report:
        base = shipped_base()
        arm = base.rung(label_noise_std=args.noise)
        from evogfn.benchmark.suite import run_task  # noqa: PLC0415

        run_task(task, {name: arm}, store, args.seeds)

    records = store.load(args.task, name)
    if records:
        rs = [surrogate_r(r) for r in records.values()]
        rs = [x for x in rs if not np.isnan(x)]
        bests = [r.best for r in records.values()]
        print(
            f"{args.task} noise={args.noise:g} n={len(records)} "
            f"mean_pearson_r={statistics.mean(rs) if rs else float('nan'):.4f} "
            f"best_mean={statistics.mean(bests):.4f} best_values={sorted(bests)}"
        )
    else:
        print(f"{args.task} noise={args.noise:g}: no records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
