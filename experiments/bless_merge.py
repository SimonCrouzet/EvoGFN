"""Bless the records the solidify-paper-controls merge staled but could not alter.

The merge that added the mask-aware GA, the surrogate-noise knob, the width-2
predicate tier and the rounds rung edited five modules that sit in almost every
record's fingerprint closure -- so on this branch the whole store reads as
stale, including headline campaigns none of the four changes could have touched.

Each change was verified to be behaviour-preserving for every *existing* arm
before this script was written, which is the one assertion `ResultStore.bless`
exists to make explicit:

* ``store``   -- pure additions (two construction counters); zero removals, so
  every existing record still deserialises unchanged.
* ``ensemble`` -- the label-noise draw is guarded by ``std > 0`` and uses its
  own RNG instance, so at the default ``0.0`` there is no draw and the
  init/batch/resample stream is untouched: every surrogate fit is byte-identical.
* ``genetic`` -- the new construction path is reached only for the new
  ``genetic-masked`` arm; ``genetic`` and ``genetic-feasible`` breed and score
  exactly as before, and the ``name`` branch changes no existing arm's output.
* ``methods``  -- existing arm builders are unchanged; the one touched signature
  threads ``label_noise_std=0.0`` (a no-op) and adds new arms/tasks beside them.
* ``suite``    -- existing task definitions are untouched; the width-2 and rounds
  tasks are added.

A record produced *by* one of the new arms, or re-run under the merged code, is
current already, so `bless` restamps nothing for it -- this only waves through
the records the merge left behind untouched.

Run AFTER the four re-run tiers have finished and after any further edit to the
five modules (e.g. the objective-family build), so the blessing is stamped
against the final content rather than an intermediate one.

Usage:
    uv run python experiments/bless_merge.py            # bless
    uv run python experiments/bless_merge.py --report   # count what is stale, bless nothing
"""

from __future__ import annotations

import argparse
import sys

from evogfn.benchmark.store import ResultStore

#: The five modules the merge changed, each verified behaviour-preserving for
#: every existing arm above. Dotted names as they appear in a record's `source`.
CHANGED_MODULES: tuple[str, ...] = (
    "evogfn.algorithms.baselines.genetic",
    "evogfn.benchmark.methods",
    "evogfn.benchmark.store",
    "evogfn.benchmark.suite",
    "evogfn.surrogate.ensemble",
)


def main(argv: list[str] | None = None) -> int:
    """Bless every stale existing-arm record against the five changed modules."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="store_true",
        help="Count records stale in a changed module without restamping any.",
    )
    parser.add_argument("--results", default="results", help="Store root.")
    args = parser.parse_args(argv)

    store = ResultStore(args.results)
    changed_set = set(CHANGED_MODULES)
    total_blessed = 0
    total_stale = 0
    for task in store.tasks():
        for method in store.methods(task):
            stale = store.stale(task, method)
            hits = sum(1 for mods in stale.values() if changed_set & set(mods))
            total_stale += hits
            if args.report or hits == 0:
                continue
            blessed = store.bless(task, method, modules=CHANGED_MODULES)
            total_blessed += blessed
            if blessed:
                print(f"{task}/{method}: blessed {blessed}")

    if args.report:
        print(f"stale in a changed module: {total_stale} records across the store")
    else:
        print(f"blessed {total_blessed} records; {total_stale} were stale in a changed module")
    return 0


if __name__ == "__main__":
    sys.exit(main())
