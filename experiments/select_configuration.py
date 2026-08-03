"""Fix the GFlowNet's configuration before the benchmark measures it.

Every classical baseline in the suite runs at hyperparameters its own authors
tuned. A GFlowNet run at inherited defaults against that field is not being
compared to it, so this phase runs first and its answer is an input to the
headline tables rather than one of their rows.

Two stages, because the full cross of objectives and reward exponents is far
more compute than the question needs:

**Stage A** compares the six training objectives at the default exponent, at
`SELECTION_SEEDS` seeds per arm -- enough that a gap worth acting on can be
separated from noise, and enough to state honestly that the rest are tied.

**Stage B** scans the reward exponent for whichever objective stage A chose. It
runs second because the winner is not known until stage A finishes, and it scans
only the winner because scanning all six would cost six times the compute. The
cost of that economy is real and worth naming: an objective that loses at the
default exponent and would have won at another is invisible to this design.

The rule is in [evogfn.benchmark.selection][], written down before the numbers
arrived. Both stages run on the diagnostic landscape, which no headline task
uses, so nothing here is tuning on the test set.

    uv run python experiments/select_configuration.py            # both stages
    uv run python experiments/select_configuration.py --report   # read, no runs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.methods import OBJECTIVES, flow_objectives
from evogfn.benchmark.selection import Scored, Selection, beta_arms, select, steps_arms
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import Purpose, Tier, objective_task, run_tier

#: Seeds per arm. High enough that the closest pair of objectives can be
#: separated rather than declared tied by default, since the rule in
#: [evogfn.benchmark.selection][] falls back to a tie-break whenever the paired
#: test cannot resolve a gap.
SELECTION_SEEDS = 100

#: Where the chosen configuration is written, so the benchmark reads a decision
#: rather than re-deriving one and possibly re-deriving it differently.
CHOICE_FILE = Path("results/selected.json")


def stage_a_methods() -> dict[str, object]:
    """The six training objectives, at the default reward exponent."""
    return {**OBJECTIVES, **flow_objectives()}


def run_stage(  # noqa: PLR0913 - a stage is named, scoped, and told what to run
    name: str,
    methods: dict[str, object],
    store: ResultStore,
    *,
    report: bool,
    seeds: int,
    runnable: range | None = None,
) -> None:
    """Run one stage's arms on the diagnostic landscape.

    Args:
        name: Stage name, for the progress line.
        methods: The arms this process is running.
        store: Where campaigns are written.
        report: Read without running.
        runnable: Which seeds this process runs. A slice rather than a count,
            because sharding by arm alone leaves the slowest arm running its
            hundred seeds inside one process while every other core idles --
            and a stage cannot finish faster than its slowest single arm.
        seeds: How many seeds to run. Taken from the caller rather than read
            from `SELECTION_SEEDS` here, so that `--seeds` governs what actually
            runs and not merely the completeness check.
    """
    runnable = range(seeds) if runnable is None else runnable
    tier = Tier(name, (objective_task(),), tuple(runnable), Purpose.SELECTION)
    if not report:
        ran = run_tier(tier, methods, store, report=_flush)  # type: ignore[arg-type]
        _flush(f"{name}: ran {ran} campaigns")


def held(store: ResultStore, methods: dict[str, object]) -> dict[str, dict[int, Scored]]:
    """Usable records per arm, dropping arms with nothing stored yet."""
    task = objective_task().name
    found = {name: store.usable(task, name) for name in methods}
    kept = {name: records for name, records in found.items() if records}
    # RunRecord satisfies Scored structurally; the store is typed concretely.
    return cast("dict[str, dict[int, Scored]]", kept)


def describe(stage: str, choice: Selection) -> str:
    """Lay out a stage's table and the decision drawn from it."""
    lines = [f"\n=== {stage} ===="]
    for name in sorted(choice.regret, key=lambda n: choice.regret[n]):
        mark = "<-- chosen" if name == choice.chosen else ("tied" if name in choice.tied else "")
        lines.append(
            f"  {name:<24} regret {choice.regret[name]:7.4f}  "
            f"div {choice.diversity[name]:6.2f}  {mark}"
        )
    lines.append(f"  decision: {choice.chosen}")
    lines.append(f"  because:  {choice.reason}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run both stages and record the configuration they select."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="Read the store without running.")
    parser.add_argument("--results", default="results", help="Where results live.")
    parser.add_argument("--seeds", type=int, default=SELECTION_SEEDS, help="Seeds per arm.")
    parser.add_argument(
        "--seed-from",
        type=int,
        default=0,
        help="First seed this process runs. Sharding by arm alone leaves the "
        "slowest arm on one core while the rest of the machine idles; a seed "
        "slice lets several processes share one arm. The completeness check "
        "still reads `--seeds`, so a shard that finishes its slice stops "
        "rather than choosing on a fraction of the evidence.",
    )
    parser.add_argument("--seed-to", type=int, default=None, help="One past the last seed.")
    parser.add_argument(
        "--only",
        action="append",
        help="Run just these arms. Sharding across processes rather than threads "
        "is what makes this safe: campaigns are independent, the store keeps one "
        "file per arm so writers never collide, and every campaign is seeded from "
        "its own seed rather than from process order -- so a sharded run and a "
        "serial one produce identical records. Raising the thread count instead "
        "would not, since a multithreaded reduction sums in completion order.",
    )
    parser.add_argument(
        "--stage",
        choices=("a", "b", "c", "both"),
        default="both",
        help="Which stage to run. Shards use 'a', then one process runs 'b'.",
    )
    parser.add_argument(
        "--print-winner",
        action="store_true",
        help="Print stage A's chosen arm and exit, for a coordinator to shard on.",
    )
    args = parser.parse_args(argv)

    configure_determinism()
    if not args.report and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    store = ResultStore(Path(args.results))
    started = time.perf_counter()

    wanted = set(args.only) if args.only else None
    objective = _stage_a(store, args, wanted)
    if not isinstance(objective, Selection):
        return objective
    exponent = _stage_b(store, args, wanted, objective)
    if not isinstance(exponent, Selection):
        return exponent
    beta = float(exponent.chosen.rsplit("-", 1)[1])
    gradient = _stage_c(store, args, wanted, objective, beta)
    if not isinstance(gradient, Selection):
        return gradient

    choice = {
        # Explicit fields rather than a name to be parsed. `run_suite` reads
        # these to rebuild the arm, and picking a configuration apart from a
        # string breaks the moment the naming scheme grows a component.
        "beta": beta,
        "steps": int(gradient.chosen.rsplit("-", 1)[1]),
        "objective": objective.chosen,
        "objective_reason": objective.reason,
        "objective_tied": list(objective.tied),
        "arm": gradient.chosen,
        "arm_reason": exponent.reason,
        "steps_reason": gradient.reason,
        "seeds": args.seeds,
        "task": objective_task().name,
    }
    if not args.report:
        CHOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHOICE_FILE.write_text(json.dumps(choice, indent=2) + "\n")
    _flush(f"\nselected {gradient.chosen}")
    _flush(f"total {time.perf_counter() - started:.0f}s")
    return 0


def _shard(
    arms: dict[str, object], wanted: set[str] | None, stage: str
) -> dict[str, object] | None:
    """Narrow a stage's arms to this process's share."""
    picked = {k: v for k, v in arms.items() if wanted is None or k in wanted}
    if not picked:
        print(f"--only matched no stage {stage} arm", file=sys.stderr)
        return None
    return picked


def _stage_a(
    store: ResultStore, args: argparse.Namespace, wanted: set[str] | None
) -> Selection | int:
    """Run the objective comparison, or this shard of it, and choose.

    Returns:
        The choice, or a process exit code when this shard cannot make one.
    """
    objectives = stage_a_methods()
    if args.stage in {"a", "both"}:
        shard = _shard(objectives, wanted, "A")
        if shard is None:
            return 2
        run_stage(
            "select-objective",
            shard,
            store,
            report=args.report or args.print_winner,
            seeds=args.seeds,
            runnable=range(args.seed_from, args.seed_to or args.seeds),
        )

    stored = held(store, objectives)
    # Selection needs every arm, not this shard's: a winner drawn from a subset
    # is a winner of the subset. A shard that finishes first therefore stops
    # here rather than choosing on partial evidence.
    missing = [name for name in objectives if len(stored.get(name, {})) < args.seeds]
    if missing:
        if args.stage == "a":
            _flush(f"shard done; still incomplete: {', '.join(sorted(missing))}")
            return 0
        print(f"stage A incomplete for {sorted(missing)}", file=sys.stderr)
        return 2
    objective = select(stored)
    if args.print_winner:
        print(objective.chosen)
        return 0
    _flush(describe("stage A: training objective", objective))
    return objective


def _stage_b(
    store: ResultStore, args: argparse.Namespace, wanted: set[str] | None, objective: Selection
) -> Selection | int:
    """Scan the reward exponent for the objective stage A chose."""
    exponents = beta_arms(objective.chosen)
    if args.stage in {"b", "both"}:
        shard = _shard(exponents, wanted, "B")  # type: ignore[arg-type]
        if shard is None:
            return 2
        run_stage(
            "select-beta",
            shard,
            store,
            report=args.report,
            seeds=args.seeds,
            runnable=range(args.seed_from, args.seed_to or args.seeds),
        )
    stored = held(store, exponents)  # type: ignore[arg-type]
    if [n for n in exponents if len(stored.get(n, {})) < args.seeds]:
        _flush("stage B shard done; not every exponent is complete yet")
        return 0
    exponent = select(stored)
    _flush(describe("stage B: reward exponent", exponent))
    return exponent


def _stage_c(
    store: ResultStore,
    args: argparse.Namespace,
    wanted: set[str] | None,
    objective: Selection,
    beta: float,
) -> Selection | int:
    """Scan gradient steps -- the GFlowNet's proxy budget -- for the chosen arm.

    Runs last because it depends on both earlier choices. Proxy spend is a
    reported column in the results table, so `steps` is a number the paper
    prints and is decided here rather than inherited as an internal default.
    """
    arms = steps_arms(objective.chosen, beta)
    if args.stage in {"c", "both"}:
        shard = _shard(arms, wanted, "C")  # type: ignore[arg-type]
        if shard is None:
            return 2
        run_stage(
            "select-steps",
            shard,
            store,
            report=args.report,
            seeds=args.seeds,
            runnable=range(args.seed_from, args.seed_to or args.seeds),
        )
    stored = held(store, arms)  # type: ignore[arg-type]
    if [n for n in arms if len(stored.get(n, {})) < args.seeds]:
        _flush("stage C shard done; not every step count is complete yet")
        return 0
    chosen = select(stored)
    _flush(describe("stage C: gradient steps (proxy budget)", chosen))
    return chosen


def _flush(message: str) -> None:
    """Print immediately, so a long phase can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
