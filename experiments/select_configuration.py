"""Fix the GFlowNet's configuration before the benchmark measures it.

Every classical baseline in the suite runs at hyperparameters its own authors
tuned. A GFlowNet run at inherited defaults against that field is not being
compared to it, so this phase runs first and its answer is an input to the
headline tables rather than one of their rows.

**Stage A** compares the six training objectives at the default exponent, at
`SELECTION_SEEDS` seeds per arm -- enough that a gap worth acting on can be
separated from noise, and enough to state honestly that the rest are tied.

**Stage B** scans the reward exponent for whichever objective stage A chose. It
runs second because the winner is not known until stage A finishes, and it scans
only the winner because scanning all six would cost six times the compute.

**Stage C** stops moving one axis at a time. A one-axis-at-a-time design cannot
see interactions, and -- worse -- it never reaches the parameters that make the
leading objectives *families* rather than points, since those were never on any
of its axes. So stage C samples a joint space at random and does it in two
halves that must not be confused with each other:

*The screen* runs many configurations at `SCREEN_SEEDS` seeds. Its only output
is a shortlist. It cannot rank anything, no number it produces is reported, and
the shortlist is a human's call rather than this script's -- the finalists are
named on the command line.

*The confirmation* runs the shortlist at the full seed count through the same
pre-declared rule every earlier stage used, **plus the incumbent, always**. That
last part is what makes the screen safe: it can only add a better configuration,
never displace the standing one on evidence too thin to displace anything.

The rule is in [evogfn.benchmark.selection][], written down before the numbers
arrived. Every stage runs on the diagnostic landscape, which no headline task
uses, so nothing here is tuning on the test set.

    uv run python experiments/select_configuration.py            # every stage
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
from evogfn.benchmark.selection import (
    CONFIRM_FLOOR,
    CONFIRM_SLOTS,
    SCREEN_SAMPLING_SEED,
    SCREEN_SEEDS,
    SCREEN_SIZE,
    SCREENED_OBJECTIVES,
    Configuration,
    Scored,
    Screen,
    Selection,
    beta_arms,
    confirmation_set,
    incumbent,
    propose_allocation,
    rank_screen,
    sample_screen,
    screen_arms,
    select,
)
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
        choices=("a", "b", "c", "c-screen", "c-confirm", "both"),
        default="both",
        help="Which stage to run. Shards use 'a', then one process runs 'b'. "
        "Stage C splits: 'c-screen' samples and runs the joint screen, "
        "'c-confirm' re-measures the finalists named by --confirm, and 'c' does "
        "both -- which in practice stops after the screen, since the finalists "
        "are chosen once the screen has landed.",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default=SCREEN_SIZE,
        help="Configurations drawn per screened objective.",
    )
    parser.add_argument(
        "--screen-seeds",
        type=int,
        default=SCREEN_SEEDS,
        help="Seeds per screened configuration. Low on purpose: the screen "
        "nominates and the confirmation measures, so resolution bought here is "
        "bought where no claim is drawn.",
    )
    parser.add_argument(
        "--screen-seed",
        type=int,
        default=SCREEN_SAMPLING_SEED,
        help="Seed for the draw. The candidate set is a function of this and "
        "the declared grids, so it can be reproduced and checked.",
    )
    parser.add_argument(
        "--screen-shown",
        type=int,
        default=15,
        help="How much of each screen's ordering to print.",
    )
    parser.add_argument(
        "--confirm",
        action="append",
        default=[],
        help="A finalist's arm name. Repeatable, and required before the "
        "confirmation runs: allocating a fixed budget between the screens is a "
        "judgement rather than an arithmetic, so it is made by a person and "
        "written down here. The incumbent is confirmed regardless and does not "
        "need naming.",
    )
    parser.add_argument(
        "--confirm-slots",
        type=int,
        default=CONFIRM_SLOTS,
        help="Finalists the printed allocation proposal splits between screens.",
    )
    parser.add_argument(
        "--confirm-floor",
        type=int,
        default=CONFIRM_FLOOR,
        help="Fewest finalists the proposal gives any screen, so a ten-seed "
        "screen cannot write an objective off entirely.",
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
    confirmed = _stage_c(store, args, wanted, objective, beta)
    if not isinstance(confirmed, Selection):
        return confirmed

    # Read back from the winning arm's stored settings rather than from its
    # name. The record carries what the methodology closed over, so it is what
    # actually ran; `from_record` still parses the name and refuses when the two
    # disagree, which is the only way that disagreement is ever visible.
    winner = store.usable(objective_task().name, confirmed.chosen)
    chosen = Configuration.from_record(confirmed.chosen, next(iter(winner.values())).parameters)
    choice = {
        # Explicit fields rather than a name to be parsed. `run_suite` reads
        # these to rebuild the arm, and picking a configuration apart from a
        # string breaks the moment the naming scheme grows a component.
        "beta": chosen.beta,
        "steps": chosen.steps,
        "lam": chosen.lam,
        "mix": chosen.mix,
        "hidden_dim": chosen.hidden_dim,
        "objective": chosen.objective,
        "objective_reason": objective.reason,
        "objective_tied": list(objective.tied),
        "arm": confirmed.chosen,
        "arm_reason": exponent.reason,
        "confirmed_reason": confirmed.reason,
        # The screen's provenance, so the candidate set can be redrawn. Its
        # numbers are deliberately absent: it nominated, and nothing it measured
        # is reported.
        "screen_seed": args.screen_seed,
        "screen_size": args.screen_size,
        "screen_seeds": args.screen_seeds,
        "finalists": sorted(args.confirm),
        "seeds": args.seeds,
        "task": objective_task().name,
    }
    if not args.report:
        CHOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CHOICE_FILE.write_text(json.dumps(choice, indent=2) + "\n")
    _flush(f"\nselected {confirmed.chosen}")
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


def _screen_arms(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    """The sampled configurations of every screened objective, by objective."""
    return {
        name: cast(
            "dict[str, object]",
            screen_arms(sample_screen(name, size=args.screen_size, seed=args.screen_seed)),
        )
        for name in SCREENED_OBJECTIVES
    }


def _screen_slice(args: argparse.Namespace) -> range:
    """Which screen seeds this process runs.

    Clipped to the screen's own seed count rather than the confirmation's, so a
    shard sharding the confirmation does not silently ask the screen for seeds
    no arm there is meant to hold.
    """
    return range(
        min(args.seed_from, args.screen_seeds),
        min(args.seed_to or args.screen_seeds, args.screen_seeds),
    )


def describe_screen(screen: Screen, *, shown: int) -> str:
    """Lay out the top of one screen, and state how little of it is resolved."""
    lines = [
        f"\n--- screen: {screen.objective} "
        f"({len(screen.ranked)} configurations x {len(screen.seeds)} seeds) ---",
        "  NOT A RESULT. These means nominate candidates and nothing else.",
    ]
    for entry in screen.ranked[:shown]:
        mark = "" if entry.separated else "within noise of the leader"
        lines.append(
            f"  {entry.name:<34} regret {entry.regret:7.4f} +-{entry.spread:6.4f}  "
            f"div {entry.diversity:6.2f}  {mark}"
        )
    tied = len(screen.indistinguishable)
    lines.append(
        f"  resolution: {tied} of {len(screen.ranked)} configurations cannot be "
        f"separated from this screen's leader at {len(screen.seeds)} seeds"
    )
    return "\n".join(lines)


def _stage_c(
    store: ResultStore,
    args: argparse.Namespace,
    wanted: set[str] | None,
    objective: Selection,
    beta: float,
) -> Selection | int:
    """Screen a joint space, then confirm a shortlist of it.

    Two halves with different standing. The screen samples many configurations
    at few seeds and produces candidates; the confirmation re-measures the
    shortlist at the full seed count through the same rule every earlier stage
    used. Nothing the screen prints is a result, and the shortlist is named on
    the command line rather than derived here, because allocating a fixed
    confirmation budget between objectives is a judgement rather than an
    arithmetic.

    Returns:
        The choice, or a process exit code when this shard cannot make one.
    """
    if objective.chosen not in SCREENED_OBJECTIVES:
        print(
            f"stage A chose {objective.chosen}, which stage C does not screen; the "
            f"incumbent would then be outside every screened space",
            file=sys.stderr,
        )
        return 2

    screened = _screen_arms(args)
    if args.stage in {"c", "c-screen", "both"}:
        shard = _shard({k: v for arms in screened.values() for k, v in arms.items()}, wanted, "C")
        if shard is None:
            return 2
        run_stage(
            "screen-joint",
            shard,
            store,
            report=args.report,
            seeds=args.screen_seeds,
            runnable=_screen_slice(args),
        )

    screens = []
    for name, arms in screened.items():
        stored = held(store, arms)
        if [n for n in arms if len(stored.get(n, {})) < args.screen_seeds]:
            _flush(f"screen shard done; {name} is not complete yet")
            return 0
        screens.append(rank_screen(name, stored))

    for screen in screens:
        _flush(describe_screen(screen, shown=args.screen_shown))
    _flush(_allocation_note(screens, args))

    if not args.confirm:
        _flush(
            "\nstage C is waiting on a shortlist: re-run with --confirm ARM per "
            "finalist. The incumbent is added regardless and does not need naming."
        )
        return 0
    return _confirm(store, args, wanted, incumbent(objective.chosen, beta))


def _allocation_note(screens: list[Screen], args: argparse.Namespace) -> str:
    """Put a proposed split of the confirmation budget on the screen.

    A proposal and nothing more: it is printed, never acted on, and the finalists
    still have to be named. Allocating slots to an objective whose screen looked
    worse is a decision about how much a ten-seed ranking is worth, and that is
    not a decision to take mechanically.
    """
    proposal = propose_allocation(screens, slots=args.confirm_slots, floor=args.confirm_floor)
    split = ", ".join(f"{name} {count}" for name, count in proposal.items())
    return (
        f"\n  proposed split of {args.confirm_slots} finalists: {split} "
        f"(floor {args.confirm_floor} each; a proposal, not a decision -- "
        f"name the finalists with --confirm)"
    )


def _confirm(
    store: ResultStore,
    args: argparse.Namespace,
    wanted: set[str] | None,
    standing: Configuration,
) -> Selection | int:
    """Re-measure the shortlist, and the incumbent, at the full seed count.

    This is the measurement. Everything before it in stage C exists to decide
    what goes into it.
    """
    try:
        configurations = confirmation_set(args.confirm, standing)
    except ValueError as error:
        print(f"--confirm names something that is not a screened arm: {error}", file=sys.stderr)
        return 2
    arms = cast("dict[str, object]", screen_arms(configurations))

    if args.stage in {"c", "c-confirm", "both"}:
        shard = _shard(arms, wanted, "C")
        if shard is None:
            return 2
        run_stage(
            "confirm-joint",
            shard,
            store,
            report=args.report,
            seeds=args.seeds,
            runnable=range(args.seed_from, args.seed_to or args.seeds),
        )
    stored = held(store, arms)
    if [n for n in arms if len(stored.get(n, {})) < args.seeds]:
        _flush("confirmation shard done; not every finalist is complete yet")
        return 0
    chosen = select(stored)
    _flush(describe("stage C: joint configuration (confirmed)", chosen))
    return chosen


def _flush(message: str) -> None:
    """Print immediately, so a long phase can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
