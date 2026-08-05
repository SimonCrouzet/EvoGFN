"""Run the transfer probe, resuming whatever is already stored.

The pre-declared specification is ``notes/experiments/transfer-probe.md``. What
the probe tests, what passes it, and what kills it were all fixed there before
any number arrived, and nothing in this file may be read as changing them. The
implementation and its five declared deviations live in
[evogfn.benchmark.transfer][]; the deviation a reader of these numbers must know
about is repeated below, because a deviation only visible in a docstring is not
visible.

    uv run python experiments/transfer_probe.py                  # everything
    uv run python experiments/transfer_probe.py --level near     # one distance
    uv run python experiments/transfer_probe.py --seeds 100      # the full count
    uv run python experiments/transfer_probe.py --report         # no runs, just read

Safe to interrupt and safe to re-run, on the same terms as
``experiments/run_suite.py``: every plate is written the moment it lands and a
second invocation runs only what is missing. **Resume is at the seed, not at the
arm.** A seed with even one arm outstanding pays for its campaigns at anchor A
again, because a trained policy cannot be reconstructed from a stored number --
so the cheapest way to add arms is to add them before running, and the cheapest
way to add seeds is simply to raise the count.

Results land under ``results-transfer/`` -- a different root from both other
suites, on purpose. ``best`` here is the best value in **one frozen plate at a
moved anchor**, not the best of a four-round campaign, and ``regret`` is taken
against *that anchor's* audited optimum rather than a task's. Sharing a directory
would let one table be built across both and nothing in a record would say the
columns meant different things.

## The screening deviation, stated where the numbers are read

Every arm at anchor B is screened by **one** surrogate, fitted at anchor A on
random-mutagenesis data at a training campaign's own oracle budget and generated
by none of the policies under test. The specification instead has each
transferred arm carry its own and leaves `plain-fresh` and `random` with none.
Three reasons, and the third is the one that decides it:

* screening has to be identical across arms or a difference in plate quality is
  not attributable to the proposer, which is the only thing the probe measures;
* no arm's own model can be the shared one, because a model fitted on a policy's
  trajectories judges that policy's proposals in-distribution and would screen
  its pool better;
* `plain-fresh` and `random` have no campaign at A and so no model to carry.
  Every per-arm scheme has to special-case them -- and criterion 1 is measured
  against `plain-fresh`, so the special case would sit exactly where a quiet
  inflation of the headline would live.

The cost is honest and points the right way: a surrogate fitted on random data
discriminates less, so every plate is noisier, the effect is smaller, and the
criteria are **harder** to pass.

## What these rows may not be used for

One landscape, one instance, one frozen round. This is a mechanism
demonstration, and it licenses no comparative claim about method ranking -- that
needs the resampled instances. Two things follow for anyone reading the table:

* Every arm at B is screened, **including the classical baselines and
  `random`**, which is not what any of their papers do. These are proposal
  distributions under a common screen, not the published pipelines
  [evogfn.benchmark.methods][]'s ``BASELINES`` reports.
* `plain-wide-transferred` is **not** parameter-matched to
  `conditioned-transferred`. No integer width matches, so it carries 237
  parameters more -- which makes criterion 3 a test against a better-resourced
  control, and is the safe direction, but it is not a match and must not be
  described as one.

## The margin in criterion 1 is not settled here

The specification asks for a 95% interval excluding zero **and** a margin of at
least 0.02, which reads two ways: ``significant and mean >= 0.02``, or the
stricter ``low >= 0.02``. They give different verdicts at exactly the effect
sizes this probe is likely to produce, so the reading has to be fixed before the
numbers land -- which is what the specification exists to force, and which this
script deliberately does not do on anybody's behalf. `report` prints the paired
comparisons and both readings' inputs; it declares no verdict.

Worth knowing while reading them: with ``c=2, k=4, q=4`` a measured value is a
multiple of 1/16, so no individual seed's regret can differ by 0.02 at all. A
mean difference of that size is a statement about what fraction of seeds cross a
quantisation step, and the win/tie/loss counts are the more trustworthy figure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.statistics import compare, seeds_needed
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import records_to_metric
from evogfn.benchmark.transfer import (
    ARM_NAMES,
    LEVELS,
    run_transfer_probe,
    task_name,
)

#: Seeds the specification declares. Paired: the same instance, the same anchor
#: pair and the same screening surrogate across every arm within a seed.
DEFAULT_SEEDS = 100

#: Where results go. Deliberately not ``results/`` or ``results-mo/``: see the
#: module docstring on what ``best`` and ``regret`` mean here.
DEFAULT_RESULTS = "results-transfer"

#: What criterion 1 is measured against -- the floor for "did anything transfer".
#: Named here because `report` pairs everything against it, and because reading a
#: transfer claim against anything else would answer a different question.
REFERENCE_ARM = "plain-fresh"

#: The arm criterion 2's interaction is measured against: a population that
#: re-founds itself at the new anchor. The mechanism's own prediction is that the
#: policy's advantage over *this* grows with distance.
DISTANCE_REFERENCE = "genetic-rebuilt"

#: The arm criterion 3 excludes capacity against. Not parameter-matched; see the
#: module docstring.
CAPACITY_CONTROL = "plain-wide-transferred"

# Two observations are the fewest a paired comparison can be drawn from.
_MIN_PAIRED = 2


def report(store: ResultStore, seeds: list[int], reference: str = REFERENCE_ARM) -> str:
    """Read the store and pair every arm against one, at each distance.

    Reports and decides nothing. The pass criteria are three conjuncts declared
    in the specification, and evaluating them here would put the verdict in the
    same file as the run -- which is how a criterion comes to be read after the
    numbers rather than before them. What this prints is their inputs: the paired
    regret comparison at each level, the win/tie/loss counts the specification
    asks to lead on, and the distribution of achieved anchor distances.

    Args:
        store: Where results live.
        seeds: Seeds to read.
        reference: Arm to pair against.

    Returns:
        A multi-line report.
    """
    lines = ["\n=== transfer probe -- diagnostic; licenses no method ranking"]
    for level in LEVELS:
        task = task_name(level)
        held = {name: store.usable(task, name) for name in ARM_NAMES}
        shared = [s for s in seeds if all(s in held[n] for n in ARM_NAMES if held[n])]
        lines.append(f"\n{task}  ({len(shared)} paired seeds)")
        lines.append(_anchor_note(held.get(reference, {})))
        lines.append(_screen_note(held))

        for name in ARM_NAMES:
            records = held[name]
            if not records:
                continue
            regret = records_to_metric(records, seeds, "regret")
            best = records_to_metric(records, seeds, "best")
            spread = records_to_metric(records, seeds, "diversity")
            error = regret.std(ddof=1) / len(regret) ** 0.5 if len(regret) > 1 else 0.0
            lines.append(
                f"  {name:<24} regret {np.nanmean(regret):>7.4f} +/- {error:<7.4f} "
                f"best {np.nanmean(best):>7.4f}  div {np.nanmean(spread):>5.2f}  "
                f"n={len(regret)}"
            )

        base = held.get(reference)
        if not base or len(shared) < _MIN_PAIRED:
            # Said rather than omitted: an absent section reads as "nothing
            # separated these arms", which is not the same as "nothing was run".
            lines.append(f"  reference {reference} has too few paired seeds here")
            continue
        lines.append(f"  paired on regret vs {reference} (positive favours the first):")
        for name in ARM_NAMES:
            if name == reference or not held[name]:
                continue
            mine = records_to_metric(held[name], shared, "regret")
            theirs = records_to_metric(base, shared, "regret")
            if len(mine) != len(theirs) or len(mine) < _MIN_PAIRED:
                continue
            outcome = compare(name, mine, theirs, higher_is_better=False)
            lines.append(f"    {outcome!r}")
            if not outcome.significant and (needed := seeds_needed(outcome)):
                lines.append(f"        inconclusive; ~{needed} seeds would resolve this")
    lines.append(_interaction(store, seeds))
    return "\n".join(lines)


def _anchor_note(records: dict[int, object]) -> str:
    """The distribution of achieved anchor distances, which the spec asks for.

    Nominal distances are what was requested; achieved distances are what the
    walk produced, and the specification's own threat list asks for the second.
    The **minimum** is the number that certifies the far level's claim, so it is
    printed rather than left to be inferred from a mean.

    Args:
        records: Stored records for one arm, by seed.

    Returns:
        One line.
    """
    distances = [
        float(record.parameters["achieved_distance"])  # type: ignore[attr-defined]
        for record in records.values()
        if "achieved_distance" in record.parameters  # type: ignore[attr-defined]
    ]
    if not distances:
        return "  no anchor distances recorded"
    return (
        f"  achieved anchor distance: min {min(distances):.0f}, "
        f"mean {sum(distances) / len(distances):.1f}, max {max(distances):.0f}"
    )


def _screen_note(held: dict[str, dict[int, object]]) -> str:
    """Whether every arm of every seed really did share one screen.

    The declared deviation is checkable from the store, and this is where it gets
    checked: within a seed, all ten arms must carry the same surrogate digest. A
    line reporting otherwise means the arms were screened by different models and
    the comparison between them is not the one the probe claims to be making.

    Args:
        held: Stored records by arm name and seed.

    Returns:
        One line.
    """
    by_seed: dict[int, set[str]] = {}
    dataset: set[str] = set()
    for records in held.values():
        for seed, record in records.items():
            parameters = record.parameters  # type: ignore[attr-defined]
            by_seed.setdefault(seed, set()).add(str(parameters.get("surrogate_dataset_digest")))
            dataset.add(str(parameters.get("surrogate_dataset")))
    if not by_seed:
        return "  no screening provenance recorded"
    split = sorted(seed for seed, digests in by_seed.items() if len(digests) > 1)
    if split:
        return f"  WARNING: seeds {split} did not share one screen; do not compare their arms"
    return f"  one screen per seed, fitted on {sorted(dataset)}"


def _interaction(store: ResultStore, seeds: list[int]) -> str:
    """Criterion 2's quantity: does the advantage grow with distance?

    A difference of differences, paired on seed: the policy's regret advantage
    over `DISTANCE_REFERENCE` at the far anchor, less the same advantage at the
    near one. This is the mechanism's own prediction and the part a population
    cannot produce, so it is computed as one paired quantity rather than left as
    two separate comparisons a reader has to subtract by eye.

    Args:
        store: Where results live.
        seeds: Seeds to read.

    Returns:
        One or more lines.
    """
    policy = "conditioned-transferred"
    advantage = {}
    for level in LEVELS:
        task = task_name(level)
        mine = store.usable(task, policy)
        theirs = store.usable(task, DISTANCE_REFERENCE)
        shared = [s for s in seeds if s in mine and s in theirs]
        if len(shared) < _MIN_PAIRED:
            return f"\ninteraction: too few paired seeds at {level} to compute it"
        advantage[level] = records_to_metric(theirs, shared, "regret") - records_to_metric(
            mine, shared, "regret"
        )
    if len(advantage["near"]) != len(advantage["far"]):
        return "\ninteraction: the two levels do not share a seed set, so nothing is paired"
    outcome = compare(
        f"{policy} vs {DISTANCE_REFERENCE}: far advantage - near advantage",
        advantage["far"],
        advantage["near"],
    )
    return (
        f"\ncriterion 2 (the advantage grows with distance), paired difference of "
        f"differences:\n    {outcome!r}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run or report on the transfer probe.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code: 0 on success, 2 when nothing matched the selection,
        3 when threading is not pinned and a run was asked for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        action="append",
        choices=list(LEVELS),
        help="Run only these distance levels. Sharding by level is not race-free "
        "on its own: both levels of one seed are served by the same campaigns at "
        "anchor A, so a process per level pays for those twice.",
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="How many seeds.")
    parser.add_argument(
        "--first-seed",
        type=int,
        default=0,
        help="First seed to run. Sharding by seed range is race-free -- the store "
        "takes an exclusive lock per append -- and is the way to fill a machine.",
    )
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="Where to store results.")
    parser.add_argument("--report", action="store_true", help="Report without running.")
    args = parser.parse_args(argv)

    # Before any tensor work: a multithreaded matmul sums in thread-completion
    # order, and a few hundred gradient steps turn that into a different design.
    configure_determinism()
    if not args.report and not is_deterministic():
        print("refusing to run: threading is not pinned", file=sys.stderr)
        return 3

    levels = args.level or list(LEVELS)
    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    if not seeds:
        print(f"nothing to run: {args.seeds} seeds", file=sys.stderr)
        return 2

    store = ResultStore(Path(args.results))
    started = time.perf_counter()
    if not args.report:
        ran = run_transfer_probe(store, seeds, levels=levels, report=_flush)
        _flush(f"ran {ran} arm-rounds in {time.perf_counter() - started:.0f}s")
    _flush(report(store, seeds))
    _flush(store.summarise([task_name(level) for level in levels]))
    return 0


def _flush(message: str) -> None:
    """Print immediately, so an overnight run can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
