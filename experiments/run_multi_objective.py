"""Run the multi-objective suite, resuming whatever is already stored.

Safe to interrupt and safe to re-run, on the same terms as
``experiments/run_suite.py``: every campaign is written the moment it finishes
and a second invocation runs only what is missing.

    uv run python experiments/run_multi_objective.py                 # everything
    uv run python experiments/run_multi_objective.py --tier main     # headline only
    uv run python experiments/run_multi_objective.py --seeds 50      # raise the count
    uv run python experiments/run_multi_objective.py --report        # no runs, just read

Results land in ``results/``, under this suite's own task names. The columns do
mean something different from the single-objective suite's -- there ``best`` is a
fitness and ``regret`` a distance to an audited optimum, here ``best`` is a
hypervolume and ``regret`` is IGD+ -- but the **task name** is what carries that,
not the directory. The store keys by ``(task, method, seed)`` and every reader
names the tasks it wants, so a separate root only added a second place to look
for results.

The tiers differ in role, not just in seed count
------------------------------------------------

Only **main** carries results. ``conflict`` and ``objectives`` are *explanatory*:
they say under what conditions the main table's ranking would change, and a win
that appears at one conflict level and vanishes at another is a finding about the
sweep rather than an extra headline row. ``preferences`` is the only diagnostic
that decides anything -- how many preference vectors the main-table GFlowNet arm
should get -- and it is measured at fixed total budget, so eight preferences buys
48 assays each rather than eight full campaigns.

The report prints that role next to every tier, because a reader who takes an
explanatory sweep for a result has been misled by the layout rather than by the
numbers.

Two limitations the report states rather than hides
---------------------------------------------------

**Hypervolume goes missing exactly where it matters.** The exact method in
[evogfn.metrics.pareto][] raises past 16 front points in three or more
objectives, and an arm that converges carries a wider measured front than one
that scatters. So the column can be present for the arms that did badly and
``nan`` for the arms that did well, which is worse than useless if read as a
ranking. ``ch65-real`` is read on IGD+, and the report says how many seeds lost
their hypervolume so the gap cannot be mistaken for a run that failed.

**A reference front is only as strong as what it enumerates.** Where a task's
space is too large to enumerate -- a 20-letter alphabet is enumerable only up to
L=5 -- what stands in is the exact front over a declared set of recombinations of
the objectives' planted optima. Every point in it is attained by a sequence that
exists, so IGD+ = 0 is reachable -- but it is a *subset* of the true front, and
an arm saturating it has covered what the construction found rather than the
front. The report marks which tasks are in that position.

Every record stores its own ``cpu_seconds`` and ``wall_seconds``, so what the
suite costs is a question for the store rather than for this file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.determinism import configure_determinism, is_deterministic
from evogfn.benchmark.multi_objective import (
    ABLATIONS,
    EXACT_FRONT_LIMIT,
    SCOPE_NOTES,
    MultiObjectiveTask,
    arms_for_tier,
    multi_objective_tiers,
    run_multi_objective_tier,
)
from evogfn.benchmark.statistics import compare, seeds_needed
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import Tier, records_to_metric

if TYPE_CHECKING:
    from evogfn.benchmark.tasks import Task

#: Seeds for the headline tier. Fewer than the single-objective suite's 100: a
#: multi-objective campaign carries the same oracle budget but a GFlowNet arm at
#: eight preferences trains eight policies for it, and CH65 rebuilds a 62,926-row
#: table per campaign.
MAIN_SEEDS = 50

#: Seeds for the explanatory sweeps and the preference diagnostic.
EXPLANATORY_SEEDS = 30

#: Where results go. One root for every suite; the task name is what says what
#: this suite's indicators mean -- see the module docstring.
DEFAULT_RESULTS = "results"

#: The arm every other one is compared against: a weighted-sum genetic algorithm,
#: bare, which is the pipeline a lab already runs and therefore the only thing a
#: win has to be a win over. Directed evolution *is* a genetic algorithm, so this
#: is the incumbent rather than a strawman.
#:
#: Deliberately not ``genetic+search`` -- the GA handed this campaign's surrogate
#: and an inner loop over it -- which is a hybrid nobody published and would pair
#: every headline comparison against something no reviewer has to accept. That
#: arm is kept as a decomposition row instead, where the same convention holds as
#: in ``experiments/run_suite.py``. It still answers "was it the surrogate or the
#: constructive sampler?", which is the first question a reviewer asks -- but
#: that is an attribution question, and attribution belongs in a decomposition
#: row rather than in the yardstick.
REFERENCE_ARM = "genetic"

#: Tiers that cannot use `REFERENCE_ARM` because they do not contain it, and what
#: they use instead. Mirrors ``experiments/run_suite.py``'s ``REFERENCES`` and
#: exists for the reason that one does: `report` prints "nothing is paired" when
#: the reference is absent, and a diagnostic whose whole content is a comparison
#: between its own arms would print that every time.
#:
#: The preference sweep is paired against **one** preference -- the configuration
#: the main table would run if the diagnostic came back saying the split does not
#: pay -- so each row reads as what splitting the budget bought.
REFERENCES = {"preferences": "gfn-tb-pref1"}

#: What a tier is for, printed next to it. Kept here rather than on
#: [Tier][evogfn.benchmark.suite.Tier] because it is a property of this suite's
#: reading, and `Tier.headline` already carries the part that is structural.
TIER_ROLES = {
    "main": "carries results",
    "conflict": "explanatory: says when the ranking changes, not what it is",
    "objectives": "explanatory: says what the objective count itself costs",
    "preferences": "decides the main table's preference count, and nothing else",
}

# Two observations are the fewest a paired comparison can be drawn from.
_MIN_PAIRED = 2


def _task_note(task: Task) -> str:
    """State what this task's indicators can and cannot say.

    Args:
        task: The task being reported on.

    Returns:
        One or two lines naming the reference point, whether the reference front
        is exact, and where hypervolume is at risk of not being computable.
    """
    if not isinstance(task, MultiObjectiveTask):
        return "  not a multi-objective task; nothing here applies to it"
    point = ", ".join(f"{value:g}" for value in task.reference_point)
    front = task.reference_front()
    if front is None:
        against = "no reference front, so IGD+ is unreported"
    else:
        quality = "an exact" if task.front_is_exact else "a constructed subset of the true"
        against = f"IGD+ against {quality} front of {front.shape[0]} point(s)"
    lines = [f"  hypervolume from ({point}); {against}"]
    if task.n_objectives > 2 and front is not None:  # noqa: PLR2004 - two objectives sweep exactly
        lines.append(
            f"  note: exact hypervolume needs a measured front of at most "
            f"{EXACT_FRONT_LIMIT} points at {task.n_objectives} objectives; "
            f"seeds above it record nan and are counted below"
        )
    if not task.front_is_exact:
        lines.append(
            "  note: IGD+ = 0 here means the constructed front was covered, which is "
            "weaker than covering the true front"
        )
    return "\n".join(lines)


def report(store: ResultStore, tier: Tier, reference: str | None = None) -> str:
    """Read the store and compare every arm against one, paired across seeds.

    Hypervolume and IGD+ rather than best and regret, because that is what the
    records hold -- see
    [set_indicators][evogfn.benchmark.multi_objective.set_indicators]. The
    comparison is drawn on **IGD+**, and on IGD+ alone: hypervolume is missing
    wherever the exact method could not run, and a paired test over a column
    whose absences correlate with an arm's quality would be reading the absences.

    Two things are printed that the numbers cannot say for themselves, both
    mirroring `experiments/run_suite.py`:

    * **which rows are ablations.** A decomposition row sitting unmarked among
      published pipelines is read as one of them, and its p-value against the
      reference is then read as a ranking of methods rather than as the
      attribution it is. Marked on the row *and* under its paired outcome,
      because either is read alone.
    * **where an arm's name overclaims.** ``gfn-tb-scalar`` is single-preference
      GFlowNet-AL over a fixed scalarisation, and a reader who has met MOGFN-PC
      will assume a preference-conditioned policy unless told otherwise. See
      [SCOPE_NOTES][evogfn.benchmark.multi_objective.SCOPE_NOTES].

    Args:
        store: Where results live.
        tier: The tier to report on.
        reference: Arm to compare against, or ``None`` to take whichever one
            this tier is read against -- `REFERENCE_ARM`, a published pipeline,
            except on a tier that does not contain it. See `REFERENCES`.

    Returns:
        A multi-line report.
    """
    role = TIER_ROLES.get(tier.name, "unstated")
    lines = [f"\n=== {tier!r} -- {role}"]
    names = list(arms_for_tier(tier))
    if reference is None:
        reference = REFERENCES.get(tier.name, REFERENCE_ARM)
    for task in tier.tasks:
        lines.append(f"\n{task!r}")
        lines.append(_task_note(task))
        held = {name: store.usable(task.name, name) for name in names}
        seeds = [s for s in tier.seeds if all(s in held[n] for n in names if held[n])]
        for name in names:
            records = held[name]
            if not records:
                continue
            volume = records_to_metric(records, tier.seeds, "best")
            coverage = records_to_metric(records, tier.seeds, "regret")
            spread = records_to_metric(records, tier.seeds, "diversity")
            spent = records_to_metric(records, tier.seeds, "oracle_calls")
            uncomputed = int(np.isnan(volume).sum())
            finite = volume[np.isfinite(volume)]
            error = coverage.std(ddof=1) / len(coverage) ** 0.5 if len(coverage) > 1 else 0.0
            # On the row itself, not in a footnote: a decomposition row sitting
            # unmarked among published pipelines is read as one of them.
            mark = f"  [ablation of {ABLATIONS[name]}]" if name in ABLATIONS else ""
            lines.append(
                f"  {name:<18} igd+ {np.nanmean(coverage):>7.4f} +/- {error:<7.4f} "
                f"hv {(finite.mean() if finite.size else float('nan')):>9.4f} "
                f"(nan on {uncomputed}/{len(volume)})  "
                f"div {np.nanmean(spread):>5.2f}  spent {np.nanmean(spent):>6.0f}  "
                f"n={len(coverage)}{mark}"
            )
            if note := SCOPE_NOTES.get(name):
                lines.append(f"        scope: {note}")

        base = held.get(reference)
        if not base or not seeds:
            # Said rather than silently omitted: an absent section reads as
            # "nothing separated these arms", which is a different statement
            # from "nothing was tested".
            lines.append(f"  reference {reference} has no usable seeds here, so nothing is paired")
            continue
        lines.append(f"  paired on IGD+ vs {reference} (positive favours the first):")
        for name in names:
            if name == reference or not held[name]:
                continue
            mine = records_to_metric(held[name], seeds, "regret")
            theirs = records_to_metric(base, seeds, "regret")
            if len(mine) != len(theirs) or len(mine) < _MIN_PAIRED:
                continue
            # Lower IGD+ is better, so this is a loss like regret is.
            outcome = compare(name, mine, theirs, higher_is_better=False)
            lines.append(f"    {outcome!r}")
            # Marked here as well as in the table above: read on its own, an
            # ablation's line is a p-value against a published pipeline, and that
            # is a claim about methods rather than the attribution it supports.
            if decomposes := ABLATIONS.get(name):
                lines.append(
                    f"        attribution: decomposes {decomposes}; separates the "
                    f"surrogate's contribution from the sampler's, and ranks no "
                    f"method a lab could run"
                )
            if not outcome.significant and (needed := seeds_needed(outcome)):
                lines.append(f"        inconclusive; ~{needed} seeds would resolve this")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run or report on the multi-objective suite.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code: 0 on success, 2 when nothing matched the selection,
        3 when threading is not pinned and a run was asked for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", action="append", help="Run only these tiers.")
    parser.add_argument(
        "--task",
        action="append",
        help="Run only these tasks. Sharding by task is race-free -- the store "
        "keeps one file per task and method -- so a process per task uses the "
        "cores far better than threads do, most of the work being serial Python.",
    )
    parser.add_argument("--seeds", type=int, default=MAIN_SEEDS, help="Seeds for the main tier.")
    parser.add_argument(
        "--explanatory-seeds",
        type=int,
        default=EXPLANATORY_SEEDS,
        help="Seeds for the sweeps and the preference diagnostic.",
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

    store = ResultStore(Path(args.results))
    selected = multi_objective_tiers(args.seeds, args.explanatory_seeds)
    if args.tier:
        selected = [t for t in selected if t.name in set(args.tier)]
    if args.task:
        wanted = set(args.task)
        selected = [
            Tier(t.name, tuple(k for k in t.tasks if k.name in wanted), t.seeds, t.purpose)
            for t in selected
        ]
        selected = [t for t in selected if t.tasks]
    if not selected:
        print(f"nothing matched tier={args.tier} task={args.task}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    for tier in selected:
        if not args.report:
            ran = run_multi_objective_tier(tier, arms_for_tier(tier), store, report=_flush)
            _flush(f"{tier.name}: ran {ran} campaigns")
        _flush(report(store, tier))
    _flush(f"\ntotal {time.perf_counter() - started:.0f}s")
    _flush(store.summarise())
    return 0


def _flush(message: str) -> None:
    """Print immediately, so an overnight run can be watched while it runs."""
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
