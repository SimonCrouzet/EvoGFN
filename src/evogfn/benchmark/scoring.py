"""Scoring a campaign that finished and measured nothing.

A campaign whose every proposal was infeasible finishes normally. It spent its
whole plate, it was not `RunRecord.exhausted`, and it measured nothing -- so its
``best`` is ``-inf`` and its ``regret`` ``+inf``. One such seed takes an arm's
mean to infinity, and ``nanmean`` does not help, because an infinity is not a
``nan``.

Dropping those seeds would report the arm's mean over the seeds it happened to
succeed on, which flatters whichever arm failed most often, and would break the
pairing every comparison here rests on. Scoring them at the landscape's worst
fitness keeps every seed in every arm, so the seed sets stay identical.

What it costs is that the mean is then part measurement and part convention.
Hence [scoring_note][evogfn.benchmark.scoring.scoring_note] beside the row and
[SCORING_RULE][evogfn.benchmark.scoring.SCORING_RULE] above the table: a mean
computed under a convention and a mean computed without one are different
numbers that print identically.

Why this module is imported by nothing in the fingerprint closure
------------------------------------------------------------------

Both readers of this rule -- ``experiments/run_suite.py`` and
``experiments/transfer_probe.py`` -- are report paths, and a report cannot
change what a finished campaign recorded. But
[ResultStore][evogfn.benchmark.store.ResultStore] fingerprints raw file bytes
over an import closure, so putting the rule in
[evogfn.benchmark.statistics][] -- which *is* in that closure -- would stale
every stored record on every edit to it.

So it lives here, imported only from ``experiments/`` and from
[evogfn.benchmark.transfer][], which is outside the closure for the same reason.
**Nothing under the closure may import this module.**
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from evogfn.benchmark.store import RunRecord

#: The value a landscape reports for a design that satisfies nothing. Both
#: landscape families here are non-negative, so an unmeasured campaign is scored
#: as though it had found the worst design rather than no design.
WORST_FITNESS = 0.0

#: Stated in the report itself, because a mean computed under a convention and a
#: mean computed without one are different numbers that print identically.
SCORING_RULE = (
    "  scoring: a campaign that proposed nothing feasible finished and measured "
    f"nothing; it is scored at the landscape's worst fitness ({WORST_FITNESS:.1f}), "
    "not dropped"
)


class WorstCase(Protocol):
    """What one metric reads for a record whose campaign measured nothing."""

    def __call__(self, record: RunRecord, metric: str) -> float | None:
        """Score one record at its worst case.

        Args:
            record: The campaign that measured nothing.
            metric: Field being scored.

        Returns:
            The score, or ``None`` when the metric has no worst case to give.
        """
        ...


def worst_case_from_attainable(lower: float | None) -> WorstCase:
    """Score against one audited attainable optimum, shared by every seed.

    For a task whose attainability is audited once rather than per anchor, which
    is every task in [evogfn.benchmark.suite][]. The probe audits per anchor and
    supplies its own reader instead.

    Args:
        lower: The task's attainable optimum, conservative end, or ``None`` when
            the task was never audited.

    Returns:
        A [WorstCase][evogfn.benchmark.scoring.WorstCase] closed over it.
    """

    def worst_case(record: RunRecord, metric: str) -> float | None:  # noqa: ARG001 - protocol
        if metric == "best":
            return WORST_FITNESS
        if metric != "regret" or lower is None:
            return None
        return float(lower) - WORST_FITNESS

    return worst_case


def scored_metric(
    records: Mapping[int, RunRecord],
    seeds: Sequence[int],
    metric: str,
    worst_case: WorstCase,
) -> tuple[np.ndarray, int, int]:
    """One metric per seed, with unmeasured seeds scored rather than dropped.

    Ordering mirrors
    [records_to_metric][evogfn.benchmark.suite.records_to_metric] exactly --
    seeds in the given order, absent and exhausted seeds skipped -- because the
    two are read side by side and a caller may swap one for the other.

    Args:
        records: Records by seed.
        seeds: The order to return them in.
        metric: Field name.
        worst_case: What to score a campaign that measured nothing.

    Returns:
        ``(values, scored, unscorable)``: the metric, how many entries were
        scored rather than measured, and how many measured nothing *and* had no
        worst case to score them at.
    """
    values: list[float] = []
    scored = 0
    unscorable = 0
    for seed in seeds:
        record = records.get(seed)
        if record is None or record.exhausted:
            continue
        raw = getattr(record, metric)
        value = float("nan") if raw is None else float(raw)
        if np.isfinite(value):
            values.append(value)
            continue
        floor = worst_case(record, metric)
        if floor is None:
            unscorable += 1
            values.append(float("nan"))
            continue
        scored += 1
        values.append(floor)
    return np.asarray(values, dtype=np.float64), scored, unscorable


def scoring_note(scored: int, unscorable: int) -> str:
    """How much of a mean is convention rather than measurement.

    Args:
        scored: Seeds scored at the worst case.
        unscorable: Seeds that measured nothing and could not be scored.

    Returns:
        A fragment, empty when every seed measured something.
    """
    parts = []
    if scored:
        parts.append(f"{scored} scored at worst")
    if unscorable:
        parts.append(f"{unscorable} unscorable")
    return f"  [{', '.join(parts)}]" if parts else ""
