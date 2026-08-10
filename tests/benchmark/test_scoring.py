"""A campaign that finished and measured nothing is scored, not dropped.

The failure this is about is silent by construction: an arm whose every proposal
was infeasible spends its whole plate, is not `exhausted`, and returns
``best = -inf`` / ``regret = +inf``. Before the rule below, one such seed made
the arm's mean ``inf``, its standard error ``nan`` and its paired comparison
``-inf [nan, nan] t=nan``, while the seed count went on reading as a full row.
"""

from __future__ import annotations

import numpy as np
import pytest

from evogfn.benchmark.scoring import (
    SCORING_RULE,
    WORST_FITNESS,
    scored_metric,
    scoring_note,
    worst_case_from_attainable,
)
from evogfn.benchmark.store import RunRecord, dependency_closure
from evogfn.benchmark.suite import RESULT_DEPENDENCIES, records_to_metric


def _record(*, seed, best=0.4, regret=0.6, exhausted=False, diversity=1.0):
    """One stored campaign, enough to read a metric off."""
    return RunRecord(
        task="t",
        method="m",
        seed=seed,
        protocol="P",
        best=best,
        regret=regret,
        diversity=diversity,
        feasible_fraction=1.0,
        oracle_calls=384,
        proposals=384,
        exhausted=exhausted,
    )


#: One measured seed, one that finished having measured nothing.
@pytest.fixture
def records():
    return {
        0: _record(seed=0, best=0.4, regret=0.6),
        1: _record(seed=1, best=float("-inf"), regret=float("inf")),
    }


class TestTheUnmeasuredSeedIsKept:
    def test_it_is_scored_at_the_attainable_optimum_less_the_worst_fitness(self, records):
        # `regret` is stored as `attainable - best`, so a campaign that measured
        # nothing has the regret of the worst design the landscape can return.
        # Not an arbitrary large number: it is what that seed would have scored
        # had it measured the worst thing available.
        values, scored, unscorable = scored_metric(
            records, [0, 1], "regret", worst_case_from_attainable(1.0)
        )
        assert values.tolist() == [0.6, 1.0 - WORST_FITNESS]
        assert (scored, unscorable) == (1, 0)

    def test_the_mean_is_finite_where_it_used_to_be_infinite(self, records):
        # The whole point in one assertion. `nanmean` does not rescue this,
        # because an infinity is not a `nan`.
        raw = records_to_metric(records, [0, 1], "regret")
        assert not np.isfinite(raw.mean())
        values, _, _ = scored_metric(records, [0, 1], "regret", worst_case_from_attainable(1.0))
        assert np.isfinite(values.mean())

    def test_the_seed_stays_in_so_the_comparison_stays_paired(self, records):
        # Dropping it would report the arm's mean over the seeds it happened to
        # succeed on, which flatters whichever arm failed most often -- and would
        # leave two arms with different seed sets, which is not a paired test.
        values, _, _ = scored_metric(records, [0, 1], "regret", worst_case_from_attainable(1.0))
        assert len(values) == 2

    def test_best_is_scored_at_the_worst_fitness_itself(self, records):
        values, scored, _ = scored_metric(records, [0, 1], "best", worst_case_from_attainable(1.0))
        assert values.tolist() == [0.4, WORST_FITNESS]
        assert scored == 1


class TestWhatIsNotScored:
    def test_an_exhausted_seed_is_still_skipped(self):
        # Mirrors `records_to_metric`, which the row builder calls for every
        # other column: a campaign that never finished has no measurement to
        # contribute, where one that finished measuring nothing has.
        records = {0: _record(seed=0), 1: _record(seed=1, exhausted=True)}
        values, scored, _ = scored_metric(
            records, [0, 1], "regret", worst_case_from_attainable(1.0)
        )
        assert len(values) == len(records_to_metric(records, [0, 1], "regret")) == 1
        assert scored == 0

    def test_an_unaudited_task_reports_the_seed_as_unscorable(self, records):
        # With no audited optimum there is no floor to score against, and
        # inventing one would put a number in the column that no audit supports.
        values, scored, unscorable = scored_metric(
            records, [0, 1], "regret", worst_case_from_attainable(None)
        )
        assert (scored, unscorable) == (0, 1)
        assert np.isnan(values[1])

    def test_diversity_over_an_empty_plate_has_no_worst_case(self):
        # Undefined rather than bad: a plate with nothing on it has no pairwise
        # distance, and zero would read as "every design was identical".
        records = {0: _record(seed=0, diversity=float("nan"))}
        _, scored, _ = scored_metric(records, [0], "diversity", worst_case_from_attainable(1.0))
        assert scored == 0


class TestTheConventionIsStated:
    def test_the_note_counts_both_kinds(self):
        assert scoring_note(0, 0) == ""
        assert "3 scored at worst" in scoring_note(3, 0)
        assert "2 unscorable" in scoring_note(0, 2)
        assert scoring_note(1, 1).count(",") == 1

    def test_the_rule_names_the_value_it_scores_at(self):
        # A mean computed under a convention and one computed without it print
        # identically, so the rule has to appear beside the table rather than
        # only in a docstring.
        assert f"{WORST_FITNESS:.1f}" in SCORING_RULE
        assert "not dropped" in SCORING_RULE


class TestItStaysOutOfTheFingerprintClosure:
    def test_no_stored_record_depends_on_this_module(self):
        # The reason this module exists rather than the rule living in
        # `benchmark/statistics.py`. The store fingerprints raw file bytes over
        # an import closure, so a reporting-only module inside that closure would
        # stale every stored campaign on every edit to it -- which is exactly
        # what a docstring pass over the closure once cost, at 32,062 records.
        assert "evogfn.benchmark.scoring" not in set(dependency_closure(RESULT_DEPENDENCIES))
