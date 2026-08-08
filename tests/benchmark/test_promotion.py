"""Tests for how a variant-ladder rung reaches the headline table, and when it may not.

The ladder crosses two GFlowNet mechanisms on the shipped configuration and picks
one to ship. Where that pick happens decides whether the paper's configuration was
chosen on the diagnostic landscape or on the tasks that carry its claims, and the
difference is invisible in every table either way -- the numbers are the same
numbers; only the order in which they were looked at differs.

Three failures, all silent.

**A ladder that promotes itself.** If the rungs simply appeared in `main`, the
best would be picked from the headline tasks afterwards. That is tuning on the
test set, and it leaves no trace: the winning row is a real campaign with a real
regret, and nothing on the page records that four other configurations were
looked at first. So promotion must be a step somebody takes, reading only the
ladder's own task, and `main` must be unchanged until they take it.

**A promoted rung that reads as a method.** Once the rungs are in the headline
table, four of the five are decompositions of the fifth. An unmarked
decomposition row among published pipelines is read as a pipeline that lost --
and ``+wide`` is the worst of them, being a deliberately over-sized capacity
control that nobody proposed as anything. They have to carry the same marking
`genetic+search` carries, on the row and beside every p-value.

**A promotion the ladder does not support.** The rung is named by the caller,
never derived, because deriving it would make the promotion automatic again. What
the code owes in exchange is a refusal when the named rung lost, or when the base
is promoted over a rung that beat it.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evogfn.benchmark.methods import shipped_base, variant_arms
from evogfn.benchmark.store import ResultStore, RunRecord
from evogfn.benchmark.suite import Purpose, Tier, objective_task

_SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "run_suite.py"

#: Seeds the fake ladder holds. Enough for a paired comparison to be decisive on
#: noiseless data, which is what these tests need -- the statistic itself is
#: exercised elsewhere.
SEEDS = tuple(range(10))


def _load():
    """Import ``experiments/run_suite.py`` under a name of this module's own."""
    spec = importlib.util.spec_from_file_location("run_suite_promotion", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_suite(tmp_path, monkeypatch):
    """The script, with its promotion file pointed at a temporary directory.

    Re-imported per test rather than shared: the promotion is module-level state
    read back by `methods_for`, so a test that promoted would otherwise decide
    what every later test's headline tier runs.
    """
    module = _load()
    monkeypatch.setattr(module, "PROMOTION_FILE", tmp_path / "promoted.json")
    return module


@pytest.fixture
def base():
    """The ladder's base rung -- what this project currently ships."""
    return shipped_base().name


@pytest.fixture
def rungs():
    """Every rung of the ladder, base first."""
    return list(variant_arms())


def _record(*, method, seed, regret):
    """One stored ladder campaign."""
    return RunRecord(
        task=objective_task().name,
        method=method,
        seed=seed,
        protocol="P",
        best=1.0 - regret,
        regret=regret,
        diversity=1.0,
        feasible_fraction=1.0,
        oracle_calls=384,
        proposals=384,
        fitted=None,
    )


@pytest.fixture
def ladder(tmp_path, rungs, base):
    """A stored ladder where exactly one rung beats the base, decisively.

    Every rung completes every seed, so a refusal in these tests is a refusal
    about the evidence rather than about missing records.
    """
    store = ResultStore(tmp_path / "results")
    winner = f"{base}+anchor"
    for seed in SEEDS:
        for name in rungs:
            regret = 0.30 if name == winner else 0.50 + 0.001 * seed
            store.append(_record(method=name, seed=seed, regret=regret))
    return store, winner


class TestPromotionIsASeparateStep:
    """Running the ladder must change nothing about the headline table."""

    def test_the_headline_tier_is_untouched_until_somebody_promotes(self, run_suite, rungs, base):
        # The property the whole design turns on. The ladder can be complete,
        # decisive and fully stored, and `main` still runs one GFlowNet arm --
        # so the winner cannot have been picked by looking at headline results.
        arms = run_suite.methods_for(Tier("main", (), SEEDS, Purpose.BENCHMARK))

        assert run_suite.promoted_rung() is None
        assert not any(name in arms for name in rungs if name != base)

    def test_promotion_puts_every_rung_in_the_headline_path(self, run_suite, ladder, rungs):
        store, winner = ladder
        assert run_suite.promote(store, winner, report=lambda _: None) == 0

        arms = run_suite.methods_for(Tier("main", (), SEEDS, Purpose.BENCHMARK))

        assert set(rungs) <= set(arms)

    def test_the_ladder_tier_still_runs_all_five_rungs_afterwards(self, run_suite, ladder, rungs):
        # The decision has to stay auditable. A promoted ladder that dropped its
        # losing rungs would leave nothing to check the promotion against the
        # moment it was taken.
        store, winner = ladder
        run_suite.promote(store, winner, report=lambda _: None)

        arms = run_suite.methods_for(Tier(run_suite.LADDER_TIER, (), SEEDS, Purpose.DIAGNOSTIC))

        assert set(arms) == set(rungs)

    def test_it_reads_only_the_ladder_task(self, run_suite, ladder):
        # Named in the record, so a promotion argued from a headline task would
        # be visible in the file rather than only in whoever ran it. The task is
        # the diagnostic landscape, which no headline tier uses.
        store, winner = ladder
        run_suite.promote(store, winner, report=lambda _: None)
        record = json.loads(run_suite.PROMOTION_FILE.read_text())

        assert record["task"] == objective_task().name
        assert record["tier"] == run_suite.LADDER_TIER


class TestWhatPromotionRefuses:
    """Naming the rung is the caller's; supporting it is the evidence's."""

    def test_a_rung_the_ladder_did_not_support_is_refused(self, run_suite, ladder, base):
        store, _ = ladder
        lines: list[str] = []

        assert run_suite.promote(store, f"{base}+wide", report=lines.append) != 0
        assert any("REFUSED" in line for line in lines)
        assert not run_suite.PROMOTION_FILE.exists()

    def test_promoting_the_base_over_a_rung_that_beat_it_is_refused(self, run_suite, ladder, base):
        # "Neither mechanism earns its compute" is a real outcome of a ladder and
        # is allowed -- but not when a rung won. A decision contradicting the
        # tier it was taken from is not a decision.
        store, _ = ladder
        lines: list[str] = []

        assert run_suite.promote(store, base, report=lines.append) != 0
        assert any("contradicting the tier" in line for line in lines)

    def test_the_base_may_be_promoted_when_no_rung_beat_it(self, tmp_path, run_suite, rungs, base):
        store = ResultStore(tmp_path / "flat")
        for seed in SEEDS:
            for name in rungs:
                store.append(_record(method=name, seed=seed, regret=0.5 + 0.001 * seed))

        assert run_suite.promote(store, base, report=lambda _: None) == 0
        assert run_suite.promoted_rung() == base

    def test_a_name_that_is_not_a_rung_is_refused(self, run_suite, ladder):
        store, _ = ladder

        assert run_suite.promote(store, "genetic", report=lambda _: None) != 0

    def test_an_unrun_ladder_promotes_nothing(self, tmp_path, run_suite, base):
        empty = ResultStore(tmp_path / "empty")

        assert run_suite.promote(empty, f"{base}+anchor", report=lambda _: None) != 0
        assert not run_suite.PROMOTION_FILE.exists()

    def test_the_evidence_is_stored_beside_the_decision(self, run_suite, ladder):
        # A promotion whose evidence lives only in a terminal scrollback is one
        # nobody can audit, and this file is what fixes the headline table's
        # configuration.
        store, winner = ladder
        run_suite.promote(store, winner, report=lambda _: None)
        record = json.loads(run_suite.PROMOTION_FILE.read_text())

        assert record["arm"] == winner
        assert len(record["evidence"]) == len(variant_arms()) - 1


class TestARecordThatNoLongerDescribesAnything:
    """A stale promotion must raise, never fall through to the untuned path."""

    def test_an_unfinished_record_raises(self, run_suite):
        run_suite.PROMOTION_FILE.write_text(json.dumps({"arm": "x"}))

        with pytest.raises(ValueError, match="unfinished"):
            run_suite.promoted_rung()

    def test_a_rung_the_ladder_no_longer_builds_raises(self, run_suite, base):
        # Silence here would be the worst outcome available: falling back to the
        # untuned arms would put a configuration in the headline table that no
        # promotion ever chose, under a results directory that says one did.
        run_suite.PROMOTION_FILE.write_text(
            json.dumps({"arm": "gone+anchor", "base": base, "tier": "variant-ladder", "task": "t"})
        )

        with pytest.raises(ValueError, match="not a rung of the current ladder"):
            run_suite.promoted_rung()

    def test_a_promotion_taken_against_another_base_raises(self, run_suite, base):
        run_suite.PROMOTION_FILE.write_text(
            json.dumps(
                {
                    "arm": f"{base}+anchor",
                    "base": "gfn-tb",
                    "tier": "variant-ladder",
                    "task": "t",
                }
            )
        )

        with pytest.raises(ValueError, match="no longer ships"):
            run_suite.promoted_rung()


class TestThePromotedRungsAreLabelled:
    """They are decompositions of the shipped arm, on every line they appear on."""

    def test_every_rung_but_the_winner_becomes_an_ablation(self, run_suite, ladder, rungs):
        store, winner = ladder
        run_suite.promote(store, winner, report=lambda _: None)

        marked = run_suite.ablations()

        assert {name for name in rungs if name != winner} <= set(marked)
        assert all(marked[name] == winner for name in rungs if name != winner)
        # The classical ablations keep theirs. A mapping rebuilt at promotion
        # time that dropped them would unmark `genetic+search` in the same table.
        assert marked["genetic+search"] == "genetic"

    def test_the_winner_is_not_marked_as_an_ablation_of_itself(self, run_suite, ladder):
        store, winner = ladder
        run_suite.promote(store, winner, report=lambda _: None)

        assert winner not in run_suite.ablations()

    def test_the_row_carries_the_mark(self, run_suite, ladder, base):
        store, winner = ladder
        run_suite.promote(store, winner, report=lambda _: None)
        names = [winner, f"{base}+wide"]
        held = {name: store.usable(objective_task().name, name) for name in names}

        rows, _, _ = run_suite._arm_rows(held, names, SEEDS, None)
        control = next(line for line in rows if "+wide" in line)

        assert f"[ablation of {winner}]" in control
        assert "[ablation" not in next(line for line in rows if line.strip().startswith(winner))

    def test_the_paired_line_says_what_it_attributes(self, run_suite, ladder, base):
        # Beside the p-value as well as on the row. Read on its own, a rung's
        # comparison against the shipped arm is a claim about methods rather than
        # the attribution claim it actually supports.
        store, winner = ladder
        run_suite.promote(store, winner, report=lambda _: None)
        names = [winner, f"{base}+terminal"]
        held = {name: store.usable(objective_task().name, name) for name in names}

        lines = run_suite._paired(held, names, list(SEEDS), winner, solved=set(), unfitted=set())

        assert any("separates one mechanism" in line for line in lines)

    def test_an_unpromoted_ladder_marks_nothing_extra(self, run_suite):
        # The marking must be absent when nothing is promoted, or it is on most
        # of the table and therefore on none of it.
        assert set(run_suite.ablations()) == set(run_suite.ABLATIONS)
