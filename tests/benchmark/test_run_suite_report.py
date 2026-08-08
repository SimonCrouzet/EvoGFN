"""Tests for what ``experiments/run_suite.py`` runs, and what its report refuses to hide.

Four failures, every one of them silent in the direction of flattering a table.

**An arm that never fitted, printed as one that lost.** A supervised method that
never leaves its random-screening stage spends the same budget, fills the same
plates and reports a regret arithmetically indistinguishable from a fitted one's.
Its row is then a random baseline under a supervised method's name, and the
sentence a reader writes about it -- "we beat MLDE" -- is about a comparison
nobody ran. `RunRecord.fitted` is the only thing that separates the two, so the
report has to say it on the row, in its own line, and beside every p-value that
names such an arm.

**An over-budget arm on a tier whose axis is the budget.** ``mlde-over-budget``
runs one plate beyond its task's protocol, which is the point of the arm on the
headline tiers and is exactly what ``rounds-curve`` and ``budget-gradient``
cannot hold constant: one adds a plate to a fixed total, the other shifts the arm
one plate to the right of the point it is plotted at -- a doubling at the 96-call
rung and a 1% perturbation at the 10,000-call one. The same arm would be a
different distortion at every rung, and the curve would read as a property of the
methods.

**A control on a tier that does not hold what it controls.** ``mlde+earlyfit`` is
MLDE at a training size a constrained screen can return, and it means nothing
except beside the published arm whose starved handover it exists to explain. On
the density sweep either row alone reads as a complete answer about MLDE under
feasibility constraints, and the wrong one of the two is the row that must never
be quoted as MLDE at all.

**A ladder rung printed without its mark.** ``genetic+screen``,
``genetic+search`` and ``genetic+distinct`` are rungs, not pipelines. In a table
where some decomposition rows carry ``[ablation of ...]`` and an attribution line
beside their p-value, an unmarked one does not read as unlabelled -- it reads as a
positive claim that somebody published it.

The module under test is a script rather than a package module, so it is loaded
by path. That is deliberate: importing what the suite actually runs is the only
way these tests can fail when it changes.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from evogfn.benchmark.methods import BASELINES
from evogfn.benchmark.store import ResultStore, RunRecord
from evogfn.benchmark.suite import Purpose, Tier, budget_gradient, objective_task, rounds_curve

#: The script under test, loaded from the repository rather than from the
#: installed package -- ``experiments/`` is not importable and pinning the path
#: is what keeps this test reading the file the suite is actually run from.
_SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "run_suite.py"


def _load():
    """Import ``experiments/run_suite.py`` under its own name."""
    spec = importlib.util.spec_from_file_location("run_suite_under_test", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_suite():
    """The script, imported once."""
    return _load()


def _tier(name, task=None, seeds=(0, 1, 2)):
    """A tier under one of the suite's own names, so `methods_for` branches on it."""
    return Tier(name, (task or objective_task(),), tuple(seeds), Purpose.DIAGNOSTIC)


# --------------------------------------------------------------------------
# Which arms a tier runs.
# --------------------------------------------------------------------------


class TestTheOverBudgetArm:
    """It belongs where the budget is fixed, and nowhere the budget is the axis."""

    @pytest.mark.parametrize("name", ["rounds-curve", "budget-gradient"])
    def test_it_is_kept_off_the_tiers_whose_axis_is_the_budget(self, run_suite, name):
        arms = run_suite.methods_for(_tier(name))

        assert "mlde-over-budget" not in arms, (
            f"{name} would run an arm that adds a plate to its own protocol, so a difference "
            f"between its rungs would be partly a difference in what was spent"
        )
        # The compressed arm stays: it runs the task's own protocol, so it is a
        # comparator on this axis like any other. Dropping both would make the
        # exclusion look like a decision about MLDE rather than about budgets.
        assert "mlde" in arms

    @pytest.mark.parametrize("name", ["main", "replication", "large-space"])
    def test_it_stays_where_the_budget_is_held_fixed(self, run_suite, name):
        # It is in the suite *because* it does not fit: the headline must not
        # rest on a comparator compressed to a quarter of its published training
        # set. Excluding it everywhere would delete that row.
        assert "mlde-over-budget" in run_suite.methods_for(_tier(name))

    def test_the_named_tiers_are_tiers_the_suite_actually_builds(self, run_suite):
        # A misspelling here does not raise -- `methods_for` falls through to the
        # baseline set -- so it would silently put the arm back on the curve. The
        # names are checked against the tiers those functions produce.
        built = {task.name for task in (*rounds_curve(), *budget_gradient())}

        assert any(name.startswith("rounds-") for name in built)
        assert any(name.startswith("budget-") for name in built)
        assert set(run_suite.BUDGET_AXIS_TIERS) == {"rounds-curve", "budget-gradient"}

    def test_the_exclusion_survives_a_recorded_selection(self, run_suite, monkeypatch):
        # The selection lookup rebuilds the arm mapping from `BASELINES`, which
        # holds `mlde-over-budget`. An exclusion applied only on the untuned
        # branch would quietly stop working the day a selection was recorded.
        monkeypatch.setattr(run_suite, "selected_gflownet", lambda: {"gfn-chosen": object()})
        arms = run_suite.methods_for(_tier("rounds-curve"))

        assert "gfn-chosen" in arms
        assert "mlde-over-budget" not in arms


class TestTheConstraintDensitySweepHoldsTheSupervisedPair:
    """A control on a tier without the arm it controls reports the wrong thing."""

    def test_both_mlde_arms_run_on_the_density_axis(self, run_suite):
        # `mlde+earlyfit` exists because a dense feasibility constraint starves
        # `mlde`'s training set, so the handover never happens and a random
        # screen is tabled under a supervised method's name. Density is the axis
        # that finding is a function of, and this tier is that axis -- but the
        # two rows only separate "the method is a poor fit here" from "the method
        # never fitted" when both are present. Either one alone reads as a
        # complete answer, which is the silent part.
        arms = run_suite.methods_for(_tier("constraint-density"))

        assert {"mlde", "mlde+earlyfit"} <= set(arms)

    def test_the_over_budget_arm_stays_off_it(self, run_suite):
        # It would move a second axis. The pair above differs in the training
        # size alone; adding the arm that also spends an extra plate would make
        # a difference across the family unattributable between the two.
        assert "mlde-over-budget" not in run_suite.methods_for(_tier("constraint-density"))

    def test_every_named_arm_exists(self, run_suite):
        # `methods_for` looks these up by name out of `MAIN_METHODS`. A name that
        # stopped existing raises there rather than sweeping a smaller set, and
        # this says so before a night of compute rather than during one.
        assert set(run_suite.DENSITY_ARMS) <= set(run_suite.MAIN_METHODS)


# --------------------------------------------------------------------------
# Which rows are decompositions rather than pipelines.
# --------------------------------------------------------------------------

#: Arms whose ``+`` marks something other than an added rung, and which are
#: therefore allowed to carry no ``[ablation of ...]`` mark.
#:
#: Empty, and the emptiness is the point. `mlde+earlyfit` was here on the
#: argument that it replaces a parameter rather than adding a component, so it
#: decomposes nothing -- but what makes a row a decomposition is that reading it
#: against its base isolates one thing, and `mlde` against it isolates whether
#: the published pipeline ever reached its model. Replacing rather than adding
#: changes which sentence explains the row, which is why `_attribution` now
#: chooses one, not whether the row needs a mark.
#:
#: Kept rather than deleted so the next arm with a ``+`` in its name meets the
#: argument that was already had, instead of having it again.
NOT_DECOMPOSITIONS: frozenset[str] = frozenset()


class TestEveryLadderRungIsMarked:
    """An unmarked decomposition row among published pipelines is read as one."""

    def test_no_rung_reaches_the_table_unmarked(self, run_suite):
        # A rung is an arm named `base+something` whose base is also an arm. That
        # is the whole ladder rather than its top step, and marking only the top
        # step is worse than marking none: in a table where the other
        # decomposition rows *are* labelled, an absent label reads as a positive
        # claim that the row is a pipeline somebody published.
        rungs = {
            name: name.split("+", 1)[0]
            for name in BASELINES
            if "+" in name and name.split("+", 1)[0] in BASELINES
        }
        unmarked = {
            name
            for name, base in rungs.items()
            if name not in NOT_DECOMPOSITIONS and run_suite.ABLATIONS.get(name) != base
        }

        assert not unmarked, (
            f"{', '.join(sorted(unmarked))} would print among published pipelines with no "
            f"[ablation of ...] mark and no attribution line beside its p-value"
        )

    def test_every_marked_row_gets_a_sentence_that_is_true_of_it(self, run_suite):
        # A row explaining itself wrongly is worse than an unexplained row. The
        # surrogate sentence claims the arm separates a surrogate's contribution
        # from a sampler's, which is false of an arm that changes neither; the
        # handover sentence says the arm is ours and names no published method.
        # Both would print, and only one is true, so the wrong one would read as
        # a claim about what the row measures.
        assert "surrogate" not in run_suite._attribution("mlde+earlyfit")
        assert "never reached its model" in run_suite._attribution("mlde+earlyfit")
        assert "surrogate" in run_suite._attribution("genetic+search")

    def test_it_names_only_arms_that_exist(self, run_suite):
        # A misspelling here marks nothing and raises nothing: `ablations()` is a
        # lookup by name, so a typo silently returns the row to looking like a
        # pipeline. Both sides are checked -- the row and what it decomposes --
        # because a mark pointing at an arm that is not in the table is a
        # reference a reader cannot follow.
        assert set(run_suite.ABLATIONS) <= set(BASELINES)
        assert set(run_suite.ABLATIONS.values()) <= set(BASELINES)


# --------------------------------------------------------------------------
# What the report says about an arm that never fitted.
# --------------------------------------------------------------------------


def _record(*, method, seed, fitted, best=0.4, regret=0.6):
    """One stored campaign, enough for the report to read a row off."""
    return RunRecord(
        task=objective_task().name,
        method=method,
        seed=seed,
        protocol="P",
        best=best,
        regret=regret,
        diversity=1.0,
        feasible_fraction=1.0,
        oracle_calls=384,
        proposals=384,
        fitted=fitted,
    )


@pytest.fixture
def reported(tmp_path, run_suite):
    """A store holding one fitted arm, one that never fitted, and the reference.

    All three complete every seed and spend the same budget, which is the point:
    nothing but ``fitted`` distinguishes the arm that never ran its model from
    the one that ran it and lost.
    """
    store = ResultStore(tmp_path)
    for seed in (0, 1, 2):
        store.append(_record(method="genetic", seed=seed, fitted=None, regret=0.5))
        store.append(_record(method="mlde", seed=seed, fitted=False, regret=0.6))
        store.append(_record(method="alde", seed=seed, fitted=True, regret=0.55))

    tier = _tier("main")
    names = ["genetic", "mlde", "alde"]
    return run_suite, store, tier, names


def _lines(run_suite, store, tier, names):
    """The report, with the arm list pinned so the test does not run the suite."""
    held = {name: store.usable(tier.tasks[0].name, name) for name in names}
    rows, solved, unfitted = run_suite._arm_rows(held, names, tier.seeds, None)
    paired = run_suite._paired(
        held, names, list(tier.seeds), "genetic", solved=solved, unfitted=unfitted
    )
    return rows, paired, unfitted


class TestAnArmThatNeverFitted:
    """The row exists, carries a number, and must not read as a supervised loss."""

    def test_the_row_is_marked_rather_than_dropped(self, reported):
        # Kept, because the campaign really ran and the number is really what a
        # random screen achieved on this task. Marked, because the arm's *name*
        # claims otherwise and the name is what gets quoted.
        rows, _, unfitted = _lines(*reported)
        row = next(line for line in rows if line.strip().startswith("mlde "))

        assert "unfit=3" in row
        assert unfitted == {"mlde"}

    def test_an_arm_that_did_fit_carries_no_mark(self, reported):
        rows, _, _ = _lines(*reported)
        row = next(line for line in rows if line.strip().startswith("alde "))

        assert "unfit" not in row

    def test_an_arm_with_no_model_carries_no_mark(self, reported):
        # `None` is not a failed fit. A genetic algorithm did not fail to fit a
        # model, and marking it would put the warning on most of the table and
        # so on none of it.
        rows, _, unfitted = _lines(*reported)
        row = next(line for line in rows if line.strip().startswith("genetic "))

        assert "unfit" not in row
        assert "genetic" not in unfitted

    def test_the_report_says_it_in_words_as_well(self, reported):
        run_suite, store, tier, _ = reported
        text = run_suite.report(store, tier, reference="genetic")

        assert "NEVER FITTED" in text
        assert "random baseline under a supervised method's name" in text

    def test_the_paired_comparison_says_what_it_is_comparing(self, reported):
        # Before the p-value, like the vacuous marker: a difference against an
        # arm that never fitted is a difference against random screening, and an
        # unmarked significant line here is how "we beat MLDE" gets written about
        # a row where MLDE never ran.
        _, paired, _ = _lines(*reported)
        marked = [line for line in paired if "not a method comparison" in line]

        assert len(marked) == 1
        assert "mlde" in marked[0]

    def test_a_table_of_fitted_arms_is_left_alone(self, tmp_path, run_suite):
        # The marker has to be absent when nothing is wrong, or it is noise and
        # gets ignored on the one row that carries it.
        store = ResultStore(tmp_path)
        for seed in (0, 1, 2):
            store.append(_record(method="genetic", seed=seed, fitted=None))
            store.append(_record(method="alde", seed=seed, fitted=True))
        tier = _tier("main")

        rows, paired, unfitted = _lines(run_suite, store, tier, ["genetic", "alde"])

        assert not unfitted
        assert all("unfit" not in line for line in rows)
        assert all("not a method comparison" not in line for line in paired)
