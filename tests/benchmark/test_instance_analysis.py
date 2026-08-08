"""Tests for reading replicated tasks one instance at a time, and refusing to pool.

The failure these catch is not a crash and not a wrong number in any single
cell. It is that six tables of 100 seeds each, sitting under one tier heading,
invite exactly one summary -- their mean -- and that summary is wrong in the
direction nobody checks. Pooling instance x seed pairs as independent
observations reports a standard error several times narrower than the design
supports, so the pseudo-replicated figure does not look like a mistake: it looks
like *more* evidence than the honest reading, and it is the figure that ends up
in a slide.

Two things therefore have to hold, and neither is a convention.

**The per-instance reading has to be printed.** An analysis that exists only as
an instruction in a framing document is one nobody runs. So the report itself
carries the per-draw effects, the interval across draws at ``n = draws``, and a
sign test on how many draws agreed.

**The pooled reading has to raise.** Absent is not enough: the next person who
needs a single number writes ``np.concatenate`` at the call site and never learns
why they should not have. `pooled_metric` is the door they walk through, and it
refuses for replicate draws while working for tasks that are genuinely different
experiments.

The module under test is a script, so it is loaded by path -- importing what the
suite actually runs is the only way these tests fail when it changes.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from evogfn.benchmark.store import ResultStore, RunRecord
from evogfn.benchmark.suite import MAIN, Purpose, Tier, replicate_instance, replication

_SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "run_suite.py"

#: Seeds per draw in these fixtures. Ten rather than a hundred because every
#: assertion here is about which unit the analysis treats as independent, and
#: that is a property of the arithmetic rather than of the count.
SEEDS = tuple(range(10))


def _load():
    """Import ``experiments/run_suite.py`` under a name of this module's own."""
    spec = importlib.util.spec_from_file_location("run_suite_instance_analysis", _SCRIPT)
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


def _record(*, task, method, seed, regret):
    """One stored campaign, enough for the report to read a row off."""
    return RunRecord(
        task=task,
        method=method,
        seed=seed,
        protocol="P",
        best=1.0 - regret,
        regret=regret,
        diversity=1.0,
        feasible_fraction=1.0,
        oracle_calls=396,
        proposals=396,
        fitted=None,
    )


def _split(task):
    """A replicate task's shape and draw, asserted to exist.

    Every task these fixtures touch comes from `replication`, so ``None`` here
    would mean the naming scheme and its parser had come apart -- which is the
    failure `TestTheTaskNamesItKeysOff` pins, and which must not instead surface
    as a confusing ``TypeError`` in a fixture.
    """
    split = replicate_instance(task.name)
    assert split is not None
    return split


def _alde_draws():
    """The three ALDE replicate tasks, in draw order."""
    return [task for task in replication() if _split(task)[0] == "alde"]


def _stored(tmp_path, offsets, *, noise=0.01):
    """A store where `arm` beats `genetic` by ``offsets[i]`` on draw ``i``.

    The seed noise is small and identical across draws, so anything the analysis
    reports as between-instance variation is the offsets and nothing else --
    which is what lets a test assert on the direction of a flip rather than on a
    tolerance.
    """
    store = ResultStore(tmp_path)
    rng = np.random.default_rng(0)
    for task, offset in zip(_alde_draws(), offsets, strict=True):
        for seed in SEEDS:
            wobble = float(rng.standard_normal()) * noise
            store.append(_record(task=task.name, method="genetic", seed=seed, regret=0.7 + wobble))
            store.append(
                _record(task=task.name, method="cmaes", seed=seed, regret=0.7 - offset + wobble)
            )
    return store


def _tier():
    """The replication tier, at this module's seed count."""
    return Tier("replication", replication(), SEEDS, Purpose.BENCHMARK)


class TestTheUnitOfReplication:
    """Draws are the observations; seeds are what happens inside one."""

    def test_the_across_draw_interval_counts_draws_and_not_seeds(self, tmp_path, run_suite):
        # Thirty campaigns, three draws. If `n` came back 30 the analysis would
        # be the pooled one under a per-instance heading -- which is the worst of
        # the available outcomes, because it would look correct.
        effects = run_suite.instance_effects(
            _stored(tmp_path, [0.1, 0.1, 0.1]),
            _alde_draws(),
            "alde",
            "cmaes",
            "genetic",
            SEEDS,
        )

        assert effects.across.n == 3
        assert [outcome.n for outcome in effects.per_draw] == [10, 10, 10]

    def test_the_across_draw_interval_is_wider_than_the_pooled_one_would_be(
        self, tmp_path, run_suite
    ):
        # The whole quantitative claim, on data where the instance term is real.
        # The honest interval is built from three numbers; the pooled one would
        # be built from thirty, and would be narrower by roughly the design
        # effect. Narrower is the direction that reads as more evidence.
        effects = run_suite.instance_effects(
            _stored(tmp_path, [0.02, 0.10, 0.18]),
            _alde_draws(),
            "alde",
            "cmaes",
            "genetic",
            SEEDS,
        )
        honest = effects.across.high - effects.across.low
        within = min(outcome.high - outcome.low for outcome in effects.per_draw)

        assert honest > within

    def test_a_draw_that_disagrees_is_visible_rather_than_averaged_away(self, tmp_path, run_suite):
        # An ordering flip is a claim about *one* draw. Averaged into a grand
        # mean it disappears entirely -- two draws at +0.1 and one at -0.1 pool
        # to a comfortable win -- and the flip is the finding the replication
        # tier exists to be able to see.
        effects = run_suite.instance_effects(
            _stored(tmp_path, [0.1, 0.1, -0.1]),
            _alde_draws(),
            "alde",
            "cmaes",
            "genetic",
            SEEDS,
        )

        assert effects.agreeing == 2
        assert effects.across.mean > 0.0
        assert effects.sign_test > 0.05

    def test_one_draw_yields_no_across_draw_analysis_at_all(self, tmp_path, run_suite):
        # `None`, not an interval over one number. Something shaped like an
        # across-instance result taken from a single instance is the pooled
        # reading under another name.
        store = _stored(tmp_path, [0.1, 0.1, 0.1])

        assert (
            run_suite.instance_effects(store, _alde_draws()[:1], "alde", "cmaes", "genetic", SEEDS)
            is None
        )


class TestTheRefusalToPool:
    """It has to raise, because absence is a convention and conventions break."""

    def test_pooling_replicate_draws_raises(self, tmp_path, run_suite):
        with pytest.raises(ValueError, match="refusing to pool across landscape draws"):
            run_suite.pooled_metric(
                _stored(tmp_path, [0.1, 0.1, 0.1]), _alde_draws(), "cmaes", SEEDS
            )

    def test_the_refusal_names_what_to_use_instead(self, tmp_path, run_suite):
        # A refusal that does not name the alternative gets worked around rather
        # than obeyed, and the workaround is the pooled mean it refused.
        with pytest.raises(ValueError, match="instance_effects"):
            run_suite.pooled_metric(
                _stored(tmp_path, [0.1, 0.1, 0.1]), _alde_draws(), "cmaes", SEEDS
            )

    def test_one_draw_of_each_shape_is_not_replication_and_is_allowed(self, tmp_path, run_suite):
        # The refusal is about draws of *one* shape. Two shapes with one draw
        # each vary the protocol, not the instance, so nothing here is a
        # replicate of anything and there is nothing to refuse.
        store = _stored(tmp_path, [0.1, 0.1, 0.1])
        mixed = [_alde_draws()[0], next(t for t in replication() if "evolvepro" in t.name)]

        assert run_suite.pooled_metric(store, mixed, "cmaes", SEEDS).size == len(SEEDS)

    def test_tasks_that_are_not_replicates_pool_normally(self, tmp_path, run_suite):
        # The door has to work, or nobody walks through it and the refusal is
        # never reached. Two headline tasks are genuinely different experiments.
        store = ResultStore(tmp_path)
        tasks = list(MAIN[:2])
        for task in tasks:
            for seed in SEEDS:
                store.append(_record(task=task.name, method="genetic", seed=seed, regret=0.5))

        assert run_suite.pooled_metric(store, tasks, "genetic", SEEDS).size == 2 * len(SEEDS)


class TestWhatTheReportSays:
    """The section has to be printed, and it has to name its own limits."""

    def test_the_report_carries_the_per_instance_section(self, tmp_path, run_suite):
        text = run_suite.report(_stored(tmp_path, [0.1, 0.1, -0.1]), _tier(), reference="genetic")

        assert "per instance, with the landscape draw as the unit" in text
        assert "across 3 draws of alde" in text

    def test_the_report_says_the_pooled_figure_is_refused(self, tmp_path, run_suite):
        # In words and on the page, because a reader who does not find a pooled
        # number assumes nobody computed it rather than that it was refused.
        text = run_suite.report(_stored(tmp_path, [0.1, 0.1, 0.1]), _tier(), reference="genetic")

        assert "pooled_metric() refuses" in text

    def test_the_report_names_the_sign_test_floor_at_three_draws(self, tmp_path, run_suite):
        # The one thing that can be said about `REPLICATION_SEEDS` without
        # running anything: three draws cannot reach significance with the
        # instance as the unit at any seed count. Unsaid, three unanimous draws
        # read as a confirmed ordering.
        text = run_suite.report(_stored(tmp_path, [0.1, 0.1, 0.1]), _tier(), reference="genetic")

        assert "UNDERPOWERED BY DESIGN" in text

    def test_it_prints_how_far_pooling_would_have_been_wrong(self, tmp_path, run_suite):
        # Measured, not asserted. A factor near one would mean the objection to
        # pooling was formal; the number is what makes it quantitative.
        text = run_suite.report(_stored(tmp_path, [0.02, 0.10, 0.18]), _tier(), reference="genetic")

        assert "would understate the standard error" in text

    def test_a_tier_with_no_replicates_gets_no_section(self, tmp_path, run_suite):
        # The section has to be absent when nothing is replicated, or it is
        # noise on every table and gets skipped on the one table it matters for.
        store = ResultStore(tmp_path)
        task = MAIN[0]
        for seed in SEEDS:
            store.append(_record(task=task.name, method="genetic", seed=seed, regret=0.5))
        tier = Tier("main", (task,), SEEDS, Purpose.BENCHMARK)

        assert "per instance" not in run_suite.report(store, tier, reference="genetic")


class TestTheTaskNamesItKeysOff:
    """The grouping has to come from the suite, not from a pattern written twice."""

    def test_every_replicate_task_is_recognised(self):
        # A parser that stopped matching would put every replicate in no family,
        # which yields no per-instance section and no refusal -- the pooled
        # reading, restored silently by a rename.
        assert all(replicate_instance(task.name) is not None for task in replication())

    def test_the_families_are_the_protocol_shapes(self, run_suite):
        families = run_suite.instance_families(replication())

        assert set(families) == {"alde", "evolvepro"}
        assert all(len(drawn) == 3 for drawn in families.values())

    def test_a_headline_task_is_in_no_family(self, run_suite):
        assert not run_suite.instance_families(MAIN)
