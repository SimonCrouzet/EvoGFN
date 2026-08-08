"""Tests for the five mechanism rungs in the headline table, and what they must not read as.

The rungs of [variant_arms][evogfn.benchmark.methods.variant_arms] are reported
in ``main`` beside the baselines and **nothing is picked from them**: the shipped
arm stays the base, and the other four say what each mechanism does on the tasks
that carry the claim. That decision replaced a separate ``variant-ladder`` tier
and an explicit promotion step, and every failure below is a way of half-making
the change -- each one leaves a table that looks right.

**A rung that is not there at all.** ``methods_for`` branches on the tier's name
and a misspelling does not raise: it falls through to the baseline set, so the
headline table would hold one GFlowNet arm and look exactly as it did before
anybody decided to report the mechanisms. Nothing in the output says a study is
missing, because an unrun row prints no row.

**The shipped arm in the table twice.** The base rung *is* the selected arm,
resolved from the same file. If the ladder's builder and the selection's ever
disagree about its name, the table holds one configuration under two names, pays
a second hundred campaigns per task for it, and reports the pair as two arms --
which is the single thing the selection phase exists to leave the headline
holding exactly one of.

**A rung printed without its mark, or with the wrong sentence.** Four of the five
decompose the arm beside them and none is a pipeline anybody published; ``+wide``
is a deliberately over-sized capacity control that exists only to make ``+anchor``
attributable. In a table where the classical decomposition rows *are* labelled,
an unmarked rung reads as a positive claim that somebody published it -- and a
rung carrying the surrogate sentence would claim to separate a surrogate's
contribution from a sampler's, which is false of an arm that changes neither.

**A mechanism read against the wrong arm.** The table's reference is `genetic`,
a published pipeline. A rung's difference from `genetic` is not what the rung
measures -- ``+anchor`` is one step from the shipped configuration and nothing
else -- so without a block of its own the rung's attribution line would sit
directly beneath a p-value taken against a different arm.

The module under test is a script rather than a package module, so it is loaded
by path. That is deliberate: importing what the suite actually runs is the only
way these tests can fail when it changes.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evogfn.benchmark.methods import shipped_base, variant_arms
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.store import ResultStore, RunRecord
from evogfn.benchmark.suite import Purpose, Tier, objective_task, run_tier
from evogfn.benchmark.tasks import Task
from evogfn.landscapes.ehrlich import EhrlichLandscape

_SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "run_suite.py"

#: Seeds the fake table holds. Enough for a paired comparison to be drawn on
#: noiseless data, which is all these tests need -- the statistic itself is
#: exercised elsewhere.
SEEDS = tuple(range(10))


def _load():
    """Import ``experiments/run_suite.py`` under a name of this module's own."""
    spec = importlib.util.spec_from_file_location("run_suite_mechanisms", _SCRIPT)
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


@pytest.fixture
def base():
    """The ladder's base rung -- the configuration this project ships."""
    return shipped_base().name


@pytest.fixture
def rungs():
    """Every rung, base first."""
    return list(variant_arms())


def _tier(name, seeds=SEEDS, task=None):
    """A tier under one of the suite's own names, so `methods_for` branches on it."""
    return Tier(name, (task or objective_task(),), tuple(seeds), Purpose.BENCHMARK)


def _unconstrained_task(name="unconstrained"):
    """A task whose landscape publishes no transition matrix.

    The structural shape of `gb1-anchor` and `trpb-anchor` -- the two tasks where
    ``+terminal`` has nothing to defer -- built here rather than loaded, since
    those two need their datasets and what is under test is the *derivation*, not
    the datasets. An Ehrlich instance always carries a matrix, so the absence has
    to be modelled by removing the attribute the environment reads, which is
    exactly what the empirical landscapes do by never having it.
    """

    class Unconstrained(EhrlichLandscape):
        """The same landscape with no constructibility rule to publish."""

        @property
        def transition_matrix(self):
            """Nothing, as on a landscape whose tokens have no adjacency rule."""
            return None

    return Task(
        name=name,
        purpose="a landscape with no transition matrix, as the empirical tasks are",
        build=lambda: Unconstrained(
            sequence_length=8, vocab_size=20, n_motifs=1, motif_length=2, seed=4
        ),
        protocol=Protocol(rounds=2, batch_size=8, max_mutations=2),
        max_mutations=2,
        attainable=None,
    )


def _record(*, method, seed, regret, task=None):
    """One stored campaign, enough for the report to read a row off."""
    return RunRecord(
        task=(task or objective_task()).name,
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


class TestTheRungsAreInTheHeadlineTier:
    """Reported, not promoted: they are there without anybody taking a step."""

    def test_every_rung_runs_in_main(self, run_suite, rungs):
        # The whole change in one assertion. It has to hold with no file written
        # and no flag passed, because "the rungs appear once somebody promotes"
        # is exactly the design this replaced.
        assert set(rungs) <= set(run_suite.methods_for(_tier(run_suite.MAIN_TIER)))

    def test_main_adds_exactly_the_four_rungs_that_are_not_the_shipped_arm(
        self, run_suite, rungs, base
    ):
        # The base rung *is* the selected arm, so the tier gains four arms and
        # not five. Read against `replication`, which runs the same baselines and
        # the same selected arm and none of the rungs: the difference between the
        # two tiers is the study, and a fifth entry there would be the shipped
        # configuration under a second name.
        headline = run_suite.methods_for(_tier(run_suite.MAIN_TIER))
        without = run_suite.methods_for(_tier("replication"))

        assert base in headline
        assert base in without
        assert set(headline) - set(without) == set(rungs) - {base}

    def test_a_builder_that_renamed_the_shipped_arm_raises(self, run_suite, monkeypatch):
        # The silent version of this is the bad one: the ladder's base and the
        # selection's arm are the same configuration resolved twice, and if the
        # two names drift the table holds both and reports them as two arms.
        # Nothing downstream could tell -- they are two legitimate store cells.
        monkeypatch.setattr(run_suite, "shipped_base", lambda: SimpleNamespace(name="gfn-drifted"))

        with pytest.raises(ValueError, match="two configurations of one method"):
            run_suite.methods_for(_tier(run_suite.MAIN_TIER))

    @pytest.mark.parametrize("name", ["replication", "large-space", "rounds-curve"])
    def test_no_other_tier_pays_for_them(self, run_suite, rungs, base, name):
        # A bounded cost, decided rather than drifted into: four extra arms on
        # `main` is 2,000 campaigns, and the same four on `replication` would be
        # 2,400 more to replicate a decomposition rather than a claim. If they
        # spread to the default path silently, the first sign would be a machine
        # running for a week.
        arms = run_suite.methods_for(_tier(name))

        assert not (set(rungs) - {base}) & set(arms)

    def test_the_tier_name_it_branches_on_is_a_tier_the_suite_builds(self, run_suite):
        # A misspelt `MAIN_TIER` does not raise. It falls through to the baseline
        # set, and the headline table then looks exactly as it did before the
        # mechanisms were reported -- with no rows missing, because an arm that
        # never ran prints nothing.
        built = {tier.name for tier in run_suite.tiers(10, 5)}

        assert run_suite.MAIN_TIER in built

    def test_the_selection_step_is_gone_rather_than_dormant(self, run_suite):
        # Nothing is picked from these rows, so there is no rung to promote and
        # no file that decides which one ships. Machinery left behind would say
        # otherwise: a `--promote` in the help text is a reader's evidence that
        # the headline configuration is still chosen somewhere.
        assert not hasattr(run_suite, "promote")
        assert not hasattr(run_suite, "PROMOTION_FILE")
        assert "variant-ladder" not in {tier.name for tier in run_suite.tiers(10, 5)}


class TestTheRungsAreMarkedAsDecompositions:
    """Four of the five decompose the arm beside them, on every line they print on."""

    def test_every_rung_but_the_base_decomposes_the_shipped_arm(self, run_suite, rungs, base):
        marked = run_suite.ablations()

        assert {name for name in rungs if name != base} <= set(marked)
        assert all(marked[name] == base for name in rungs if name != base)

    def test_the_shipped_arm_is_not_an_ablation_of_itself(self, run_suite, base):
        assert base not in run_suite.ablations()

    def test_the_classical_rows_keep_their_marks(self, run_suite):
        # The mapping is rebuilt from the ladder on every call, and one that
        # returned only the rungs would unmark `genetic+search` in the same table.
        assert run_suite.ablations()["genetic+search"] == "genetic"

    def test_a_rung_gets_the_mechanism_sentence_and_not_the_surrogate_one(self, run_suite, base):
        # A row explaining itself wrongly is worse than an unexplained row: a
        # rung claiming to separate a surrogate's contribution from a sampler's
        # would be describing an arm that changes neither.
        sentence = run_suite._attribution(f"{base}+wide")

        assert "separates one mechanism" in sentence
        assert "nothing is selected" in sentence
        assert "surrogate" not in sentence

    def test_the_row_carries_the_mark(self, tmp_path, run_suite, base):
        store = ResultStore(tmp_path)
        names = [base, f"{base}+wide"]
        for seed in SEEDS:
            for name in names:
                store.append(_record(method=name, seed=seed, regret=0.5))
        held = {name: store.usable(objective_task().name, name) for name in names}

        rows, _, _ = run_suite._arm_rows(held, names, SEEDS, None)

        assert f"[ablation of {base}]" in next(line for line in rows if "+wide" in line)
        # And the shipped arm is not marked, or the mark is on every GFlowNet row
        # in the table and therefore on none of them.
        assert "[ablation" not in next(
            line for line in rows if line.strip().startswith(base) and "+wide" not in line
        )


class TestTheMechanismsArePairedAgainstTheShippedArm:
    """Against the base, in a block of its own, or the attribution line lies."""

    @pytest.fixture
    def measured(self, tmp_path, base):
        """A store where one rung beats the shipped arm and a baseline sits beside it."""
        store = ResultStore(tmp_path)
        rung = f"{base}+anchor"
        for seed in SEEDS:
            store.append(_record(method=base, seed=seed, regret=0.50 + 0.001 * seed))
            store.append(_record(method=rung, seed=seed, regret=0.30))
            store.append(_record(method="genetic", seed=seed, regret=0.70))
        names = ["genetic", base, rung]
        held = {name: store.usable(objective_task().name, name) for name in names}
        return held, names, rung

    def test_the_block_pairs_against_the_base(self, run_suite, measured, base):
        held, names, rung = measured

        lines = run_suite._mechanism_pairs(held, names, list(SEEDS), solved=set(), unfitted=set())

        assert any(f"paired vs {base}" in line for line in lines)
        assert any(line.strip().startswith(rung) for line in lines)
        # The published pipelines belong in the table's own paired section and
        # not here: a `genetic` line under this heading would read as a mechanism
        # of the shipped arm, which is the one thing it is not.
        assert not any("genetic" in line for line in lines)

    def test_the_rung_line_carries_its_attribution(self, run_suite, measured):
        held, names, _ = measured

        lines = run_suite._mechanism_pairs(held, names, list(SEEDS), solved=set(), unfitted=set())

        assert any("separates one mechanism" in line for line in lines)

    def test_an_unrun_study_says_so_rather_than_printing_an_empty_block(
        self, tmp_path, run_suite, base
    ):
        # `_paired` skips an arm with no records, which is right in the main
        # table and wrong under a heading of its own: a block containing nothing
        # but its own header reads as four mechanisms that separated nothing,
        # when what happened is that none of them has run.
        store = ResultStore(tmp_path)
        for seed in SEEDS:
            store.append(_record(method=base, seed=seed, regret=0.5))
        names = [base, f"{base}+anchor"]
        held = {name: store.usable(objective_task().name, name) for name in names}

        lines = run_suite._mechanism_pairs(held, names, list(SEEDS), solved=set(), unfitted=set())

        assert any("unrun study" in line for line in lines)

    def test_a_tier_without_rungs_prints_no_block(self, tmp_path, run_suite):
        # Keyed off the arms present rather than off the tier's name, so the
        # block follows the rungs wherever they are reported -- and stays absent
        # everywhere else, where a heading over nothing would be noise.
        store = ResultStore(tmp_path)
        for seed in SEEDS:
            store.append(_record(method="genetic", seed=seed, regret=0.5))
        held = {"genetic": store.usable(objective_task().name, "genetic")}

        assert not run_suite._mechanism_pairs(
            held, ["genetic"], list(SEEDS), solved=set(), unfitted=set()
        )


class TestTheRungsThatCannotBeMeasuredAreReproducedRatherThanRun:
    """Two of the four rungs have nothing to act on where nothing constrains order.

    ``+terminal`` defers the feasibility rule from every intermediate state to the
    stop action, and on a landscape with no transition matrix there is no rule to
    defer: the two environments describe the identical graph, so the rung *is* the
    base arm's campaign and ``+terminal+anchor`` is ``+anchor``'s. Four failures
    are possible here and each one leaves a table that reads correctly.

    **Running them.** 400 campaigns, some 28 core-hours, spent recomputing rows the
    table already holds -- and two pairs of identical rows a reader may quote as
    "we tested the mechanism here and it made no difference", which is a different
    and false claim from "there was nothing here to test".

    **Storing the copy.** A stored copy is indistinguishable from a measurement the
    moment it is written: the same cell shape, the same fingerprint, the same
    seeds. Every reader of the store afterwards counts it as an independent
    campaign, and marking it only moves the problem to whichever reader forgets to
    honour the mark.

    **Dropping the row.** An empty cell in a headline table is an absence a reader
    fills in as they like, including as the sharpest result on the page.

    **Deciding it from a list of task names.** The list keeps saying the same thing
    after a task gains a transition matrix, at which point the mechanism is
    measurable and the row is a copy of an arm it no longer equals.
    """

    @pytest.fixture
    def loose(self):
        """A task where the mechanism has nothing to defer."""
        return _unconstrained_task()

    @pytest.fixture
    def measured(self, tmp_path, loose, base):
        """A store holding only the two arms the reproductions come from."""
        store = ResultStore(tmp_path)
        for seed in SEEDS:
            store.append(_record(method=base, seed=seed, regret=0.50, task=loose))
            store.append(_record(method=f"{base}+anchor", seed=seed, regret=0.30, task=loose))
        return store

    def test_the_two_terminal_rungs_are_named_as_reproductions(self, run_suite, loose, base):
        assert run_suite._reproduced_on(loose) == {
            f"{base}+terminal": base,
            f"{base}+terminal+anchor": f"{base}+anchor",
        }

    def test_a_task_that_constrains_construction_reproduces_nothing(self, run_suite):
        # Derived from the environment, so the diagnostic instance -- which has a
        # matrix -- measures every rung as before. A list of task names would give
        # the same answer today and the wrong one after a task changed.
        assert run_suite._reproduced_on(objective_task()) == {}

    def test_they_are_not_run(self, tmp_path, run_suite, loose, base):
        # The campaigns themselves. Each rung is a methodology that raises if it
        # is ever called, so a tier that reaches one fails here rather than
        # quietly spending 28 core-hours reproducing numbers it already has.
        def refuse(task, seed):
            raise AssertionError(f"reproduced rung was run on {task.name} at seed {seed}")

        arms = {f"{base}+terminal": refuse, f"{base}+terminal+anchor": refuse}
        lines: list[str] = []

        ran = run_tier(
            _tier(run_suite.MAIN_TIER, task=loose),
            arms,
            ResultStore(tmp_path),
            report=lines.append,
            omit=run_suite._reproduced_on,
        )

        assert ran == 0
        # Said on the line where the campaign would have been, because a campaign
        # that silently does not happen is indistinguishable from one nobody asked
        # for.
        assert any("not run" in line and base in line for line in lines)

    def test_nothing_is_stored_for_them(self, run_suite, measured, loose, base):
        run_suite.report(measured, _tier(run_suite.MAIN_TIER, task=loose))

        # The whole of the store decision. A copy on disk is a measurement to
        # everything that reads the store afterwards, and no marker survives every
        # reader; the report is where the copy belongs because it is where the
        # claim is made.
        assert not measured.usable(loose.name, f"{base}+terminal")
        assert not measured.usable(loose.name, f"{base}+terminal+anchor")

    def test_the_row_appears_in_italic_with_the_arm_it_copies(
        self, run_suite, measured, loose, base
    ):
        text = run_suite.report(measured, _tier(run_suite.MAIN_TIER, task=loose))
        row = next(line for line in text.splitlines() if f"*{base}+terminal*" in line)

        assert f"[reproduced from {base}]" in row
        # And the arm it copies is not itself marked, or the marking is on the
        # whole block and distinguishes nothing.
        assert not any(f"*{base}*" in line for line in text.splitlines() if "+terminal" not in line)

    def test_the_legend_says_what_italic_means(self, run_suite, measured, loose, base):
        text = run_suite.report(measured, _tier(run_suite.MAIN_TIER, task=loose))

        assert "REPRODUCTIONS, not measurements" in text
        assert f"{base}+terminal = {base}" in text
        # The sentence that separates the two readings, in as many words.
        assert "identical graph" in text

    def test_the_copy_is_not_paired_against_anything(self, run_suite, measured, loose):
        # A copy's comparison against a third arm is its source's comparison to
        # the last digit, and against the source itself every paired difference is
        # exactly zero -- a statistic with no standard error, which would print as
        # a failed test rather than as a comparison that never existed.
        text = run_suite.report(measured, _tier(run_suite.MAIN_TIER, task=loose))
        lines = [line for line in text.splitlines() if "vs" in line and "not paired" in line]

        assert lines
        assert all("not paired" in line for line in lines)
        # Against the source itself the refusal names the degeneracy rather than
        # the redundancy: the two are different reasons and one wording would be
        # wrong about one of them.
        assert any("exactly zero by construction" in line for line in lines)
        assert any("already printed under the name of the arm" in line for line in lines)

    def test_a_stored_copy_is_a_contradiction_and_raises(self, run_suite, measured, loose, base):
        # The loud failure. Records under a rung this suite reproduces mean either
        # the task gained a transition matrix -- so the rung is measurable and the
        # reproduction is stale -- or something wrote a copy into the store.
        # Silently preferring either one is worse than stopping: one prints an
        # independent-looking row the suite believes cannot exist, the other
        # discards a measurement somebody paid for.
        for seed in SEEDS:
            measured.append(_record(method=f"{base}+terminal", seed=seed, regret=0.4, task=loose))

        with pytest.raises(ValueError, match="Both claims cannot hold"):
            run_suite.report(measured, _tier(run_suite.MAIN_TIER, task=loose))

    def test_the_mechanism_block_is_not_a_block_of_copies(self, run_suite, measured, loose, base):
        # A block whose only content is reproductions is the "unrun study" case
        # with numbers in it, and worse for being readable: four rows that agree
        # perfectly, under a heading promising a decomposition.
        held = {
            name: measured.usable(loose.name, name)
            for name in (base, f"{base}+terminal", f"{base}+terminal+anchor")
        }
        copies = run_suite._reproduce(measured, loose, held)

        lines = run_suite._mechanism_pairs(
            held, list(held), list(SEEDS), solved=set(), unfitted=set(), reproduced=copies
        )

        assert any("unrun study" in line for line in lines)
