"""Tests for how the suite's tasks are configured, and what a run stores.

Three failures are under test here, and all three had already happened.

**A campaign that cannot move.** Every task anchored its search at the wild type
for its whole life, so four rounds of a four-mutation budget reached four
mutations rather than sixteen and the planted optimum was outside the search
space by construction. `test_a_reanchored_campaign_moves_its_anchor` is the
end-to-end check that the mechanism is actually wired through the methodologies,
and `test_the_optimum_is_reachable_under_the_configured_budget` re-derives, per
task, the audit numbers the suite now declares rather than trusting them.

**A regret against a target nothing could reach.** `run_task` stored
``landscape.optimum - best``, which on ``large-space`` was 95% floor. The tests
here pin the stored number to the *attainable* optimum instead, and pin the
absence of one where no audit exists -- an unaudited task storing no regret is
the behaviour, not an oversight.

**A declaration that drifts from the landscape.** A declared bound above the
landscape's own optimum claims a design scoring above the maximum, and a
declared bound below what an arm reaches makes every regret on the task
negative. Both are refused rather than reported.

**A campaign that failed, stored as nothing at all.** `run_task` caught the
exhaustion, printed a line and moved on, so an arm that could not run anywhere on
a task left *no record* -- and an empty cell in a table is an absence any reader
may fill in, including as that table's sharpest result. `TestAnExhaustedCampaign`
pins the record it now writes: that it exists, that it carries no measurement
dressed up as a zero, that it says how far the campaign got, and that the readers
which average do not quietly average it in.

The heavier re-derivations are marked ``slow``. ``large-space`` is not among
them at all: its beam search at L=256 runs for minutes, and the place to
re-check it is ``experiments/audit_optima.py``, which exists for exactly that.
"""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from evogfn.algorithms.base import Sampler
from evogfn.benchmark.attainable import attainable_optimum, reanchored_attainable
from evogfn.benchmark.methods import BASELINES, OBJECTIVES, anchor_arms
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import (
    CAPPED,
    CONSTRAINT_DENSITIES,
    DIAGNOSTIC_DENSITY,
    MAIN,
    REPLICATION_SEEDS,
    anchor_study,
    budget_gradient,
    constraint_density,
    fixed_anchor_task,
    objective_task,
    records_to_metric,
    replication,
    rounds_curve,
    run_task,
)
from evogfn.benchmark.tasks import Attainable, Task
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.loop.campaign import Campaign

#: A landscape small enough to run a real campaign against inside a test, and
#: chosen so that random mutagenesis improves on the parent more than once in
#: four rounds. That last part is not incidental: the campaign moves its anchor
#: only on a *strict* improvement, and an Ehrlich reward is a product of
#: quantised terms and so flat across most of a neighbourhood, which means an
#: arm can run a whole campaign without the anchor ever moving. A toy where that
#: happened would make this file pass while testing nothing.
TOY = {
    "sequence_length": 16,
    "vocab_size": 4,
    "n_motifs": 1,
    "motif_length": 4,
    "quantization": 4,
    "max_spacing": 2,
    "transition_density": 0.5,
    "seed": 2,
}

#: Tasks whose re-anchoring audit is affordable in a test. ``large-space`` is
#: excluded by size rather than by choice; ``gb1-anchor`` has no planted optimum
#: to march to and no anchor to move.
AUDITABLE = ("feasibility", "protocol-alde", "protocol-evolvepro")


def toy_landscape() -> EhrlichLandscape:
    """Build the shared toy instance."""
    return EhrlichLandscape(**TOY)  # type: ignore[arg-type]


def toy_task(*, reanchor: bool, attainable: Attainable | None, rounds: int = 4) -> Task:
    """A task cheap enough to run end to end.

    Args:
        reanchor: Whether the anchor follows the best design measured so far.
        attainable: What to declare the search space contains.
        rounds: Design-build-test-learn cycles.

    Returns:
        The task.
    """
    return Task(
        name="toy",
        purpose="a toy, for testing that the wiring does what the table says",
        build=toy_landscape,
        protocol=Protocol(rounds=rounds, batch_size=16, max_mutations=4),
        max_mutations=4,
        reanchor=reanchor,
        attainable=attainable,
    )


def nominal(task: Task) -> float:
    """The landscape's own optimum, which regret used to be measured against."""
    optimum = task.landscape().optimum
    assert optimum is not None
    return float(np.max(optimum))


# --------------------------------------------------------------------------
# The declaration: consistent with the landscape, and complete.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [*MAIN, *budget_gradient(), *rounds_curve(), objective_task()],
    ids=lambda t: t.name,
)
def test_every_task_declares_what_it_can_reach(task):
    # The gap this closes: a task with no declaration stores regret against the
    # landscape's optimum, which is the failure the whole mechanism replaces.
    # Absence is allowed by the type and must not be allowed by the suite.
    assert task.attainable is not None, (
        f"{task.name} declares no attainable optimum, so every regret stored on it would be "
        f"against a target nobody has checked is reachable"
    )
    audited = task.attainable_optimum(nominal(task))
    assert audited is not None
    assert audited.lower <= audited.upper <= audited.nominal


@pytest.mark.parametrize(
    "task",
    [*MAIN, *budget_gradient(), *rounds_curve(), objective_task()],
    ids=lambda t: t.name,
)
def test_the_search_budget_is_cumulative_only_where_the_anchor_moves(task):
    rounds = task.protocol.rounds
    expected = task.max_mutations * rounds if task.reanchor else task.max_mutations
    assert task.search_budget == expected
    # The whole point of the fix, stated as an assertion: a re-anchored campaign
    # reaches further than one round, and a fixed one never does.
    if task.reanchor and rounds > 1:
        assert task.search_budget > task.max_mutations


def test_the_capped_phrase_is_the_one_the_audit_greps_for():
    # ``experiments/audit_optima.py`` decides "capped on purpose" from
    # "DEFECT" by looking for this substring in a task's purpose. Rewording
    # `CAPPED` without it would turn every Ehrlich task in the suite into a
    # reported defect, and the audit exits non-zero on one.
    assert "mutation budget is deliberately capped" in CAPPED


def test_a_declaration_above_the_landscapes_optimum_is_refused():
    # A bound claiming a design scores above the maximum is a broken audit, and
    # a broken bound reads exactly like a fact once it is in a table.
    task = toy_task(reanchor=False, attainable=Attainable.exactly(2.0, "wishful"))
    with pytest.raises(ValueError, match="above the landscape's own optimum"):
        task.attainable_optimum(1.0)


def test_a_half_declared_interval_is_refused():
    with pytest.raises(ValueError, match="both ends of an interval or as neither"):
        Attainable(lower=0.5, upper=None, source="x")


def test_an_inverted_interval_is_refused():
    with pytest.raises(ValueError, match="disagrees with itself"):
        Attainable.between(0.9, 0.5, "x")


def test_a_bound_without_provenance_is_refused():
    with pytest.raises(ValueError, match="how it was measured"):
        Attainable.exactly(0.5, "  ")


def test_deferring_to_the_landscape_means_no_regret_floor():
    resolved = Attainable.whole_optimum("nothing is out of reach").resolve(
        task="t", budget=4, nominal=8.76
    )
    assert resolved.is_exact
    assert resolved.regret_floor == (0.0, 0.0)


# --------------------------------------------------------------------------
# The audit, re-derived rather than trusted.
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", AUDITABLE)
def test_the_optimum_is_reachable_under_the_configured_budget(name):
    """What the task says it can reach, measured at the shape it actually runs.

    Not "is the planted optimum inside the radius": the radius is per round now
    and is deliberately smaller than that distance on every Ehrlich task here.
    The question is whether the *campaign* -- its rounds, its radius and its
    anchor rule together -- can construct a design attaining the declared value.
    """
    task = next(t for t in MAIN if t.name == name)
    declared = task.attainable_optimum(nominal(task))
    assert declared is not None

    if task.reanchor:
        measured = reanchored_attainable(
            task, per_round=task.max_mutations, rounds=task.protocol.rounds
        )
    else:
        measured = attainable_optimum(task)

    # The searched bound is witnessed by a design the environment built, so it
    # is a claim about reachability rather than about the search's luck.
    assert measured.lower >= declared.lower - 1e-9, (
        f"{name} declares it can reach {declared.lower} and the audit reached only "
        f"{measured.lower}; the declaration is optimistic"
    )
    assert measured.upper <= declared.upper + 1e-9, (
        f"{name} declares an upper bound of {declared.upper} and the audit certifies "
        f"{measured.upper}; the declaration is looser than what was proved"
    )


@pytest.mark.slow
def test_the_diagnostics_reach_their_optimum_at_the_shared_radius():
    # Seven diagnostics, one landscape, one radius. Measured at the fewest rounds
    # any of them runs, since rounds only add reach -- so this one run covers
    # the lot without paying for it seven times.
    task = objective_task()
    fewest = min(t.protocol.rounds for t in (*budget_gradient(), *rounds_curve(), objective_task()))
    measured = reanchored_attainable(task, per_round=task.max_mutations, rounds=fewest)
    declared = task.attainable_optimum(nominal(task))
    assert declared is not None
    assert measured.lower >= declared.lower - 1e-9


# --------------------------------------------------------------------------
# The mechanism, end to end through a real methodology.
# --------------------------------------------------------------------------


def test_a_reanchored_campaign_moves_its_anchor():
    # The measurement the whole of job one exists to produce. `anchor_distance`
    # flat at zero across every round is what a campaign that re-searched one
    # Hamming ball for its whole life looked like, and it looked like nothing at
    # all in the report.
    campaign = BASELINES["random"](toy_task(reanchor=True, attainable=None), 0)
    trace = campaign.run().anchor_trace()

    assert len(trace) == 4
    assert trace[0] == 0, "the first round searches from the wild type by definition"
    assert all(later >= earlier for earlier, later in itertools.pairwise(trace)), (
        f"anchor distance went backwards: {trace}; the anchor only moves on an improvement"
    )
    assert max(trace) > 0, f"the anchor never moved: {trace}"


def test_a_fixed_anchor_campaign_never_moves():
    campaign = BASELINES["random"](toy_task(reanchor=False, attainable=None), 0)
    result = campaign.run()
    assert result.anchor_trace() == [0, 0, 0, 0]
    # And the environment is the object it was handed, not a re-anchored copy.
    assert campaign.environment is not None
    assert np.array_equal(campaign.environment.parent, toy_landscape().feasible_sequence(0))


def test_a_reanchored_campaign_outreaches_a_fixed_one():
    # Same sampler, same seed, same radius: the only difference is whether the
    # anchor was allowed to follow the ledger.
    moved = BASELINES["random"](toy_task(reanchor=True, attainable=None), 0).run()
    fixed = BASELINES["random"](toy_task(reanchor=False, attainable=None), 0).run()
    wild_type = toy_landscape().feasible_sequence(0)

    def furthest(result):
        return int((np.asarray(result.sequences) != wild_type[None, :]).sum(axis=1).max())

    assert furthest(moved) > 4, "a re-anchored campaign should leave the first Hamming ball"
    assert furthest(fixed) <= 4, "a fixed anchor cannot measure anything outside its own ball"


@pytest.mark.parametrize("name", ["random", "hill-climb", "genetic", "cmaes", "mlde"])
def test_every_baseline_can_follow_a_moved_anchor(name):
    # The campaign refuses at construction when a sampler can neither be informed
    # of a move nor rebuilt for it, so this is the check that every methodology
    # supplies a factory -- and it is a construction-time check because that is
    # when the refusal happens, before a quarter of the budget has been spent.
    campaign = BASELINES[name](toy_task(reanchor=True, attainable=None, rounds=2), 0)
    assert campaign.environment is not None


# --------------------------------------------------------------------------
# What a run stores: regret against the attainable optimum, or nothing.
# --------------------------------------------------------------------------


def stored(tmp_path, task):
    """Run one arm on one seed and read back the record it wrote."""
    store = ResultStore(tmp_path)
    run_task(task, {"random": BASELINES["random"]}, store, [0], report=lambda _: None)
    return store.load(task.name, "random")[0]


def test_regret_is_stored_against_the_attainable_optimum(tmp_path):
    # 0.5 rather than 1.0, so the two candidate targets are far apart and the
    # test cannot pass by coincidence.
    declared = Attainable.exactly(0.5, "declared for the test")
    record = stored(tmp_path, toy_task(reanchor=True, attainable=declared))

    assert record.regret == pytest.approx(0.5 - record.best)
    assert record.regret != pytest.approx(
        nominal(toy_task(reanchor=True, attainable=declared)) - record.best
    )


def test_an_arm_at_the_attainable_optimum_has_no_regret_left(tmp_path):
    # A task an arm has exhausted reports zero or below, which is what makes
    # "solved" detectable in the report rather than something to be inferred
    # from a small number.
    floor = Attainable.exactly(0.0, "everything is at least this good")
    record = stored(tmp_path, toy_task(reanchor=True, attainable=floor))
    assert record.regret is not None
    assert record.regret <= 0.0


def test_a_real_campaign_stores_what_it_cost(tmp_path):
    """Guards a cost column that is present in the schema and empty in the data.

    Both clocks default to zero, which is what makes an old record loadable and
    what makes a field nobody wired up indistinguishable from a free campaign.
    So this runs an actual campaign rather than stamping a record by hand: the
    only thing it can pass on is `run_task` measuring and passing the numbers
    through.
    """
    record = stored(tmp_path, toy_task(reanchor=True, attainable=None))

    assert record.cpu_seconds > 0.0
    assert record.wall_seconds > 0.0
    # Elapsed time is the outer bound: processor time is measured inside it, and
    # can only exceed it if the campaign ran threads in parallel, which the toy
    # does not. Reversed would mean the two clocks had been swapped -- which no
    # single-clock check could catch, and which would make the comparable figure
    # the contended one.
    #
    # The tolerance is proportional rather than a fixed microsecond, and that is
    # not slack for its own sake. This toy runs in under ten milliseconds, so a
    # scheduler hiccup of a fraction of a millisecond -- which is what a loaded
    # machine delivers, and this suite is routinely run beside a dozen campaigns
    # -- exceeds an absolute epsilon while saying nothing about which clock is
    # which. A swap, the failure actually under test, puts the two figures on
    # opposite sides by the whole duration and is caught by any tolerance well
    # below 1. What was here before failed on four separate runs today, none of
    # them a swap.
    assert record.cpu_seconds <= record.wall_seconds * 1.25 + 1e-3


def test_an_arm_that_repairs_its_decode_stores_how_often_it_had_to(tmp_path):
    """Guards the attribution field reading zero for an arm that repairs constantly.

    The number decides whose result a CMA-ES score is. Its decoder is a
    per-position argmax over a relaxation that cannot express a transition
    constraint, so on a constrained task the raw decode is unbuildable and a
    projection chooses the design instead; at a repaired share near one, every
    design credited to the method was selected by that projection subject to the
    method's preferences, which is a different sentence from the one a reader
    would otherwise write.

    A default of zero is what makes the failure silent: an unwired field and an
    arm whose relaxation found the constructible set unaided are the same record.
    So this runs a real campaign on a constrained task rather than stamping a
    record, and the only way it passes is `run_task` reading the counter off the
    sampler the campaign finished with.
    """
    store = ResultStore(tmp_path)
    task = toy_task(reanchor=True, attainable=None)
    run_task(task, {"cmaes": BASELINES["cmaes"]}, store, [0], report=lambda _: None)
    record = store.load(task.name, "cmaes")[0]

    # Strictly above zero, not merely in range: zero is the default an unwired
    # field returns, so a bounds check would pass on exactly the bug this is
    # here to catch. The toy carries a transition constraint, and a separable
    # Gaussian over per-position logits cannot represent one -- so a raw decode
    # that satisfied it would be luck, and never satisfying it is the expected
    # behaviour rather than a defect.
    assert record.repaired_fraction > 0.0
    assert record.repaired_fraction <= 1.0


def test_an_arm_that_decodes_nothing_stores_a_zero_repaired_share(tmp_path):
    """The other half: zero must mean "decodes no relaxation", not "unmeasured".

    Read by attribute, so a sampler that carries no such counter stores a plain
    zero and the column stays comparable. Without this the field would be read
    as a repair rate for every arm in the table.
    """
    record = stored(tmp_path, toy_task(reanchor=True, attainable=None))

    assert record.repaired_fraction == 0.0


def test_a_campaign_that_reports_no_duplicates_stores_a_zero_share(tmp_path):
    """Guards the duplicate share landing as `None` or `nan` rather than absent.

    The number is read off the campaign result by attribute, so an arm running
    against a campaign that does not report it must store a plain zero: a share
    of nothing, which is the honest reading and is comparable down the column.
    """
    record = stored(tmp_path, toy_task(reanchor=True, attainable=None))

    assert record.duplicate_fraction is not None
    assert 0.0 <= record.duplicate_fraction <= 1.0


def test_an_unaudited_task_stores_no_regret(tmp_path):
    # Absence rather than a plausible wrong number: a regret against an
    # unaudited optimum is indistinguishable from a real one once stored.
    record = stored(tmp_path, toy_task(reanchor=True, attainable=None))
    assert record.regret is None
    assert np.isfinite(record.best)


#: A sparse Ehrlich instance, on which rejection sampling actually gives up.
#: The point of running the shipped ``genetic-feasible`` arm against it rather
#: than a stub is that the whole finding is about a real arm on a real
#: constraint: a stub can be made to exhaust, but only a real one can show
#: *why* it exhausted.
SPARSE = {
    "sequence_length": 16,
    "vocab_size": 8,
    "n_motifs": 1,
    "motif_length": 4,
    "quantization": 4,
    "max_spacing": 2,
    "transition_density": 0.05,
    "seed": 2,
}


def sparse_task() -> Task:
    """A task whose feasible set is too thin for rejection sampling."""
    return Task(
        name="sparse",
        purpose="a constraint sparse enough that rejection sampling stalls",
        build=lambda: EhrlichLandscape(**SPARSE),  # type: ignore[arg-type]
        protocol=Protocol(rounds=4, batch_size=16, max_mutations=4),
        max_mutations=4,
        reanchor=False,
        attainable=None,
    )


def test_a_rejection_arm_stores_the_draws_that_explain_its_cost(tmp_path):
    """Guards three counters that are in the schema and empty in the data.

    They default to zero, which makes an unwired column and an arm that rejected
    nothing the same record. So this runs the shipped rejection arm rather than
    stamping one: the only way it passes is `run_task` reading the counters off
    the sampler the campaign finished with.

    ``draws_unmutated`` is the one that matters. Without it the rejection rate is
    the whole story and it reads as a cost the method is absorbing; with it, the
    draws that survived turn out to be overwhelmingly the anchor unchanged --
    admitted because an unmutated design is trivially reachable, and worth
    nothing because it is the design the arm started from.
    """
    store = ResultStore(tmp_path)
    task = toy_task(reanchor=True, attainable=None)
    run_task(
        task, {"genetic-feasible": BASELINES["genetic-feasible"]}, store, [0], report=lambda _: None
    )
    record = store.load(task.name, "genetic-feasible")[0]

    assert record.draws_attempted > 0
    assert 0 < record.draws_rejected < record.draws_attempted
    assert record.draws_unmutated > 0
    assert record.draws_unmutated <= record.draws_attempted - record.draws_rejected


def test_an_arm_that_rejects_nothing_stores_zero_draws(tmp_path):
    # Zero has to mean "runs no rejection loop", not "rejected nothing": read by
    # attribute, so an arm carrying no such counter stores a plain zero and the
    # column stays comparable down its length.
    record = stored(tmp_path, toy_task(reanchor=True, attainable=None))

    assert (record.draws_attempted, record.draws_rejected, record.draws_unmutated) == (0, 0, 0)


def test_the_arm_that_gives_up_keeps_the_numbers_that_explain_why(tmp_path):
    """The two halves of this fix, together, on the case that motivated both.

    A real rejection arm on a real sparse constraint, stored rather than
    discarded, and carrying the counters that say what went wrong. Every
    assertion here is about something that used to be absent: the record itself,
    and -- once there is a record -- the reason the rejection rate on it is not
    the alarming number.

    The counters have to be incremented *before* the sampler raises. The one run
    that gives up is the run whose numbers explain the failure, and accounting
    written after the raise is accounting that never happens on it.
    """
    store = ResultStore(tmp_path)
    task = sparse_task()
    run_task(
        task, {"genetic-feasible": BASELINES["genetic-feasible"]}, store, [0], report=lambda _: None
    )
    record = store.load(task.name, "genetic-feasible")[0]

    assert record.exhausted
    assert record.draws_attempted > 0
    assert record.draws_rejected > 0
    # The finding, as an assertion: most of what survived the filter was the
    # anchor itself. Were this small, the rejection rate beside it would be the
    # whole story and the story would be that rejection is coping.
    admitted = record.draws_attempted - record.draws_rejected
    assert record.draws_unmutated > admitted / 2


class OneDesign(Sampler):
    """A sampler holding exactly one design, which is how a campaign exhausts.

    Nothing exotic about the mechanism: the campaign refuses to re-measure a
    design an earlier round already assayed, so an arm that can only produce one
    thing fills its first plate and can never fill a second. That is the same
    terminal condition a rejection GA reaches on a sparse feasible set, reached
    in a tenth of a second and without depending on any particular landscape
    being hard.
    """

    def __init__(self, length):
        super().__init__()
        self._length = length

    def propose(self, n):
        self._count(n)
        return np.zeros((n, self._length), dtype=np.int32)


def exhausting(task, seed):  # noqa: ARG001 - this arm cannot vary with its seed
    """An arm that completes round zero and then has nothing left to propose."""
    landscape = task.landscape()
    return Campaign(
        landscape=landscape,
        sampler=OneDesign(landscape.sequence_length),
        rounds=task.protocol.rounds,
        batch_size=task.protocol.batch_size,
        pool_size=task.protocol.batch_size,
    )


class TestAnExhaustedCampaign:
    """The record a run that could not finish leaves behind.

    Every failure here is an absence, and an absence is what nothing downstream
    can catch: a missing record reads as a seed nobody ran, a zeroed metric reads
    as a campaign that measured badly, and an exhausted seed folded into a mean
    reads as a result.
    """

    @pytest.fixture
    def task(self):
        # A declared optimum, so that a regret *could* have been computed here.
        # Against a task declaring none, `regret is None` would pass for the
        # wrong reason and this would test nothing.
        return toy_task(reanchor=False, attainable=Attainable.exactly(1.0, "declared for the test"))

    @pytest.fixture
    def stored_run(self, tmp_path, task):
        store = ResultStore(tmp_path)
        ran = run_task(task, {"stuck": exhausting}, store, [0], report=lambda _: None)
        return store, ran

    def test_the_seed_is_stored_rather_than_vanishing(self, stored_run, task):
        store, ran = stored_run
        held = store.load(task.name, "stuck")

        assert 0 in held, "an arm that exhausted left no record, so it is absent from the store"
        assert held[0].exhausted
        # Counted as run, because it was: it cost a campaign's worth of time and
        # is now held. A return of zero on a pass that wrote records is the one
        # thing this number is read for.
        assert ran == 1

    def test_the_failure_is_not_run_again_on_the_next_sweep(self, stored_run, task):
        # Stored and current, like any other record. Were it stale or unkeyed,
        # every later sweep would pay for the same failure again -- and worse,
        # the seed would keep reading as missing, which is the state this record
        # exists to end.
        store, _ = stored_run
        assert store.missing(task.name, "stuck", [0]) == []

    def test_it_holds_no_measurement_dressed_up_as_a_number(self, stored_run, task):
        # `nan` and `None`, never zero. Zero is a measurement -- "found nothing
        # good", "built nothing constructible" -- and it averages into a column
        # beside real ones without anything to mark it, which is the failure this
        # record was added to prevent rather than a milder form of it.
        record = stored_run[0].load(task.name, "stuck")[0]

        assert np.isnan(record.best)
        assert record.regret is None
        assert np.isnan(record.diversity)
        assert np.isnan(record.feasible_fraction)
        assert record.top_sequences == []

    def test_it_says_how_far_the_campaign_actually_got(self, stored_run, task):
        # A run that gave up in round four having measured 288 designs and one
        # that gave up in round one having measured none are different findings,
        # and a record that zeroed both would report them identically. The toy
        # fills exactly one plate before it runs out of designs.
        record = stored_run[0].load(task.name, "stuck")[0]

        assert len(record.rounds) == 1
        assert record.oracle_calls == task.protocol.batch_size
        assert record.proposals >= task.protocol.batch_size
        assert len(record.trace) == 1
        assert record.wall_seconds > 0.0

    def test_a_reader_that_averages_leaves_it_out(self, stored_run, task):
        # The half of the fix that is easy to miss. Storing the failure puts a
        # `nan` where every table reads a number, and one exhausted seed would
        # otherwise turn a whole arm's mean into `nan` while the seed count went
        # on claiming a full row. The exclusion is deliberate and the callers
        # report the count separately.
        failed = stored_run[0].load(task.name, "stuck")[0]
        finished = replace(failed, seed=1, exhausted=False, best=0.5, regret=0.5)
        held = {0: failed, 1: finished}

        assert records_to_metric(held, [0, 1], "best") == pytest.approx([0.5])
        assert records_to_metric(held, [0, 1], "regret") == pytest.approx([0.5])

    def test_an_older_record_still_loads_as_a_campaign_that_finished(self, tmp_path):
        # `exhausted` defaults to False, so every record written before the field
        # existed keeps meaning what it meant. A default of True, or a required
        # field, would silently reclassify or drop the whole store -- `load`
        # skips a line it cannot build a record from, so a required field would
        # make every stored campaign disappear rather than fail loudly.
        assert not stored(tmp_path, toy_task(reanchor=True, attainable=None)).exhausted


def test_the_stored_provenance_names_the_radius_and_the_anchor_rule(tmp_path):
    # Rounds and batch size alone cannot tell a 4-mutation fixed-anchor run from
    # a 4-per-round re-anchored one, and those are different experiments.
    moved = stored(tmp_path / "moved", toy_task(reanchor=True, attainable=None))
    fixed = stored(tmp_path / "fixed", toy_task(reanchor=False, attainable=None))
    assert "re-anchored" in moved.protocol
    assert "fixed anchor" in fixed.protocol
    assert "4/round" in moved.protocol
    assert moved.protocol != fixed.protocol


# --------------------------------------------------------------------------
# The diagnostics that vary constructibility and the anchor rule, and above
# all the cells they must not define twice.
# --------------------------------------------------------------------------


def permitted_share(task):
    """Fraction of token pairs this task's landscape allows to be adjacent."""
    return float((task.landscape().transition_matrix > 0).mean())


def instance_shape(task):
    """Everything about a task's Ehrlich instance except its constraint."""
    landscape = task.landscape()
    return (
        landscape.sequence_length,
        landscape.alphabet.size,
        landscape.n_motifs,
        landscape.motif_length,
        landscape.quantization,
    )


def unwrapped(arm):
    """An arm's methodology as a bare callable, declaring nothing about itself."""
    return arm.run


def test_the_density_sweep_reuses_the_task_it_already_defines():
    # The failure this exists for costs compute twice and then hides it: the
    # store keys on (task, arm), so a rung defined as a renamed copy of an
    # existing task runs every one of its campaigns again and files them where
    # nothing compares them to the originals. The rung at the shared
    # diagnostic's own density has to *be* that task, object for object.
    rungs = {task.name: task for task in constraint_density()}
    shared = objective_task()

    assert shared.name in rungs, (
        f"the density sweep defines {sorted(rungs)} and none of them is {shared.name}, "
        f"so the rung at density {DIAGNOSTIC_DENSITY} is a twin of a task that already exists"
    )
    reused = rungs[shared.name]
    assert repr(reused) == repr(shared)
    assert reused.attainable == shared.attainable
    assert reused.build is shared.build


def test_every_density_rung_varies_the_density_and_nothing_else():
    # A sweep whose rungs differ in a second parameter is not measuring its own
    # axis, and nothing downstream could tell: every rung would still produce a
    # number, and the curve through them would still look like a curve.
    tasks = constraint_density()

    assert len({instance_shape(task) for task in tasks}) == 1
    assert {repr(task.protocol) for task in tasks} == {repr(objective_task().protocol)}
    assert {task.max_mutations for task in tasks} == {objective_task().max_mutations}
    assert {task.reanchor for task in tasks} == {True}


def test_the_density_rungs_run_from_the_sparsest_to_no_constraint_at_all():
    # The top rung is the axis's own control: with every adjacency permitted
    # nothing bred can fail to be constructible, so a non-zero share measured
    # there is a fault in the measurement rather than a property of a landscape.
    # Without it a small share elsewhere cannot be told from a small bug.
    shares = [permitted_share(task) for task in constraint_density()]

    assert shares == sorted(shares)
    assert shares[-1] == 1.0
    assert shares[0] < shares[-1]
    assert list(CONSTRAINT_DENSITIES) == sorted(CONSTRAINT_DENSITIES)


def test_the_density_rungs_the_audit_does_not_cover_declare_nothing():
    # Copying the shared diagnostic's audited value across would be the failure
    # the attainable mechanism exists to prevent: lowering the density shrinks
    # the reachable set, so that value is a target the sparser rungs cannot
    # reach, and every regret stored on them would carry a floor no method could
    # clear and no method caused.
    shared = objective_task()
    for task in constraint_density():
        if task.name == shared.name:
            assert task.attainable is not None
        else:
            assert task.attainable is None, (
                f"{task.name} declares an attainable optimum nobody audited at its density"
            )


def test_the_fixed_anchor_control_is_the_diagnostic_task_with_one_thing_changed():
    # A control that differs in two things controls for neither. The anchor rule
    # is the axis; the instance, the protocol and the radius are held.
    fixed = fixed_anchor_task()
    moved = objective_task()

    assert fixed.build is moved.build
    assert fixed.protocol is moved.protocol
    assert fixed.max_mutations == moved.max_mutations
    assert moved.reanchor
    assert not fixed.reanchor
    assert fixed.name != moved.name
    # And no regret, because `DIAGNOSTIC_ATTAINABLE` is what four *re-anchored*
    # rounds reach. Stored against a fixed anchor it would report the difference
    # between two reachable sets as this arm's shortfall.
    assert fixed.attainable is None
    assert fixed.search_budget == fixed.max_mutations


def test_no_two_cells_of_the_anchor_study_name_the_same_key():
    # (task, arm) is the store's key. Two cells sharing one are one cell running
    # twice, and the second write silently replaces the first.
    keys = [(cell.task.name, cell.arm) for cell in anchor_study()]

    assert len(set(keys)) == len(keys)


def test_the_anchor_study_reuses_the_shipped_cell_rather_than_renaming_it():
    # The re-anchored, policy-carrying cell is the shipped configuration and has
    # already been run. Declaring it under a new task or arm name would buy a
    # second copy of it and leave the two uncheckable against each other.
    cells = {(cell.task.name, cell.arm): cell for cell in anchor_study()}
    shipped = (objective_task().name, "gfn-tb")

    assert shipped in cells
    assert cells[shipped].moves_anchor
    assert cells[shipped].carries_policy is True
    assert set(anchor_arms()) >= {cell.arm for cell in anchor_study()}
    # The arms are the suite's own objects, not re-declarations of them.
    assert anchor_arms()["gfn-tb"] is OBJECTIVES["gfn-tb"]
    assert anchor_arms()["genetic"] is BASELINES["genetic"]


def test_the_fixed_anchor_row_carries_one_gflownet_cell():
    # Carrying and rebuilding differ only in what happens when the anchor moves,
    # and it never moves here, so a second GFlowNet cell on this row would store
    # one campaign under two arm names -- the twin the study is arranged to
    # avoid, in the row where it is hardest to notice.
    fixed = [cell for cell in anchor_study() if not cell.moves_anchor]

    assert sorted(cell.arm for cell in fixed) == ["genetic", "gfn-tb"]
    assert all(cell.carries_policy is None for cell in fixed)


def test_both_axes_of_the_anchor_study_are_actually_varied():
    # Two orthogonal mechanisms, and a study missing either row answers a
    # different question: without the fixed-anchor row, "carrying helps" cannot
    # be told from "carrying helps *because* the ball moved"; without the
    # baseline, re-anchoring's effect cannot be told from a property of the
    # protocol that every method shares.
    cells = anchor_study()

    assert {cell.moves_anchor for cell in cells} == {True, False}
    assert {cell.carries_policy for cell in cells if cell.moves_anchor} == {True, False, None}
    assert {cell.arm for cell in cells} >= {"genetic"}


def test_a_real_campaign_stores_the_settings_its_arm_resolved_to(tmp_path):
    """Guards a provenance column that is in the schema and empty in the data.

    The mapping is read off the methodology by attribute, so an arm that
    declares nothing and an arm whose settings never reach the record are
    indistinguishable once stored -- and the whole point of the field is to stop
    a later reader parsing the arm's name for what it ran at.
    """
    record = stored(tmp_path, toy_task(reanchor=True, attainable=None))

    assert record.parameters["family"] == "classical"
    assert record.parameters["sampler"] == "random"
    assert record.parameters["surrogate"] is False


def test_a_methodology_that_states_no_settings_stores_an_empty_mapping(tmp_path):
    # A plain callable declares nothing, and the record says so rather than
    # inventing a configuration for a closure nobody can see into.
    store = ResultStore(tmp_path)
    task = toy_task(reanchor=True, attainable=None)
    run_task(task, {"anonymous": unwrapped(BASELINES["random"])}, store, [0], report=lambda _: None)

    assert store.load(task.name, "anonymous")[0].parameters == {}


def _motifs(task) -> bytes:
    """The planted motifs, which are what distinguishes one Ehrlich draw."""
    landscape = task.landscape()
    assert isinstance(landscape, EhrlichLandscape)
    return bytes(landscape.motifs.tobytes())


class TestReplication:
    """The replicates exist to vary one thing, so the test is that they vary one thing."""

    def _headline(self, shape):
        return next(t for t in MAIN if t.name == f"protocol-{shape}")

    @pytest.mark.parametrize("shape", ["alde", "evolvepro"])
    def test_a_replicate_differs_from_its_headline_task_only_in_the_draw(self, shape):
        # The whole value of a replicate is that a difference in its result is a
        # difference between instances. Any other axis drifting -- a radius, a
        # round count, the anchor rule -- would make the comparison measure that
        # instead, and would do so invisibly, since both tasks would still run
        # and still report a number.
        headline = self._headline(shape)
        for task in replication():
            if not task.name.startswith(f"replicate-{shape}-"):
                continue
            assert task.protocol.rounds == headline.protocol.rounds
            assert task.protocol.batch_size == headline.protocol.batch_size
            assert task.max_mutations == headline.max_mutations
            assert task.reanchor == headline.reanchor

    @pytest.mark.parametrize("shape", ["alde", "evolvepro"])
    def test_every_replicate_draws_a_different_landscape(self, shape):
        # Two replicates on one draw would look like independent evidence and be
        # a repeat, which is the failure this whole tier was added to fix.
        #
        # Compared on the motifs rather than on the optimum: every Ehrlich
        # instance is built so its optimum is 1.0, so an optimum-based check
        # would pass on three copies of the same landscape.
        drawn = {
            _motifs(task) for task in replication() if task.name.startswith(f"replicate-{shape}-")
        }
        assert len(drawn) == len(REPLICATION_SEEDS)

    @pytest.mark.parametrize("shape", ["alde", "evolvepro"])
    def test_no_replicate_redraws_the_headline_instance(self, shape):
        # Distinct seeds are necessary and not sufficient: two seeds could in
        # principle land on the same motifs, and that replicate would be
        # answering the question by restating it.
        headline = _motifs(self._headline(shape))
        for task in replication():
            if task.name.startswith(f"replicate-{shape}-"):
                assert _motifs(task) != headline

    def test_no_replicate_reuses_the_shared_instance(self):
        # Seed 2 is the draw both headline tasks share; a replicate on it would
        # answer the question by restating it.
        assert 2 not in REPLICATION_SEEDS

    def test_the_replicates_are_named_apart_from_the_headline_tasks(self):
        # Store keys are file names, so a collision would append a replicate's
        # campaigns to the headline task's own record file.
        assert {t.name for t in replication()}.isdisjoint({t.name for t in MAIN})
