"""Tests for the multi-objective suite: what it declares, and what a run stores.

These are the failures under test, and each is one this suite could plausibly
ship with because none of them raises.

**An indicator measured from nowhere.** Hypervolumes taken from different
reference points are not comparable and neither number records its point, so a
task that forgot to state one, or a campaign that silently picked up whatever the
landscape happened to expose, would produce a column that mixes scales. The task
refuses to exist without a reference point, and the point is in the `repr` every
record stores as provenance.

**A trade-off applied in one place and not another.** A campaign that ranked its
pool under one preference while its sampler bred under a different one -- or
under none, because a classical baseline refuses an objective matrix -- would
search somewhere its own ledger never claimed was good.

**A reference front that cannot be reached, or is reached by accident.** The
CH65 front must exclude variants tied at the detection floor, and the Ehrlich
fronts must consist of points some sequence actually attains.

**A preference ensemble that is not at fixed budget.** Running eight preferences
at a full budget each and comparing that against one preference at a full budget
is not a comparison, and it is what "eight preferences" would mean if the split
were forgotten.

**A headline table paired against a hybrid nobody published.** This one had
already shipped: the reference arm was ``genetic+proxy``, a genetic algorithm
handed this campaign's surrogate and an inner loop over it, and no plain
``genetic`` arm existed at all -- so the suite carried an ablation without the
pipeline it ablates and paired every result against it.
`TestTheLadderIsWhatItSaysItIs` pins the shape that replaces it, mirroring
``tests/benchmark/test_methods.py`` arm for arm: an arm silently regaining a
surrogate, or a rung quietly carrying two changes, is a table of hybrids again
and nothing in the numbers would say so.

**An arm that exists but can never be reported.** This one had also shipped:
`mogfn-pc` was built and unit-tested, and every tier that ran it was a
diagnostic, so the only genuinely multi-objective arm in the suite was reachable
from nothing that carries results. An unregistered arm looks exactly like an arm
that lost -- it is simply absent from the table -- so the same class pins which
tier reaches it, and pins the two registrations of it against each other.

The CH65 tests need the downloaded dataset and say so. The end-to-end arm tests
are kept to a two-round toy with a four-design plate, which is enough to exercise
every seam the suite has and cheap enough to run on every commit.
"""

import numpy as np
import pytest

from evogfn.acquisition.rules import Greedy
from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
from evogfn.algorithms.baselines.nsga2 import NSGA2
from evogfn.algorithms.gflownet.flow_objectives import SubTrajectoryBalance
from evogfn.algorithms.gflownet.preference_sampler import PreferenceConditionedSampler
from evogfn.algorithms.inner_loop import ProxyOptimising
from evogfn.benchmark.multi_objective import (
    ABLATIONS,
    ARMS,
    CONFLICT_SWEEP,
    DEFAULT_POOL,
    EXACT_FRONT_LIMIT,
    MO_LENGTH,
    MO_VOCAB_SIZE,
    MULTI_OBJECTIVE_MAIN,
    SCOPE_NOTES,
    UNSELECTED_POOL_PLATES,
    MultiObjectiveTask,
    PreferenceEnsemble,
    ScalarizedObserving,
    arms_for_tier,
    ch65_reference_front,
    conflict_sweep,
    enumerated_front,
    multi_objective_tiers,
    objective_count_sweep,
    preference_arms,
    preference_conditioned_arm,
    preference_task,
    preference_vectors,
    recombination_front,
    run_multi_objective_task,
    scalarized_gflownet_arm,
    set_indicators,
)
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.store import ResultStore
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.base import MAX_ENUMERABLE_SIZE
from evogfn.landscapes.ch65 import CH65_DETECTION_FLOOR, CH65Landscape
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.multi_ehrlich import MultiEhrlichLandscape
from evogfn.loop.campaign import Campaign
from evogfn.loop.ledger import CampaignResult
from evogfn.metrics.pareto import non_dominated, pymoo_available
from evogfn.rewards.scalarization import WeightedSum

EVERY_TASK = [
    *MULTI_OBJECTIVE_MAIN,
    *conflict_sweep(),
    *objective_count_sweep(),
    preference_task(),
]

#: Every task whose landscape is a multi-Ehrlich instance, which is all of them
#: but `ch65-real`. These are the ones the enumerable alphabet was bought for.
EVERY_EHRLICH_TASK = [task for task in EVERY_TASK if task.name != "ch65-real"]


def toy_landscape(conflict=1.0):
    """A multi-Ehrlich instance small enough to run several campaigns against.

    Full conflict by default, so its front is more than a single point -- a front
    of one would make every coverage assertion below vacuous.
    """
    return MultiEhrlichLandscape.with_conflict(
        sequence_length=16,
        vocab_size=4,
        n_objectives=2,
        n_motifs=1,
        motif_length=4,
        quantization=4,
        max_spacing=2,
        transition_density=0.5,
        conflict=conflict,
        seed=3,
    )


#: Rounds and plate for the end-to-end runs. Small, because the property under
#: test is accounting rather than optimisation, and two rounds is the fewest that
#: can exercise the campaign's memory of an earlier one.
ROUNDS = 2
BATCH = 4


def toy_task(*, rounds=ROUNDS, batch=BATCH, front=recombination_front):
    """A multi-objective task cheap enough to run end to end."""
    return MultiObjectiveTask(
        name="toy",
        purpose="a toy, for testing that the wiring does what the table says",
        build=toy_landscape,
        protocol=Protocol(rounds=rounds, batch_size=batch, max_mutations=4),
        max_mutations=4,
        reanchor=True,
        reference_point=(0.0, 0.0),
        front=front,
        front_is_exact=False,
    )


# --------------------------------------------------------------------------
# What a task declares.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", EVERY_TASK, ids=lambda t: t.name)
def test_every_task_states_where_its_hypervolume_is_measured_from(task):
    # Without this the campaign falls back to whatever the landscape exposes,
    # and a landscape gaining or moving a `reference_point` property would
    # silently rescale every result already in the store.
    assert task.reference_point
    assert task.n_objectives == len(task.reference_point)
    assert all(np.isfinite(task.reference_point))


def test_a_task_without_a_reference_point_is_refused():
    with pytest.raises(ValueError, match="must state the reference point"):
        MultiObjectiveTask(
            name="nowhere",
            purpose="x",
            build=toy_landscape,
            protocol=Protocol(rounds=1, batch_size=2, max_mutations=4),
            max_mutations=4,
        )


@pytest.mark.parametrize("task", EVERY_TASK, ids=lambda t: t.name)
def test_the_provenance_string_carries_the_reference_point(task):
    # `run_multi_objective_task` stores this as a record's protocol field. Two
    # records at 4x96=384 whose hypervolumes came from different points are not
    # comparable, and a string naming only the protocol could not tell them
    # apart.
    text = repr(task)
    assert task.name in text
    assert f"{task.n_objectives} objectives" in text
    assert f"{task.reference_point[0]:g}" in text


@pytest.mark.parametrize("task", EVERY_TASK, ids=lambda t: t.name)
def test_the_search_budget_is_cumulative_only_where_the_anchor_moves(task):
    expected = task.max_mutations * task.protocol.rounds if task.reanchor else task.max_mutations
    assert task.search_budget == expected


def test_the_tiers_say_which_of_them_carries_results():
    tiers = multi_objective_tiers(4, 2)
    headline = [tier.name for tier in tiers if tier.headline]
    # Exactly one tier carries claims. The sweeps say when a ranking would
    # change and the preference diagnostic decides one setting; promoting either
    # to a result is the misreading this suite is laid out to prevent.
    assert headline == ["main"]
    assert {t.name for t in tiers} == {"main", "conflict", "objectives", "preferences"}


def test_the_multi_ehrlich_parent_is_drawn_rather_than_planted():
    # `Task.parent` raises for anything but GB1 and Ehrlich, so a multi-objective
    # task that did not override it could not be run at all. The starting point
    # must also be independent of the answer: a parent that *was* a planted
    # optimum would hand every arm the front for free.
    task = toy_task()
    landscape = task.landscape()
    parent = task.parent(landscape)
    assert landscape.is_feasible(parent[None, :])[0]
    assert not any(np.array_equal(parent, optimum) for optimum in landscape.optimal_sequences)


# --------------------------------------------------------------------------
# Preferences.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_objectives", [2, 3, 4])
@pytest.mark.parametrize("count", [1, 4, 8])
def test_preferences_lie_on_the_simplex(n_objectives, count):
    weights = preference_vectors(n_objectives, count, seed=1)
    assert weights.shape == (count, n_objectives)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_one_preference_is_the_neutral_one():
    # Anything else would be the benchmark, rather than the biology, claiming
    # which objective matters.
    np.testing.assert_allclose(preference_vectors(3, 1), [[1 / 3, 1 / 3, 1 / 3]])


def test_two_objectives_get_an_even_grid_including_both_ends():
    weights = preference_vectors(2, 5)
    np.testing.assert_allclose(weights[0], [0.0, 1.0])
    np.testing.assert_allclose(weights[-1], [1.0, 0.0])
    assert len(np.unique(weights[:, 0])) == 5


def test_a_two_objective_grid_does_not_depend_on_the_seed():
    # The diagnostic varies the *count*. A grid that moved per seed would be
    # varying the draw as well, and the two effects would not be separable.
    np.testing.assert_array_equal(
        preference_vectors(2, 4, seed=0), preference_vectors(2, 4, seed=9)
    )


def test_three_objectives_do_depend_on_the_seed():
    # No even lattice has four points on a triangle, so these are drawn -- and
    # drawing them per seed is what averages the comparison over draws.
    left = preference_vectors(3, 4, seed=0)
    right = preference_vectors(3, 4, seed=1)
    assert not np.allclose(left, right)


def test_asking_for_no_preferences_is_refused():
    with pytest.raises(ValueError, match="count must be at least 1"):
        preference_vectors(2, 0)


# --------------------------------------------------------------------------
# Reference fronts.
# --------------------------------------------------------------------------


def test_aligned_objectives_have_a_one_point_front():
    # The landscape's own documentation says the exact front at conflict 0 is a
    # single point, and this is the one setting where the construction and the
    # truth provably coincide -- so it is where the construction is checkable.
    np.testing.assert_allclose(recombination_front(toy_landscape(conflict=0.0)), [[1.0, 1.0]])


def test_every_point_of_a_constructed_front_is_attained_by_a_real_sequence():
    # An unattainable "front" would put a permanent floor under IGD+, so no arm
    # could ever reach zero and the indicator would stop discriminating at the
    # top. Every candidate is a real sequence and every infeasible one scores
    # -inf and is dropped, so finiteness is the property that says so.
    landscape = toy_landscape()
    front = recombination_front(landscape)
    assert front.shape == (front.shape[0], 2)
    assert np.isfinite(front).all()
    assert non_dominated(front).all()
    # Each objective's planted optimum is in the candidate set and scores 1.0 on
    # its own objective, so nothing can dominate it away from the column maximum.
    np.testing.assert_allclose(front.max(axis=0), 1.0)


@pytest.mark.parametrize("task", EVERY_TASK, ids=lambda t: t.name)
def test_every_task_is_scored_against_the_true_front(task):
    # `front_is_exact` is what a table has to print next to an IGD+, because an
    # IGD+ of zero against a constructed subset means "covered what one
    # construction found" and against the real front means "covered the front".
    # Every task here can now say the second, and the flag must not drift back
    # to claiming it without the enumeration behind it.
    assert task.front is not None
    assert task.front_is_exact
    assert "exact front" in repr(task)


def test_the_ehrlich_instances_are_enumerable_at_all():
    # The whole reason the alphabet is 4. This is the arithmetic that decides
    # whether `exact_pareto_front` runs or raises, so it is asserted rather than
    # left to the first campaign to discover.
    assert MO_VOCAB_SIZE**MO_LENGTH <= MAX_ENUMERABLE_SIZE
    # And not vacuously: one letter more would not fit, so the length is at the
    # useful end of the guard rather than an arbitrary small number.
    assert MO_VOCAB_SIZE ** (MO_LENGTH + 2) > MAX_ENUMERABLE_SIZE


@pytest.mark.slow
@pytest.mark.parametrize("task", EVERY_EHRLICH_TASK, ids=lambda t: t.name)
def test_the_enumerated_front_is_a_real_front_of_attained_values(task):
    # Enumerating 4^10 sequences takes a second or two per task, which is why
    # this is slow and why `reference_front` memoises it -- a fifty-seed arm must
    # not pay for it fifty times.
    front = task.reference_front()
    assert front is not None
    assert front.shape[1] == task.n_objectives
    assert np.isfinite(front).all()
    assert non_dominated(front).all()
    # Each objective attains 1.0 at its own planted optimum, and that sequence is
    # in the space being enumerated, so every column maximum is exactly 1.
    np.testing.assert_allclose(front.max(axis=0), 1.0)


def front_sizes(tasks):
    """How many points each task's exact reference front holds."""
    fronts = [task.reference_front() for task in tasks]
    assert all(front is not None for front in fronts)
    return [front.shape[0] for front in fronts]


@pytest.mark.slow
def test_the_conflict_sweep_brackets_where_the_front_starts_to_spread():
    # The failure this replaces: conflict 0.25 gave a one-point front identical
    # to conflict 0.0, so two of five rungs measured no trade-off at all. The
    # rungs are now placed on a scan of the dial, and what makes them worth their
    # runtime is that the transition happens *inside* the sweep.
    sizes = front_sizes(conflict_sweep())
    assert CONFLICT_SWEEP[0] == 0.0
    assert CONFLICT_SWEEP[-1] == 1.0
    assert sizes[0] == 1, "conflict 0 is the aligned control: one point, provably"
    assert sizes[-1] > 1, "conflict 1 must have a trade-off to spread over"
    # The transition happens once and inside the sweep: every collapsed rung
    # comes before every spread one, so two adjacent rungs bracket it and none
    # of the sweep steps over it.
    collapsed = [size == 1 for size in sizes]
    assert collapsed == sorted(collapsed, reverse=True)
    # Most of the sweep is above the transition. The old rungs put three of five
    # below it on this instance and spanned it with a gap of 0.25; these put two
    # below and span it with 0.2.
    assert sum(collapsed) == 2


@pytest.mark.slow
def test_more_objectives_make_a_thicker_front():
    # The confound `objective_count_sweep` exists to separate from anything a
    # method does: the front itself grows with the objective count, so a
    # coverage indicator gets harder for reasons that are not the method's.
    sizes = front_sizes(objective_count_sweep())
    assert sizes == sorted(sizes)
    assert sizes[0] > 1
    # And the four-objective front alone outgrows what inclusion-exclusion can
    # take, which is what makes the optional exact backend load-bearing here
    # rather than a nicety.
    assert sizes[-1] > EXACT_FRONT_LIMIT


def test_the_enumerated_front_refuses_a_landscape_that_is_not_multi_ehrlich():
    with pytest.raises(TypeError, match="not enumerated for its front"):
        enumerated_front(
            EhrlichLandscape(sequence_length=8, vocab_size=4, n_motifs=1, motif_length=2)
        )


def test_a_constructed_front_refuses_a_landscape_with_no_planted_optima():
    with pytest.raises(TypeError, match="has none"):
        recombination_front(
            EhrlichLandscape(sequence_length=8, vocab_size=4, n_motifs=1, motif_length=2)
        )


@pytest.mark.requires_data
def test_the_ch65_front_drops_the_variants_tied_at_the_detection_floor():
    landscape = CH65Landscape()
    front = ch65_reference_front(landscape)
    space = landscape.enumerate()
    values = np.asarray(landscape.evaluate(space), dtype=np.float64)
    measured = landscape.is_measured(space)

    with_censored = int(non_dominated(values[measured]).sum())
    # 20 with the censored variants, 19 without: exactly one front point is
    # non-dominated only because an objective could not resolve it, and it is
    # unreachable anyway -- a censored value sits *on* the reference point, and
    # hypervolume counts only designs strictly above it.
    assert with_censored == 20
    assert front.shape == (19, 3)
    assert (front > CH65_DETECTION_FLOOR).all()


def test_the_ch65_front_is_refused_for_any_other_landscape():
    with pytest.raises(TypeError, match="cannot be computed"):
        ch65_reference_front(toy_landscape())


# --------------------------------------------------------------------------
# The scalarising adapter.
# --------------------------------------------------------------------------


def _toy_environment():
    """An environment over the toy landscape, anchored at a feasible parent."""
    landscape = toy_landscape()
    return landscape, MutationEnvironment(
        landscape.feasible_sequence(0),
        landscape.alphabet,
        max_mutations=4,
        transitions=landscape.transition_matrix,
    )


def test_a_classical_baseline_refuses_an_objective_matrix():
    # The premise of the adapter. If this ever stops raising, the adapter is
    # papering over something that has been fixed properly.
    landscape, env = _toy_environment()
    sampler = GeneticAlgorithm(env, seed=0)
    batch = sampler.propose(4)
    with pytest.raises(ValueError, match="must be scalarised"):
        sampler.observe(batch, landscape.evaluate(batch))


def test_the_adapter_lets_it_rank_under_a_stated_trade_off():
    landscape, env = _toy_environment()
    wrapped = ScalarizedObserving(
        GeneticAlgorithm(env, seed=0), scalarization=WeightedSum(), preference=np.array([0.5, 0.5])
    )
    batch = wrapped.propose(4)
    wrapped.observe(batch, landscape.evaluate(batch))
    assert wrapped.proposals_made == 4


def test_the_adapter_passes_a_single_objective_batch_straight_through():
    # The proxy the inner loop searches against returns (n, 1), so both widths
    # arrive within one round. Scalarising the scalar would apply a two-entry
    # preference to something that is not an objective vector, and would raise.
    _, env = _toy_environment()
    wrapped = ScalarizedObserving(
        GeneticAlgorithm(env, seed=0), scalarization=WeightedSum(), preference=np.array([0.5, 0.5])
    )
    batch = wrapped.propose(4)
    wrapped.observe(batch, np.arange(4, dtype=np.float64)[:, None])


def test_the_adapter_forwards_a_moved_anchor():
    # The campaign checks the *outermost* object for `reanchored`. A wrapper
    # without the hook sends every arm it wraps down the rebuild path, and the
    # populations each baseline carefully carries are discarded silently.
    landscape, env = _toy_environment()
    inner = GeneticAlgorithm(env, seed=0)
    wrapped = ScalarizedObserving(
        inner, scalarization=WeightedSum(), preference=np.array([0.5, 0.5])
    )
    moved = wrapped.reanchored(env.reanchored(landscape.feasible_sequence(1)))
    assert isinstance(moved, ScalarizedObserving)
    assert isinstance(moved.inner, GeneticAlgorithm)


# --------------------------------------------------------------------------
# The arms, end to end.
# --------------------------------------------------------------------------


def test_a_scalar_acquisition_is_refused_against_several_objectives():
    # The reason every arm here, NSGA-II included, is built with a
    # ScalarizedAcquisition. Refused at construction, before any oracle call.
    landscape, env = _toy_environment()
    with pytest.raises(ValueError, match="state the trade-off explicitly"):
        Campaign(
            landscape=landscape,
            sampler=NSGA2(env, seed=0),
            acquisition=Greedy(),
            rounds=1,
            batch_size=2,
            pool_size=4,
        )


#: The arms that are pipelines as published -- bare, and the comparison the paper
#: is actually about. Everything else in `ARMS` is a decomposition row and says so
#: in its own name.
PUBLISHED = ("random", "nsga2", "genetic", "gfn-tb-scalar", "mogfn-pc")

#: The two arms `ARMS` holds at their shipped training budget, and the toy
#: rebuilds an end-to-end test runs instead. 300 gradient steps a round is
#: minutes per case on tests that run on every commit, and the conditioned arm's
#: eight-trade-off grid costs another factor of four on top -- 22 s against 5 s,
#: measured. Everything else is the registry's own construction, so what these
#: exercise is still the shipped arm's wiring rather than a lookalike.
TOY_TRAINING = {
    "gfn-tb-scalar": scalarized_gflownet_arm(1, steps=4),
    "mogfn-pc": preference_conditioned_arm(2, steps=4),
}


def toy_arm(name):
    """`ARMS[name]`, at a training budget a per-commit test can afford."""
    return TOY_TRAINING.get(name, ARMS[name])


#: The arms whose surrogate is constitutive of the pipeline rather than an
#: addition to it: both GFlowNets, whose policies are not trainable at 384 assays
#: without one. Spelled out rather than taken from `TOY_TRAINING`'s keys, which
#: are the same two arms for an unrelated reason -- one list is about what a
#: pipeline *is*, the other about what a test can afford, and tying them would
#: make a future cheap GFlowNet silently count as bare.
CONSTITUTIVE_SURROGATE = ("gfn-tb-scalar", "mogfn-pc")

#: What each arm's own paper says its candidate pool is, written the way
#: `PLATE_POOL` is resolved -- against the task rather than as a literal. A
#: genetic algorithm's population is its evaluation batch (Stanton et al.) and
#: mutagenesis proposes a neighbourhood of the current point, so both are one
#: plate; NSGA-II's generation is neither a plate nor a library; and every
#: screened rung keeps the library, because at a plate the model would rank
#: `BATCH` candidates into `BATCH` wells and change nothing at all. `mogfn-pc`
#: keeps it for a different reason again: it divides the pool between its
#: trade-offs, so a plate would leave one design each and the per-preference
#: ranking would rank nothing.
EXPECTED_POOL = {
    "random": BATCH,
    "nsga2": BATCH * UNSELECTED_POOL_PLATES,
    "genetic": BATCH,
    "gfn-tb-scalar": DEFAULT_POOL,
    "mogfn-pc": DEFAULT_POOL,
    "random+screen": DEFAULT_POOL,
    "genetic+screen": DEFAULT_POOL,
    "genetic+search": DEFAULT_POOL,
}


class TestTheLadderIsWhatItSaysItIs:
    """Each arm's name is a claim about its pipeline, checked against the object."""

    def test_the_arms_are_exactly_the_published_pipelines_plus_the_ladder(self):
        # Pinned as a set so that adding an arm is a deliberate edit here. An arm
        # that appears without a rung to stand on is a hybrid back in the table,
        # which is exactly how `genetic+proxy` came to be the reference.
        assert set(ARMS) == set(PUBLISHED) | set(ABLATIONS)

    def test_every_ablation_names_a_pipeline_that_is_actually_here(self):
        # The fault this suite shipped with: `genetic+proxy` decomposed a
        # `genetic` arm that did not exist, so the decomposition row was the
        # yardstick by default and there was nothing to decompose it against.
        assert set(ABLATIONS).isdisjoint(PUBLISHED)
        assert set(ABLATIONS.values()) <= set(PUBLISHED)

    @pytest.mark.parametrize("name", PUBLISHED)
    def test_every_published_arm_is_bare(self, name):
        # A deep ensemble in front of a baseline whose paper has none is what
        # made the headline comparison a comparison between hybrids. The two
        # GFlowNet arms are excluded: their surrogate is constitutive of the
        # pipeline rather than an extra, being what makes a policy trainable at
        # this budget at all. `mogfn-pc` has to be excluded by name rather than
        # left to the assertions, which it would *pass* -- its campaign-level
        # surrogate is `None`, but because one there would re-rank its
        # preference-diverse pool under a single trade-off, not because the
        # pipeline has no model. Its model is on the sampler, and a check that
        # read `_surrogate is None` as "bare" would call this arm bare forever.
        if name in CONSTITUTIVE_SURROGATE:
            pytest.skip("the surrogate is constitutive of the GFlowNet pipelines")
        campaign = ARMS[name](toy_task(), 0)
        assert isinstance(campaign, Campaign)

        assert campaign._surrogate is None, f"{name} was handed a surrogate its paper has none of"
        inner = getattr(campaign.sampler, "inner", campaign.sampler)
        assert not isinstance(inner, ProxyOptimising), (
            f"{name} was allowed to optimise against a model its paper has none of"
        )

    def test_the_ladder_adds_exactly_one_thing_per_rung(self):
        # Read down the column: nothing, then a screen, then search against the
        # same model. A rung that quietly carried two changes would attribute
        # both to whichever one it was named for.
        task = toy_task()
        bare = ARMS["genetic"](task, 0)
        screen = ARMS["genetic+screen"](task, 0)
        search = ARMS["genetic+search"](task, 0)
        assert isinstance(bare, Campaign)
        assert isinstance(screen, Campaign)
        assert isinstance(search, Campaign)

        assert (bare._surrogate, screen._surrogate is None, search._surrogate is None) == (
            None,
            False,
            False,
        )
        assert not isinstance(getattr(screen.sampler, "inner", None), ProxyOptimising), (
            "the +screen rung filters the pool; its search is still blind"
        )
        assert isinstance(getattr(search.sampler, "inner", None), ProxyOptimising)
        # And the scalarisation is not one of the things a rung adds: every
        # classical arm breeds under the trade-off its campaign ranks with, so a
        # rung cannot be measuring "plus a weighting" as well as its own name.
        assert all(
            isinstance(campaign.sampler, ScalarizedObserving) for campaign in (bare, screen, search)
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_POOL))
    def test_each_arm_gets_the_pool_its_paper_specifies(self, name):
        # The bare arms used to get 2048 like everything else, which gave a
        # genetic algorithm a pool twenty times its own population -- and, with
        # no surrogate to screen it, threw nineteen twentieths of that away
        # unmeasured. `PLATE_POOL` resolves against the task, so a protocol
        # sweep moves the population with the plate.
        campaign = ARMS[name](toy_task(), 0)
        assert isinstance(campaign, Campaign)
        assert campaign._pool_size == EXPECTED_POOL[name]

    def test_an_arm_whose_name_overclaims_carries_a_scope_note(self):
        # `gfn-tb-scalar` is GFlowNet-AL over a fixed weighted sum, and a reader
        # who has met MOGFN-PC will assume a preference-conditioned policy --
        # more so now that the real one is a row of the same table. The report
        # prints this beside the row; without it the arm's name is the only thing
        # saying what was run, and it says something stronger than the truth.
        assert "MOGFN-PC" in SCOPE_NOTES["gfn-tb-scalar"]
        # A note keyed to a name no arm has is a note that never prints, which is
        # exactly what a rename leaves behind if the key is not moved with it.
        assert set(SCOPE_NOTES) <= set(ARMS)
        # And the arm the note redirects to has to be somewhere the reader can
        # find it: the note names `mogfn-pc`, so a suite that dropped that arm
        # would leave the redirect pointing at nothing.
        assert "mogfn-pc" in ARMS

    def test_the_conditioned_arm_reaches_the_tier_that_carries_results(self):
        # The failure this closes, which shipped once already: `mogfn-pc` was
        # built, tested and registered in `preference_arms`, and every tier that
        # ran it was a diagnostic -- so the one arm in this suite that is
        # genuinely multi-objective could not appear in a result at all. Nothing
        # raised and no test failed; the row was simply never produced.
        headline = [tier for tier in multi_objective_tiers(1, 1) if tier.headline]
        assert all("mogfn-pc" in arms_for_tier(tier) for tier in headline)

    def test_the_two_registrations_of_the_conditioned_arm_agree(self):
        # `mogfn-pc` is registered twice under one name -- in `ARMS` for the
        # headline row, in `preference_arms` for the decomposition that explains
        # it. Configured differently they would be two arms sharing a key, the
        # diagnostic would be evidence about a row the table never printed, and
        # only the name would claim otherwise. Compared by construction rather
        # than by identity: the two are built by separate calls, so `is` would
        # fail on arms that are in fact the same.
        task = toy_task()
        headline = ARMS["mogfn-pc"](task, 0)
        diagnostic = preference_arms()["mogfn-pc"](task, 0)
        assert isinstance(headline, Campaign)
        assert isinstance(diagnostic, Campaign)
        assert isinstance(headline.sampler, PreferenceConditionedSampler)
        assert isinstance(diagnostic.sampler, PreferenceConditionedSampler)
        assert headline._pool_size == diagnostic._pool_size
        np.testing.assert_array_equal(headline.sampler.preferences, diagnostic.sampler.preferences)


class TestEveryArmSpendsItsWholeBudget:
    """The invariant every budget-indexed claim in this suite rests on."""

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_every_arm_spends_exactly_the_budget_it_was_given(self, name):
        # Run at the pool each arm's paper specifies, which for the bare arms is
        # one plate -- the configuration under which a shortfall would show, and
        # which the old global 2048 masked completely. These tasks are also far
        # more saturated than their single-objective counterparts, so a plate
        # pool is exactly where an arm would run out of unmeasured designs.
        task = toy_task()
        result = toy_arm(name)(task, 0).run()

        assert result.oracle_calls == task.protocol.rounds * task.protocol.batch_size
        assert [record.evaluated for record in result.rounds] == [BATCH] * ROUNDS
        assert len(result.sequences) == task.protocol.budget

    @pytest.mark.parametrize("name", sorted(ARMS))
    def test_every_arm_reports_a_duplicate_share_in_range(self, name):
        # The name is a contract: the store reads this off the result by
        # attribute, so a share that goes missing or leaves [0, 1] lands in the
        # table as a silently empty -- or silently wrong -- column. It is also
        # the column that says what a plate-sized population costs.
        result = toy_arm(name)(toy_task(), 0).run()

        assert 0.0 <= result.duplicate_fraction <= 1.0


@pytest.mark.parametrize("name", sorted(ARMS))
def test_every_arm_runs_and_reports_both_indicators(name):
    task = toy_task()
    result = toy_arm(name)(task, 0).run()

    assert result.oracle_calls == task.protocol.budget
    assert result.values.shape[1] == 2
    got = set_indicators(task, result)
    volume, coverage = got["best"], got["regret"]
    assert volume is not None
    # Hypervolume can legitimately be zero here -- an Ehrlich design scoring 0 on
    # an objective is at the reference point and encloses nothing -- so what is
    # asserted is that it is a number rather than that it is large.
    assert volume >= 0.0
    assert coverage is not None
    assert coverage >= 0.0


def test_nsga2_ranks_the_objective_matrix_rather_than_a_scalarisation():
    # The whole point of having it: it must see the vectors. If the campaign ever
    # started handing samplers a reduced value, this arm would degenerate into a
    # dominance-ranked GA over one weighting and stop being a control.
    campaign = ARMS["nsga2"](toy_task(), 0)
    campaign.run()
    sampler = campaign.sampler
    assert isinstance(sampler, NSGA2)
    assert sampler.values is not None
    assert sampler.values.shape[1] == 2


def test_the_gflownet_arm_defaults_to_the_objective_its_records_were_made_under():
    # Parameterising the objective must not have moved the default: every stored
    # multi-objective record was produced under trajectory balance, and an arm
    # that silently became something else would make old and new records
    # incomparable without either of them saying so.
    campaign = scalarized_gflownet_arm(1, steps=2)(toy_task(), 0)
    assert "TrajectoryBalance" in campaign.sampler.name


def test_the_gflownet_arm_trains_under_whichever_objective_it_was_given():
    # The seam the single-objective selection reaches this suite through. Its
    # answer was `gfn-subtb`, and without this it could not be run here at all.
    task = toy_task()
    arm = scalarized_gflownet_arm(1, SubTrajectoryBalance(lam=0.9), steps=2, learn_flow=True)
    campaign = arm(task, 0)
    assert "SubTrajectoryBalance" in campaign.sampler.name
    result = campaign.run()
    # Trained, not merely constructed: a flow objective that could not run would
    # raise inside the first round that has a fitted proxy.
    assert result.oracle_calls == task.protocol.budget
    assert int(getattr(campaign.sampler, "proxy_calls", 0)) > 0


def test_a_flow_objective_without_a_flow_head_raises_rather_than_degrading():
    # Why `learn_flow` is exposed alongside the objective instead of being left
    # at its default. Falling back to a constant flow would turn subtrajectory
    # balance into a worse trajectory balance and report it under the other
    # name, which is a wrong number rather than a missing one.
    arm = scalarized_gflownet_arm(1, SubTrajectoryBalance(lam=0.9), steps=2)
    with pytest.raises(RuntimeError, match="no flow head"):
        arm(toy_task(), 0).run()


def test_the_preference_diagnostic_runs_the_objective_it_is_handed():
    # The diagnostic decides how many preferences the main-table arm gets, so it
    # has to be able to run the same configuration that table will.
    arms = preference_arms(SubTrajectoryBalance(lam=0.9), learn_flow=True)
    campaign = arms["gfn-tb-pref1"](toy_task(), 0)
    assert "SubTrajectoryBalance" in campaign.sampler.name


def test_the_search_rung_optimises_the_proxy_rather_than_only_meeting_it():
    # What the `+search` rung is for. A GFlowNet that only beats a baseline which
    # never looks at the model has beaten the access, not the method -- so the
    # rung has to really give the GA the same free evaluations, and a proxy that
    # was wired up but never called would leave the attribution unmeasured while
    # looking measured.
    campaign = ARMS["genetic+search"](toy_task(), 0)
    campaign.run()
    assert int(getattr(campaign.sampler, "proxy_calls", 0)) > 0


def test_the_screen_rung_fits_a_model_without_searching_against_it():
    # The rung below it, and the one that says whether a screen alone explains a
    # gap. It must fit the surrogate -- otherwise it is `genetic` under another
    # name -- and must not spend a single proxy call, or it is `genetic+search`
    # under another name and the ladder has two rungs measuring one thing.
    campaign = ARMS["genetic+screen"](toy_task(), 0)
    result = campaign.run()

    assert isinstance(campaign, Campaign)
    assert campaign._surrogate is not None
    assert int(getattr(campaign.sampler, "proxy_calls", 0)) == 0
    assert result.oracle_calls == toy_task().protocol.budget


# --------------------------------------------------------------------------
# The preference ensemble.
# --------------------------------------------------------------------------


def test_several_preferences_share_one_budget_rather_than_multiplying_it():
    task = toy_task(rounds=2, batch=4)
    single = scalarized_gflownet_arm(1, steps=2)(task, 0)
    split = scalarized_gflownet_arm(4, steps=2)(task, 0)
    assert isinstance(split, PreferenceEnsemble)
    # Four preferences of one design a round against one preference of four:
    # the same total, which is what makes the two comparable at all.
    assert split.budget == single.budget
    assert len(split.campaigns) == 4


def test_the_merged_result_is_the_union_of_what_every_preference_measured():
    task = toy_task(rounds=2, batch=4)
    ensemble = scalarized_gflownet_arm(2, steps=2)(task, 0)
    result = ensemble.run()
    assert result.oracle_calls == task.protocol.budget
    assert result.values.shape == (len(result.sequences), 2)
    # Renumbered, so the merged ledger reads in order rather than restarting at
    # zero once per preference.
    assert [record.index for record in result.rounds] == list(range(len(result.rounds)))


def test_a_split_that_leaves_less_than_one_design_each_is_refused():
    arm = scalarized_gflownet_arm(8, steps=2)
    with pytest.raises(ValueError, match="less than one design each"):
        arm(toy_task(rounds=2, batch=4), 0)


# --------------------------------------------------------------------------
# What a stored record holds.
# --------------------------------------------------------------------------


def _large_front_result():
    """A result whose front outgrows what inclusion-exclusion can take.

    Three objectives and 20 mutually non-dominated points, which is what a
    converged arm produces on `ch65-real` (19 measured) and less than what
    `mo-objectives-4`'s reference front alone runs to (23).
    """
    angles = np.linspace(0.1, 1.4, EXACT_FRONT_LIMIT + 4)
    # One objective rises as another falls, so no point dominates any other.
    front = np.column_stack([np.cos(angles) + 1.0, np.sin(angles) + 1.0, np.ones(angles.size)])
    assert non_dominated(front).sum() > EXACT_FRONT_LIMIT
    return CampaignResult(
        sampler="toy",
        rounds=(),
        sequences=np.zeros((front.shape[0], 4), dtype=np.int32),
        values=front,
        reference_point=np.zeros(3),
    )


@pytest.mark.skipif(pymoo_available(), reason="with the `moo` extra there is nothing to refuse")
def test_an_uncomputable_hypervolume_is_stored_as_nan_rather_than_raising():
    # What a core install does past the built-in limit. The measurements are the
    # product and they survive; the indicator says "not computed" and propagates
    # rather than raising out of the middle of a campaign.
    volume = set_indicators(toy_task(), _large_front_result())["best"]
    assert volume is not None
    assert np.isnan(volume)


@pytest.mark.skipif(not pymoo_available(), reason="needs the optional `moo` extra")
def test_a_large_front_is_a_number_rather_than_nan_with_the_extra_installed():
    # The reason the extra exists. Before it, the arms whose hypervolume went
    # missing were the ones with the best fronts -- the column was hardest to
    # populate exactly where it mattered most.
    volume = set_indicators(toy_task(), _large_front_result())["best"]
    assert volume is not None
    assert not np.isnan(volume)
    assert volume > 0.0


def test_a_result_with_no_reference_point_is_refused_rather_than_stored():
    result = CampaignResult(
        sampler="toy",
        rounds=(),
        sequences=np.zeros((2, 4), dtype=np.int32),
        values=np.zeros((2, 2)),
    )
    with pytest.raises(ValueError, match="no reference point"):
        set_indicators(toy_task(), result)


def test_a_run_is_stored_once_and_resumed_rather_than_repeated(tmp_path):
    store = ResultStore(tmp_path)
    task = toy_task()
    arms = {"random": ARMS["random"]}

    assert run_multi_objective_task(task, arms, store, [0, 1], report=lambda _: None) == 2
    # The second call is the whole point of the store: raising a tier's seed
    # count must cost the new seeds and not the old ones.
    assert run_multi_objective_task(task, arms, store, [0, 1], report=lambda _: None) == 0
    assert run_multi_objective_task(task, arms, store, [0, 1, 2], report=lambda _: None) == 1

    held = store.usable("toy", "random")
    assert sorted(held) == [0, 1, 2]
    record = held[0]
    assert record.protocol == repr(task)
    assert record.oracle_calls == task.protocol.budget
    # Recorded against *this* module, not `benchmark.methods`: the arms are built
    # here, and a record that under-declares what it depends on cannot notice an
    # edit to the very arm that produced it.
    assert "evogfn.benchmark.multi_objective" in record.source


def test_the_designs_kept_for_inspection_are_the_ones_on_the_measured_front(tmp_path):
    store = ResultStore(tmp_path)
    task = toy_task()
    run_multi_objective_task(task, {"random": ARMS["random"]}, store, [0], report=lambda _: None)
    record = store.usable("toy", "random")[0]
    # Not "the best ten": with several objectives there is no order to take a
    # top ten under without first inventing a trade-off.
    assert record.top_sequences
    assert all(len(design) == task.landscape().sequence_length for design in record.top_sequences)
