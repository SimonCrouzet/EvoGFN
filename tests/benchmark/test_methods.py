"""Tests for the arms themselves: what each one is, and what each one spends.

Two failures are under test, and both had already happened.

**A plate that was never filled.** `Campaign` asked its sampler for one pool,
dropped everything it had measured before, and assayed whatever survived. At the
2048-candidate pool every arm used to get, there were always at least 96 distinct
rows left, so the fault was invisible. Give an arm the pool its own paper
specifies -- a genetic algorithm's population is one plate -- and it showed:
measured directly, a random arm assayed 6 wells of 8 and then 4, a genetic arm 4
and then 3. Half the oracle budget went unspent, no round reported an error, and
the budget every claim is indexed by was wrong by a factor of two in the
direction that flatters whichever method repeated itself least.
`test_every_arm_spends_exactly_the_budget_it_was_given` is the end-to-end check
that no arm can do that again, run at a pool equal to the plate so that the
condition which used to hide it is gone. The duplicate-heavy sampler that makes
the same point against the loop in isolation lives in
``tests/loop/test_campaign.py::TestThePlateIsAlwaysFull``.

**Baselines nobody published.** Every classical arm received a deep ensemble
screening a 2048-candidate pool down to a plate -- a step in none of their
papers -- and the reference arm the headline table paired against was one of
those hybrids. The published pipeline is now the default and every addition is a
named rung, so `test_every_bare_published_arm_is_bare` and
`test_each_arm_gets_the_pool_its_paper_specifies` pin the thing that was wrong
rather than the code that fixed it: an arm silently
regaining a surrogate, or regaining the one global pool size, is a table of
hybrids again and nothing in the numbers would say so.

**Bare is not the same as published.** One arm's *own paper* specifies a
surrogate and a named acquisition rule, so testing it for bareness would enforce
the opposite of the principle. `BARE` and `MODELLED` split the two cases, and
`test_alde_carries_exactly_the_configuration_its_paper_names` is where the
components that would otherwise go missing are written down -- an arm still
called ALDE with a greedy rule and an initialisation-variance ensemble is MLDE
with extra rounds.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from evogfn.acquisition.rules import Thompson
from evogfn.algorithms.baselines import SimulatedAnnealing
from evogfn.algorithms.gflownet.flow_objectives import SubTrajectoryBalance
from evogfn.algorithms.gflownet.objectives import TrajectoryBalance
from evogfn.algorithms.inner_loop import ProxyOptimising
from evogfn.benchmark.methods import (
    BASELINES,
    DEFAULT_POOL,
    OBJECTIVES,
    _RebuiltOnMove,
    anchor_arms,
    gflownet,
    variant_arms,
)
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.tasks import Task
from evogfn.env.mutation import MutationEnvironment, TerminalFeasibilityEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.loop.campaign import ReanchorableSampler
from evogfn.models import AnchorConditionedPolicy, SequencePolicy
from evogfn.surrogate.ensemble import DEFAULT_MEMBERS as PUBLISHED_ENSEMBLE_MEMBERS
from evogfn.surrogate.ensemble import DeepEnsemble

#: Rounds and plate for the end-to-end runs. Small, because the property under
#: test is accounting rather than optimisation, and two rounds is the fewest that
#: can exercise the campaign's memory of an earlier one.
ROUNDS = 2
BATCH = 8

#: A landscape small enough to run every arm against inside a test, with a
#: feasibility constraint so that `genetic-feasible` has something to reject.
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

#: What each arm's own paper says its candidate pool is. `PLATE_POOL` resolves
#: against the task, so the expectation is written the same way: a genetic
#: algorithm's population is its evaluation batch (Stanton et al.), CMA-ES's is
#: lambda, hill climbing proposes a neighbourhood of the current point, the two
#: site-saturation walks propose exactly the designs their protocol names,
#: AdaLead's screening happens inside its own rollout so what it hands up is
#: already a plate, and MLDE's and ALDE's is an exhaustive library they screen on
#: purpose. Three orders of magnitude apart, which is why one global value could
#: not have been right for more than one of them.
#:
#: The screened ablations keep the library pool because a screen with nothing to
#: screen is not a screen: at a pool of one plate the model would rank `BATCH`
#: candidates into `BATCH` wells and change nothing at all.
EXPECTED_POOL = {
    "random": BATCH,
    "hill-climb": BATCH,
    "single-step": BATCH,
    "recomb": BATCH,
    "genetic": BATCH,
    "genetic-feasible": BATCH,
    "cmaes": BATCH,
    "adalead": BATCH,
    "mlde": DEFAULT_POOL,
    "alde": DEFAULT_POOL,
    "random+screen": DEFAULT_POOL,
    "genetic+screen": DEFAULT_POOL,
    "genetic+search": DEFAULT_POOL,
    "genetic+distinct": BATCH,
}

#: Published pipelines whose own papers describe no model at all, so a surrogate
#: in front of them would be a step nobody proposed. This is most of the table,
#: and the comparison the paper is actually about.
BARE = (
    "random",
    "hill-climb",
    "single-step",
    "recomb",
    "genetic",
    "genetic-feasible",
    "cmaes",
    "adalead",
    "mlde",
)

#: Published pipelines that *contain* a model, so bareness is the wrong test for
#: them. ``mlde`` and ``adalead`` are in `BARE` above and not here on purpose:
#: both fit their own regressor internally, so the campaign still hands them
#: nothing. ``alde`` is the one arm whose published configuration is a campaign
#: surrogate read through a stated acquisition rule, which is why it is the only
#: name here.
MODELLED = ("alde",)

#: Every arm that is somebody's published pipeline rather than a decomposition
#: row. Read by the ladder test, which pins the rest of `BASELINES` as rungs.
PUBLISHED = (*BARE, *MODELLED)


def toy_task(*, reanchor: bool = True) -> Task:
    """A task cheap enough to run every arm against end to end."""
    return Task(
        name="toy",
        purpose="a toy, for testing that every arm spends what it was given",
        build=lambda: EhrlichLandscape(**TOY),  # type: ignore[arg-type]
        protocol=Protocol(rounds=ROUNDS, batch_size=BATCH, max_mutations=4),
        max_mutations=4,
        reanchor=reanchor,
        attainable=None,
    )


def settings(arm):
    """What an arm recorded about its own configuration."""
    return arm.parameters


def source_of(arm):
    """The code object behind an arm, which is all a fingerprint can see."""
    return arm.run.__code__


def proxy_spend(campaign):
    """What a finished campaign's sampler reports having spent on the proxy."""
    return campaign.sampler.proxy_calls


def environment_of(campaign):
    """The graph a campaign searches. Always present for the arms in this module."""
    return campaign.environment


def held_policy(sampler):
    """The policy a GFlowNet sampler trains."""
    return sampler._policy


def policy_of(campaign):
    """The policy a GFlowNet campaign's sampler trains."""
    return held_policy(campaign.sampler)


def reanchored_sampler(campaign, env):
    """The sampler a campaign carries into a moved anchor, via the hook it prefers."""
    return campaign.sampler.reanchored(env)


def one_step_from(env):
    """A design one legal substitution away from an environment's anchor."""
    state = env.initial(1)
    action = int(np.flatnonzero(env.forward_mask(state)[0, : env.n_mutation_actions])[0])
    return env.step(state, np.array([action])).sequences[0]


def counting_sampler(*, proxy_calls, bred, unconstructible, proposals=0):
    """Something with a GFlowNet sampler's tallies and nothing else."""
    return SimpleNamespace(
        proxy_calls=proxy_calls,
        bred_designs=bred,
        unconstructible_designs=unconstructible,
        proposals_made=proposals,
        name="stub",
    )


class TestTheLadderIsWhatItSaysItIs:
    """Each arm's name is a claim about its pipeline, checked against the object."""

    def test_the_arms_are_exactly_the_published_pipelines_plus_the_ladder(self):
        # Pinned as a set so that adding an arm is a deliberate edit here. An arm
        # that appears without a rung to stand on is a hybrid back in the table,
        # and an arm that disappears is a comparator a reviewer expects going
        # missing without anyone deciding it should.
        assert set(BASELINES) == set(PUBLISHED) | {
            "random+screen",
            "genetic+screen",
            "genetic+search",
            "genetic+distinct",
        }

    @pytest.mark.parametrize("name", BARE)
    def test_every_bare_published_arm_is_bare(self, name):
        # The failure: a deep ensemble in front of a baseline whose paper has
        # none, which made the headline comparison a comparison between hybrids
        # and made the reference arm one of them.
        campaign = BASELINES[name](toy_task(), 0)

        assert campaign._surrogate is None, f"{name} was handed a surrogate its paper has none of"
        assert not isinstance(campaign.sampler, ProxyOptimising), (
            f"{name} was allowed to optimise against a model its paper has none of"
        )
        assert not campaign._distinct_batch

    @pytest.mark.parametrize("name", BARE)
    def test_no_bare_arm_ranks_on_anything_but_the_prediction(self, name):
        # Greedy is the shipped rule and the axis `alde` is defined by moving.
        # An arm that acquired an uncertainty-reading rule silently would be
        # ranked on a different quantity from the arms beside it, and the only
        # trace would be in results nobody could attribute.
        assert settings(BASELINES[name])["acquisition"] == "Greedy"
        assert settings(BASELINES[name])["bootstrap"] is False

    def test_alde_carries_exactly_the_configuration_its_paper_names(self):
        # Its own authors' bench configuration: one-hot encodings, a five-member
        # DNN ensemble with bootstrapping, and Thompson sampling. Every one of
        # those is a claim about the arm that the arm's name cannot carry, and
        # three of them are settings any refactor could quietly drop -- at which
        # point the row would still be called ALDE and would be MLDE with extra
        # rounds.
        campaign = BASELINES["alde"](toy_task(), 0)
        recorded = settings(BASELINES["alde"])
        ensemble = campaign._surrogate

        assert isinstance(campaign._acquisition, Thompson)
        assert recorded["acquisition"] == "Thompson"
        assert isinstance(ensemble, DeepEnsemble)
        assert ensemble.bootstraps
        assert ensemble.n_members == PUBLISHED_ENSEMBLE_MEMBERS
        # The library screen, not a plate: ALDE ranks a combinatorial library and
        # measures its top, which at a pool of one plate would rank BATCH
        # candidates into BATCH wells and be no screen at all.
        assert campaign._pool_size == DEFAULT_POOL

    def test_the_screened_ablation_and_alde_are_not_the_same_arm(self):
        # They share a surrogate over a library pool, and a reader who took them
        # for one another would read a published pipeline as a decomposition row
        # or the reverse. Three things separate them, and this is where that is
        # written down.
        ablation = BASELINES["genetic+screen"](toy_task(), 0)
        alde = BASELINES["alde"](toy_task(), 0)
        ensemble = ablation._surrogate

        assert type(ablation._acquisition) is not type(alde._acquisition)
        assert isinstance(ensemble, DeepEnsemble)
        assert not ensemble.bootstraps
        assert (
            settings(BASELINES["genetic+screen"])["sampler"]
            != settings(BASELINES["alde"])["sampler"]
        )

    def test_annealing_is_not_an_arm(self):
        # It appears in no baseline table of either lineage this suite is
        # measured against, so it is nobody's expected comparator and must not
        # occupy a results row. The sampler itself is still importable, which is
        # the whole of the difference between "removed" and "not run".
        assert "annealing" not in BASELINES
        assert SimulatedAnnealing is not None

    def test_the_ladder_adds_exactly_one_thing_per_rung(self):
        # Read down the column: nothing, then a screen, then search against the
        # same model. A rung that quietly carried two changes would attribute
        # both to whichever one it was named for.
        task = toy_task()
        bare = BASELINES["genetic"](task, 0)
        screen = BASELINES["genetic+screen"](task, 0)
        search = BASELINES["genetic+search"](task, 0)

        assert (bare._surrogate, screen._surrogate is None, search._surrogate is None) == (
            None,
            False,
            False,
        )
        assert not isinstance(screen.sampler, ProxyOptimising), (
            "the +screen rung filters the pool; its search is still blind"
        )
        assert isinstance(search.sampler, ProxyOptimising)

    def test_only_the_distinct_arm_changes_the_plate_rule(self):
        # `genetic+distinct` is the same algorithm under a different plate rule,
        # so the rule must be the *only* thing separating it from `genetic` --
        # otherwise the arm measures the rule plus whatever else moved.
        task = toy_task()
        bare = BASELINES["genetic"](task, 0)
        distinct = BASELINES["genetic+distinct"](task, 0)

        assert distinct._distinct_batch
        assert not bare._distinct_batch
        assert distinct._surrogate is None
        assert distinct._pool_size == bare._pool_size

    @pytest.mark.parametrize("name", sorted(EXPECTED_POOL))
    def test_each_arm_gets_the_pool_its_paper_specifies(self, name):
        # The setting that hid the budget fault. One global `max(2048, batch * 4)`
        # gave a genetic algorithm a pool twenty times its own population and
        # gave MLDE the same number as a hill climber, and at 2048 there were
        # always enough distinct candidates to fill a plate however badly an arm
        # had converged.
        campaign = BASELINES[name](toy_task(), 0)
        assert campaign._pool_size == EXPECTED_POOL[name]


class TestEveryArmSpendsItsWholeBudget:
    """The invariant every budget-indexed claim in the paper rests on."""

    @pytest.mark.parametrize("name", sorted(BASELINES))
    def test_every_arm_spends_exactly_the_budget_it_was_given(self, name):
        # Run at the pool each arm's paper specifies, which for most of them is
        # exactly one plate -- the configuration under which the shortfall was
        # measured, and which the old global 2048 masked completely.
        result = BASELINES[name](toy_task(), 0).run()

        assert result.oracle_calls == ROUNDS * BATCH, (
            f"{name} spent {result.oracle_calls} of {ROUNDS * BATCH} oracle calls"
        )
        assert [record.evaluated for record in result.rounds] == [BATCH] * ROUNDS
        assert len(result.sequences) == ROUNDS * BATCH

    @pytest.mark.parametrize("name", sorted(BASELINES))
    def test_no_arm_re_measures_a_design_from_an_earlier_round(self, name):
        # Cross-round memory is protocol, on the same footing as re-anchoring: a
        # lab does not re-order a variant whose number is already in the
        # notebook, and an arm that had collapsed onto one mode would otherwise
        # spend its whole budget re-measuring it.
        result = BASELINES[name](toy_task(), 0).run()

        start = 0
        earlier: set[bytes] = set()
        for record in result.rounds:
            plate = np.ascontiguousarray(result.sequences[start : start + record.evaluated])
            start += record.evaluated
            keys = {row.tobytes() for row in plate}
            assert not (keys & earlier), f"{name} re-ordered a design in round {record.index}"
            earlier |= keys

    @pytest.mark.parametrize("name", sorted(BASELINES))
    def test_every_arm_reports_a_duplicate_share_in_range(self, name):
        # The name is a contract: the store reads this off the result by
        # attribute, so a share that goes missing or leaves [0, 1] lands in the
        # table as a silently empty -- or silently wrong -- column.
        result = BASELINES[name](toy_task(), 0).run()

        assert 0.0 <= result.duplicate_fraction <= 1.0

    def test_the_distinct_arm_fills_its_plate_without_a_repeat(self):
        # It has to *fill* the plate, or it would be reporting the other rule's
        # cost as a shorter campaign rather than as a duplicate share -- which is
        # the original fault wearing the ablation's name.
        result = BASELINES["genetic+distinct"](toy_task(), 0).run()

        assert result.oracle_calls == ROUNDS * BATCH
        assert result.duplicate_fraction == 0.0
        assert all(record.duplicates == 0 for record in result.rounds)


class TestAnArmSaysWhatItRanAt:
    """An arm's configuration is recorded, because its name cannot carry it."""

    def test_two_arms_from_one_closure_are_separated_only_by_their_settings(self):
        # The gap a code fingerprint cannot close, stated as an assertion. The
        # beta scan builds nine arms out of this one factory, so all nine are
        # the same source and hash identically; nothing but the recorded
        # settings distinguishes what they ran at, and the arm's name -- the
        # only other candidate -- is a string somebody parses.
        low = gflownet(TrajectoryBalance(), beta=1.0)
        high = gflownet(TrajectoryBalance(), beta=10.0)

        assert source_of(low) is source_of(high)
        assert settings(low)["beta"] == 1.0
        assert settings(high)["beta"] == 10.0

    @pytest.mark.parametrize("name", sorted(BASELINES))
    def test_a_baselines_record_agrees_with_the_campaign_it_builds(self, name):
        # A provenance field that disagrees with the object it describes is
        # worse than an absent one: it reads as fact and cannot be checked
        # against anything, since the campaign it described is long gone by the
        # time anyone reads the record.
        arm = BASELINES[name]
        campaign = arm(toy_task(), 0)
        recorded = settings(arm)

        assert recorded["surrogate"] is (campaign._surrogate is not None)
        assert recorded["proxy_access"] is isinstance(campaign.sampler, ProxyOptimising)
        assert recorded["distinct_batch"] is campaign._distinct_batch
        # `PLATE_POOL` is recorded as the zero it is and resolves per task, so
        # the check is against the plate the task actually measures.
        assert (recorded["pool_size"] or BATCH) == campaign._pool_size

    def test_the_two_genetic_arms_are_told_apart_by_what_built_them(self):
        # Same class, different constructor arguments: a record naming only the
        # sampler's class could not say which of these two it came from, and
        # rejection sampling is the whole content of one of them.
        assert settings(BASELINES["genetic"])["sampler"] == "genetic"
        assert settings(BASELINES["genetic-feasible"])["sampler"] == "feasible-genetic"

    def test_an_objectives_own_setting_is_read_off_the_objective(self):
        # `lam` is resolved inside the objective before this module sees it, so
        # a record built from the arguments alone would report the shipped
        # default for every arm in a lambda scan.
        arm = gflownet(SubTrajectoryBalance(lam=0.25), learn_flow=True)

        assert settings(arm)["objective"] == "SubTrajectoryBalance"
        assert settings(arm)["lam"] == 0.25
        assert "lam" not in settings(OBJECTIVES["gfn-tb"])

    def test_an_unstated_objective_is_recorded_as_unstated(self):
        # Not as the name of whatever the sampler currently defaults to: that
        # would put a choice nobody made into the record, and it would go on
        # claiming it after the default moved.
        assert settings(gflownet())["objective"] == "default"


class TestTheRebuiltArmIsTheSameArmWithoutItsMemory:
    """The amortisation control: one axis moved, and the compute held equal."""

    def test_only_the_carried_arm_can_follow_a_moved_anchor_in_place(self):
        # How the rebuild is forced. The campaign resolves the re-anchoring hook
        # first and falls back to its factory, so an arm that must be rebuilt is
        # one whose sampler does not offer the hook. Were the wrapper to grow a
        # `reanchored` method, this arm would silently become the carried one
        # and the ablation would compare a method against itself.
        rebuilt = anchor_arms()["gfn-tb-rebuilt"](toy_task(), 0)
        carried = anchor_arms()["gfn-tb"](toy_task(), 0)

        assert not isinstance(rebuilt.sampler, ReanchorableSampler)
        assert isinstance(carried.sampler, ReanchorableSampler)

    def test_the_two_arms_agree_on_everything_that_costs_compute(self):
        # The result is worthless if they do not: an arm that starts each ball
        # from nothing *and* gets fewer gradient steps to recover with loses on
        # budget rather than on transfer, and nothing in the numbers separates
        # the two explanations.
        carried = settings(anchor_arms()["gfn-tb"])
        rebuilt = settings(anchor_arms()["gfn-tb-rebuilt"])

        assert carried["carry_policy"] is True
        assert rebuilt["carry_policy"] is False
        assert {k: v for k, v in carried.items() if k != "carry_policy"} == {
            k: v for k, v in rebuilt.items() if k != "carry_policy"
        }

    def test_the_study_reuses_the_shipped_arms_rather_than_rebuilding_them(self):
        # The store keys on (task, arm). An arm re-declared under an existing
        # name from a second expression is the same cell only while the two
        # expressions agree, and the day they stop agreeing the study silently
        # reuses campaigns that ran something else.
        arms = anchor_arms()
        assert arms["gfn-tb"] is OBJECTIVES["gfn-tb"]
        assert arms["genetic"] is BASELINES["genetic"]

    def test_a_rebuilt_arm_totals_its_spend_across_every_anchor(self):
        # A rebuilt sampler starts its counters at zero, so the campaign's
        # compute read off the last one alone would report the final anchor's
        # rounds as the whole run's -- understating exactly the number a reader
        # uses to check that this arm and the carried one cost the same.
        first = counting_sampler(proxy_calls=128, bred=40, unconstructible=10, proposals=96)
        second = counting_sampler(proxy_calls=64, bred=20, unconstructible=0, proposals=96)
        wrapper = _RebuiltOnMove(second, [first, second])

        assert wrapper.proxy_calls == 192
        assert wrapper.bred_designs == 60
        assert wrapper.unconstructible_designs == 10
        assert wrapper.proposals_made == 192
        assert wrapper.unconstructible_fraction == pytest.approx(10 / 60)

    def test_a_rebuilt_arm_that_bred_nothing_reports_a_share_of_nothing(self):
        # Zero, not a division by zero, and it means "nothing was bred" rather
        # than "everything bred was constructible". The two are only
        # distinguishable with `bred_designs` beside it.
        wrapper = _RebuiltOnMove(
            counting_sampler(proxy_calls=1, bred=0, unconstructible=0),
            [counting_sampler(proxy_calls=1, bred=0, unconstructible=0)],
        )

        assert wrapper.unconstructible_fraction == 0.0

    @pytest.mark.slow
    def test_the_two_arms_spend_the_same_proxy_budget_when_the_anchor_moves(self):
        # Measured rather than argued, because it is the premise of the whole
        # comparison. Both retrain every round at the same step count, so the
        # spend is equal by construction -- and this is what would notice if a
        # rebuild ever skipped or doubled a retrain.
        task = toy_task()
        carried = gflownet(TrajectoryBalance(), steps=2)(task, 0)
        rebuilt = gflownet(TrajectoryBalance(), steps=2, carry_policy=False)(task, 0)
        carried.run()
        rebuilt.run()

        assert proxy_spend(carried) > 0
        assert proxy_spend(rebuilt) == proxy_spend(carried)

    @pytest.mark.slow
    def test_with_the_anchor_still_the_two_arms_are_one_campaign(self):
        # Why the fixed-anchor row of the study holds one GFlowNet cell and not
        # two. Nothing is ever rebuilt there, so the axis cannot act, and a
        # second arm would store one experiment under two names -- paying twice
        # and leaving nobody able to check the two rows against each other.
        task = toy_task(reanchor=False)
        carried = gflownet(TrajectoryBalance(), steps=2)(task, 0).run()
        rebuilt = gflownet(TrajectoryBalance(), steps=2, carry_policy=False)(task, 0).run()

        assert np.array_equal(carried.sequences, rebuilt.sequences)
        assert carried.best_value == rebuilt.best_value


class TestTheVariantLadderIsAddedRatherThanApplied:
    """Two mechanisms, each off by default, so every existing result still stands.

    The failure guarded against is a variant that leaks. Both of these change
    what a campaign searches -- one the graph, one the policy's input -- so an
    arm that acquired either silently would be tabled against arms it is no
    longer comparable to, and nothing in the stored numbers would say which rows
    those were. `genetic-feasible` is named separately below because it reaches
    the same environment class through `feasible_only` and is the arm most
    likely to be moved by accident.
    """

    def variant_env(self, name):
        return environment_of(variant_arms()[name](toy_task(), 0))

    def variant_policy(self, name):
        return policy_of(variant_arms()[name](toy_task(), 0))

    def test_every_shipped_arm_still_masks_feasibility_at_every_step(self):
        for name, arm in {**BASELINES, **OBJECTIVES}.items():
            assert type(environment_of(arm(toy_task(), 0))) is MutationEnvironment, name

    def test_genetic_feasible_is_untouched(self):
        # It reaches the environment through `feasible_only`, so it is the arm a
        # change to the default masking rule would move without being named. Its
        # rejection burden is the control for the feasibility claim, and a
        # rejection burden measured on a graph that no longer rejects is nothing.
        campaign = BASELINES["genetic-feasible"](toy_task(), 0)
        assert type(environment_of(campaign)) is MutationEnvironment

    def test_every_shipped_gflownet_arm_still_sees_only_the_sequence(self):
        for name, arm in OBJECTIVES.items():
            assert type(policy_of(arm(toy_task(), 0))) is SequencePolicy, name

    def test_the_terminal_arm_searches_the_terminal_only_graph(self):
        assert isinstance(self.variant_env("gfn-tb+terminal"), TerminalFeasibilityEnvironment)
        assert type(self.variant_env("gfn-tb+anchor")) is MutationEnvironment

    def test_the_anchor_arm_conditions_its_policy_on_the_anchor(self):
        policy = self.variant_policy("gfn-tb+anchor")
        assert isinstance(policy, AnchorConditionedPolicy)
        assert type(self.variant_policy("gfn-tb+terminal")) is SequencePolicy

    def test_the_both_arm_takes_both_and_the_two_do_not_interfere(self):
        campaign = variant_arms()["gfn-tb+terminal+anchor"](toy_task(), 0)
        assert isinstance(environment_of(campaign), TerminalFeasibilityEnvironment)
        assert isinstance(policy_of(campaign), AnchorConditionedPolicy)

    def test_the_conditioned_policy_starts_bound_to_the_graph_it_walks(self):
        # A held anchor can disagree with the environment's, and the
        # disagreement is silent: the policy conditions on a parent nobody is
        # searching from and the loss stays finite.
        campaign = variant_arms()["gfn-tb+anchor"](toy_task(), 0)
        assert policy_of(campaign).anchor.tolist() == environment_of(campaign).parent.tolist()

    def test_the_conditioned_policy_follows_a_moved_anchor(self):
        # The campaign prefers the sampler's own re-anchoring hook over its
        # factory, so this is the path a re-anchoring task actually takes, and
        # it never passes through the factory that built the policy.
        campaign = variant_arms()["gfn-tb+anchor"](toy_task(), 0)
        env = environment_of(campaign)
        moved_to = env.reanchored(one_step_from(env))
        moved = reanchored_sampler(campaign, moved_to)

        assert held_policy(moved) is policy_of(campaign), "no weight should be rebuilt"
        assert held_policy(moved).anchor.tolist() == moved_to.parent.tolist()

    def test_each_variant_changes_exactly_the_flag_it_is_named_for(self):
        # An arm that also moved the step count, the reward exponent or the
        # trunk width would be losing or winning on compute, and the ladder
        # would report that as a feasibility or conditioning result.
        base = settings(variant_arms()["gfn-tb"])
        flags = ("terminal_feasibility", "anchor_conditioned")
        for name in ("gfn-tb+terminal", "gfn-tb+anchor", "gfn-tb+terminal+anchor"):
            resolved = settings(variant_arms()[name])
            assert {k: v for k, v in resolved.items() if k not in flags} == {
                k: v for k, v in base.items() if k not in flags
            }, name

    def test_the_flags_are_off_wherever_they_are_not_named(self):
        assert settings(variant_arms()["gfn-tb"])["terminal_feasibility"] is False
        assert settings(variant_arms()["gfn-tb"])["anchor_conditioned"] is False
        assert settings(variant_arms()["gfn-tb+terminal"])["anchor_conditioned"] is False
        assert settings(variant_arms()["gfn-tb+anchor"])["terminal_feasibility"] is False
        both = settings(variant_arms()["gfn-tb+terminal+anchor"])
        assert both["terminal_feasibility"] is True
        assert both["anchor_conditioned"] is True

    def test_the_ladder_reuses_the_shipped_arm_rather_than_rebuilding_it(self):
        # The store keys on (task, arm), so a re-declared `gfn-tb` is the same
        # cell only while two expressions agree. Looked up, they cannot disagree.
        assert variant_arms()["gfn-tb"] is OBJECTIVES["gfn-tb"]

    def test_the_variants_are_not_in_the_objective_sweep(self):
        # `OBJECTIVES` is resolved key-by-key to a training objective by the
        # configuration sweep. Neither variant is an objective -- both are plain
        # trajectory balance -- so a key here would be looked up as one and fail.
        assert not set(variant_arms()) - {"gfn-tb"} & set(OBJECTIVES)

    @pytest.mark.slow
    def test_a_variant_arm_runs_a_campaign_end_to_end(self):
        # Cheap, and the only check that the two mechanisms survive contact with
        # sampling, training and a moved anchor rather than only with unit
        # fixtures. A dead-ended trajectory reaching the assay is the expected
        # behaviour of the terminal-only arm, not an error, so this must pass.
        arm = gflownet(
            TrajectoryBalance(), steps=2, terminal_feasibility=True, anchor_conditioned=True
        )
        result = arm(toy_task(), 0).run()
        assert result.sequences.shape[0] == ROUNDS * BATCH
