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

**And ours is not published either.** `ADAPTED` is the third case: a published
pipeline running a parameter of ours in place of one of its own. The bareness
and budget rules apply to it unchanged, but it must never join `PUBLISHED`,
because a row we configured quoted under somebody's citation is the same failure
as a hybrid -- reported at the level of the arm rather than of its components.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from evogfn.acquisition.rules import Thompson
from evogfn.algorithms.baselines import SimulatedAnnealing
from evogfn.algorithms.baselines.mlde import (
    ADAPTED_TRAINING_SIZE,
    DEFAULT_TRAINING_SIZE,
    PUBLISHED_BUDGET,
)
from evogfn.algorithms.gflownet.flow_objectives import SubTrajectoryBalance
from evogfn.algorithms.gflownet.objectives import TrajectoryBalance
from evogfn.algorithms.inner_loop import ProxyOptimising
from evogfn.benchmark.methods import (
    BASELINES,
    DEFAULT_BETA,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_POOL,
    DEFAULT_TRAINING_STEPS,
    OBJECTIVES,
    SELECTED_CONFIGURATION,
    LadderBase,
    _environment,
    _parameter_count,
    _RebuiltOnMove,
    anchor_arms,
    constrains_construction,
    gflownet,
    matched_capacity,
    matched_capacity_for,
    reproduced_rungs,
    shipped_base,
    variant_arms,
)
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.selection import _build_objective
from evogfn.benchmark.suite import DIAGNOSTIC_INSTANCE, _arm_parameters
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
    "mlde-over-budget": DEFAULT_POOL,
    "mlde+earlyfit": DEFAULT_POOL,
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
    "mlde-over-budget",
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

#: Arms that are a published pipeline with a parameter of *ours* in place of one
#: of its own. Not `PUBLISHED`, and the separation is the whole point: the same
#: bareness and budget rules apply to them, but a table that listed them among
#: the published rows would be quoting our engineering under somebody's
#: citation. They wear the ``+`` for the reason ``cmaes+dp`` does.
ADAPTED = ("mlde+earlyfit",)

#: Rounds an arm runs beyond its task's own protocol, by name. Empty for every
#: arm but one, and it has to stay that way: the harness's whole pairing argument
#: is that one protocol reaches every arm, so an arm spending more is a departure
#: that has to be declared somewhere a test can read it.
#:
#: MLDE is the departure. Its protocol is a training set of 384 plus a designed
#: plate, so on this suite's four-plate budget the published method has no
#: configuration at all -- its training set alone is the whole budget. The
#: compressed arm and the over-budget arm are both run, and this is what says
#: which is which.
EXTRA_ROUNDS = {"mlde-over-budget": 1}


def budget_of(name):
    """Oracle calls an arm is entitled to on `toy_task`, which is not the same for all."""
    return (ROUNDS + EXTRA_ROUNDS.get(name, 0)) * BATCH


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


def unconstrained_task():
    """The same toy with every transition allowed, so every assay yields a value.

    Feasibility is what `TOY` exists to exercise, and it is exactly what has to be
    absent for a question about a *training set* to be well posed: a sampler that
    gathers measurements counts the ones that came back with a number, and an
    infeasible design comes back with none.
    """
    return Task(
        name="unconstrained",
        purpose="a toy with no feasibility constraint, for supervised handover",
        build=lambda: EhrlichLandscape(**{**TOY, "transition_density": 1.0}),  # type: ignore[arg-type]
        protocol=Protocol(rounds=ROUNDS, batch_size=BATCH, max_mutations=4),
        max_mutations=4,
        reanchor=True,
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


def mlde_of(campaign):
    """The MLDE sampler a campaign holds, read for its own budget accounting.

    Its training split, whether it ever fitted, and how far below the published
    sample it sits are all properties of `MLDE` rather than of `Sampler`, and the
    campaign is typed as holding the latter.
    """
    return campaign.sampler


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


#: A ladder base that does not depend on which configuration is currently
#: selected. The shape tests below hold for any base, and pinning them to
#: `results/selected.json` would make them fail the day a selection moves for
#: reasons that have nothing to do with the two mechanisms. Trajectory balance
#: at the module defaults, so `arm` and the fields beside it describe the same
#: configuration -- which is itself the precondition every rung rests on.
LADDER = LadderBase(
    name="base",
    arm=gflownet(TrajectoryBalance()),
    objective=TrajectoryBalance(),
    learn_flow=False,
    beta=DEFAULT_BETA,
    steps=DEFAULT_TRAINING_STEPS,
    hidden_dim=DEFAULT_HIDDEN_DIM,
)


def rung(suffix):
    """A ladder rung's name, which is the base arm's name plus what it adds."""
    return f"{LADDER.name}{suffix}"


def diagnostic_task():
    """A task on the diagnostic landscape, at 32 positions over 20 tokens."""
    return shaped_task("diagnostic", **DIAGNOSTIC_INSTANCE)


def shaped_task(name, **instance):
    """A task on an Ehrlich instance of a named shape.

    A parameter count is a statement about *one* sizing -- the sequence length
    fixes the trunk's input width and the alphabet fixes the action head's output
    -- so the capacity control has to be measured on each shape the tier runs, and
    a helper that varies the shape is what makes "each shape" testable at all.
    """
    return Task(
        name=name,
        purpose="an instance of a named shape, for sizing the capacity control",
        build=lambda: EhrlichLandscape(**instance),
        protocol=Protocol(rounds=ROUNDS, batch_size=BATCH, max_mutations=4),
        max_mutations=4,
        reanchor=True,
        attainable=None,
    )


def parameter_count(arm, task=None):
    """Learnable parameters in the policy an arm builds on a task.

    Counted off the policy the arm itself constructs rather than off one built to
    match it, so what is measured is the network that would train.
    """
    return sum(
        int(p.numel())
        for p in policy_of(arm(diagnostic_task() if task is None else task, 0)).parameters()
    )


def recorded_selection():
    """What `results/selected.json` holds, or ``None`` when nothing is recorded."""
    if not SELECTED_CONFIGURATION.exists():
        return None
    return json.loads(SELECTED_CONFIGURATION.read_text())


class TestTheLadderIsWhatItSaysItIs:
    """Each arm's name is a claim about its pipeline, checked against the object."""

    def test_the_arms_are_exactly_the_published_pipelines_plus_the_ladder(self):
        # Pinned as a set so that adding an arm is a deliberate edit here. An arm
        # that appears without a rung to stand on is a hybrid back in the table,
        # and an arm that disappears is a comparator a reviewer expects going
        # missing without anyone deciding it should.
        assert set(BASELINES) == set(PUBLISHED) | set(ADAPTED) | {
            "random+screen",
            "genetic+screen",
            "genetic+search",
            "genetic+distinct",
            # Ours, not published: the feasibility claim's construction control,
            # a GA masked the same way a GFlowNet policy is -- see `_masked_genetic`.
            "genetic-masked",
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
        #
        # The entitlement is per arm rather than global because one arm is
        # deliberately over budget, and both halves of that matter: an arm that
        # spent less than its own entitlement is the original shortfall, and an
        # arm that spent more than `EXTRA_ROUNDS` declares has taken a budget
        # nobody granted it -- which is the same fault as the shortfall, wearing
        # the other sign, and would be invisible in a table indexed by the task's
        # protocol.
        budget = budget_of(name)
        result = BASELINES[name](toy_task(), 0).run()

        assert result.oracle_calls == budget, (
            f"{name} spent {result.oracle_calls} of {budget} oracle calls"
        )
        assert [record.evaluated for record in result.rounds] == [BATCH] * (budget // BATCH)
        assert len(result.sequences) == budget

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


class TestMldeIsRunAtItsOwnBudgetAsWellAsOurs:
    """The one arm that spends more than its task's budget, and why it is allowed.

    MLDE's published protocol is 384 screened variants plus a designed plate:
    480 assays, against the 384 this suite gives everyone. There is no
    configuration in which it fits -- its *training set alone* is the whole
    budget -- so the compressed arm trains on one plate where Wittmann et al.
    train on four, which their own results say is a weaker MLDE.

    The failure this class is arranged against is not that the over-budget arm
    exists but that it could silently stop being MLDE. Two ways, both quiet:
    it takes the extra assays and screens at random for the whole campaign,
    never fitting anything, in which case a random baseline is sitting in the
    table under a supervised method's name; or the shared protocol moves, the
    arm's training split moves with it, and the row goes on being described as
    the published 384 + 96 while being neither.
    """

    def four_plate_task(self):
        """A task on the protocol the headline table runs: four plates of 96."""
        return Task(
            name="four-plate",
            purpose="the shared protocol, for checking MLDE resolves to its own",
            build=lambda: EhrlichLandscape(**TOY),  # type: ignore[arg-type]
            protocol=Protocol(rounds=4, batch_size=96, max_mutations=4),
            max_mutations=4,
            reanchor=True,
            attainable=None,
        )

    def test_on_the_shared_protocol_the_arm_resolves_to_the_published_numbers(self):
        # Derived rather than pinned: the training split is "every plate but the
        # last", which *is* 384 and 96 on four plates of 96 and would be
        # something else on any other shape. So this is the assertion that says
        # whether the arm is still Wittmann et al.'s split -- and the day the
        # shared protocol moves it fails, rather than the arm quietly running a
        # different split under the published protocol's name.
        campaign = BASELINES["mlde-over-budget"](self.four_plate_task(), 0)

        assert campaign.budget == PUBLISHED_BUDGET
        assert mlde_of(campaign).required_budget == PUBLISHED_BUDGET
        assert not mlde_of(campaign).runs_below_published_training_size

    def test_the_compressed_arm_is_the_one_that_does_not_fit_the_published_split(self):
        # Both rows are run and they are not the same method. Were the compressed
        # arm to quietly acquire the published training size it would spend its
        # whole budget screening and design nothing, and were the over-budget arm
        # to lose it the table would carry two identical rows under two names.
        compressed = BASELINES["mlde"](self.four_plate_task(), 0)
        published = BASELINES["mlde-over-budget"](self.four_plate_task(), 0)

        assert mlde_of(compressed).runs_below_published_training_size
        assert not mlde_of(published).runs_below_published_training_size
        assert compressed.budget < published.budget

    def test_the_extra_plate_is_spent_on_designs_and_not_on_more_screening(self):
        # The silent failure: an arm handed a larger budget that never reaches
        # its own training size screens at random for every round, ends unfitted,
        # and is reported as MLDE. `is_fitted` is the only thing that separates
        # "a supervised method ran" from "a random baseline ran under its name",
        # and nothing about the spend or the plate count would show it.
        #
        # Run on the unconstrained landscape because the handover is gated on
        # *usable* measurements: `MLDE.observe` drops an infeasible assay, having
        # no fitness to regress on, so on a constrained landscape the screening
        # plates yield fewer training examples than they cost and the handover
        # slips. That interaction belongs to `mlde` as much as to this arm and is
        # not what this test is about -- it is recorded in `_mlde_as_published`.
        campaign = BASELINES["mlde-over-budget"](unconstrained_task(), 0)
        campaign.run()

        assert mlde_of(campaign).is_fitted
        assert mlde_of(campaign).training_examples == budget_of("mlde-over-budget")

    def test_only_the_over_budget_arm_departs_from_the_tasks_protocol(self):
        # The harness's pairing argument is that one protocol reaches every arm.
        # Exactly one arm is exempt, it is named for the exemption, and a second
        # arm acquiring an extra round would break the pairing everywhere while
        # every individual campaign still looked correct.
        for name, arm in BASELINES.items():
            campaign = arm(toy_task(), 0)
            assert campaign._rounds == ROUNDS + EXTRA_ROUNDS.get(name, 0), name
        assert set(EXTRA_ROUNDS) <= set(BASELINES)
        # The name is the only place the departure is legible in a results table:
        # the store writes the *task's* protocol beside every record, so a row
        # spending more than that protocol says while being named like the arms
        # beside it is a budget-indexed comparison that is silently not one.
        assert all("over-budget" in name for name in EXTRA_ROUNDS)


class TestTheAdaptedMldeArmIsOursAndCanFit:
    """The third MLDE row: ours, at everybody's budget, sized so a fit can happen.

    `mlde` and `mlde-over-budget` gate their handover on *usable* measurements,
    and on a constrained landscape most wells return nothing to regress on. At
    the suite's measured feasible share of 0.053 a 384-assay campaign buys about
    20 training examples against their 96 and 384, so neither ever fits and both
    rows are random screens under a supervised method's name. That is the
    finding, both arms keep reporting it, and this arm is what makes it readable:
    at a training size a constrained screen returns, the pair separates "a
    constrained space suits this method badly" from "this method never fitted".

    The failure this class is arranged against is the arm quietly becoming one of
    the other two. It fits here and the default does not, and that gap is the
    whole content of the row -- if an edit closes it in either direction the
    table carries three arms answering one question, or a fourth random screen.
    """

    def four_round_task(self, *, density):
        """Four plates of 32, at whichever transition density is asked for.

        Four rounds because the handover is tested at a round boundary, and 32 so
        that three screening plates *could* return `mlde`'s 96 measurements --
        which is what makes the constrained run below a statement about
        feasibility rather than about the budget.
        """
        return Task(
            name=f"density-{density}",
            purpose="four plates, for whether MLDE's handover can happen at all",
            build=lambda: EhrlichLandscape(**{**TOY, "transition_density": density}),
            protocol=Protocol(rounds=4, batch_size=32, max_mutations=4),
            max_mutations=4,
            reanchor=True,
            attainable=None,
        )

    def test_it_reaches_a_fit_where_the_shipped_arm_never_does(self):
        # The measurement the arm was added for. Both spend 128 assays on the
        # same landscape; the default gathers a fraction of the 96 usable
        # measurements it waits for and screens at random to the end, and
        # `is_fitted` is the only thing in either result that says which of the
        # two campaigns was supervised.
        task = self.four_round_task(density=0.5)
        adapted = BASELINES["mlde+earlyfit"](task, 0)
        default = BASELINES["mlde"](task, 0)
        adapted.run()
        default.run()

        assert mlde_of(adapted).is_fitted
        assert not mlde_of(default).is_fitted
        assert mlde_of(default).training_examples < 96

    def test_the_shipped_arm_fits_the_same_protocol_once_feasibility_is_removed(self):
        # What makes the row above about feasibility and not about the budget.
        # Given the same four plates with every transition allowed, the default
        # arm gathers its 96 usable measurements and hands over -- so the only
        # thing it lacked on the constrained landscape was wells that reported.
        campaign = BASELINES["mlde"](self.four_round_task(density=1.0), 0)
        campaign.run()

        assert mlde_of(campaign).is_fitted

    def test_it_buys_the_fit_with_the_budget_it_was_already_given(self):
        # The deviation is a parameter of ours, not an assay of ours. An arm that
        # reached its handover by taking extra plates would be a second
        # `mlde-over-budget` -- comparable to nothing in the table beside it --
        # and the extra spend would be invisible in the stored protocol.
        task = self.four_round_task(density=0.5)
        result = BASELINES["mlde+earlyfit"](task, 0).run()

        assert result.oracle_calls == 4 * 32
        assert [record.evaluated for record in result.rounds] == [32] * 4
        assert settings(BASELINES["mlde+earlyfit"])["extra_rounds"] == 0

    def test_the_two_arms_it_was_added_beside_are_unchanged(self):
        # They are the finding. Lowering `mlde`'s training size to make it fit,
        # rather than adding a third arm, would delete the result and leave the
        # suite with two arms answering the same question.
        four_plate = Task(
            name="four-plate",
            purpose="the shared protocol, for checking the other arms did not move",
            build=lambda: EhrlichLandscape(**TOY),  # type: ignore[arg-type]
            protocol=Protocol(rounds=4, batch_size=96, max_mutations=4),
            max_mutations=4,
            reanchor=True,
            attainable=None,
        )
        compressed = mlde_of(BASELINES["mlde"](four_plate, 0))
        published = mlde_of(BASELINES["mlde-over-budget"](four_plate, 0))
        adapted = mlde_of(BASELINES["mlde+earlyfit"](four_plate, 0))

        assert compressed.required_budget == DEFAULT_TRAINING_SIZE + 96
        assert published.required_budget == PUBLISHED_BUDGET
        assert adapted.required_budget == ADAPTED_TRAINING_SIZE + 96

    def test_it_is_bare_and_ranks_on_the_prediction_like_the_arms_it_pairs_with(self):
        # It fits its own ensemble, so a campaign surrogate in front of it would
        # be a second model filtering the first one's output -- and it would make
        # the pair with `mlde` a comparison between a hybrid and a baseline
        # rather than between two training sizes.
        campaign = BASELINES["mlde+earlyfit"](toy_task(), 0)

        assert campaign._surrogate is None
        assert not isinstance(campaign.sampler, ProxyOptimising)
        assert settings(BASELINES["mlde+earlyfit"])["acquisition"] == "Greedy"

    def test_the_name_and_the_record_both_mark_the_row_as_ours(self):
        # A reader must not take this for the published method at any point they
        # meet it. The registry key carries the `+` that `cmaes+dp` uses for the
        # same purpose, and the record names the builder rather than the class --
        # which is the only field distinguishing this campaign from `mlde`'s once
        # the objects are gone, both being an `MLDE` over a library pool.
        assert set(ADAPTED) <= set(BASELINES)
        assert all("+" in name for name in ADAPTED)
        assert not set(ADAPTED) & set(PUBLISHED)
        assert settings(BASELINES["mlde+earlyfit"])["sampler"] == "mlde-adapted"
        assert settings(BASELINES["mlde"])["sampler"] == "mlde"


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
        # The store writes the *task's* protocol beside these parameters, so this
        # is the only field in the record that says an arm ran longer than that
        # protocol. A recorded zero on a campaign with an extra round would
        # describe the run as within budget while it was not, and every
        # budget-indexed comparison downstream would read it that way.
        assert recorded["extra_rounds"] == campaign._rounds - ROUNDS

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

    Built on `LADDER` rather than on whatever `results/selected.json` currently
    holds: what these assert is the ladder's *shape*, which must hold for any
    base, and a fixed base keeps them from turning red the day a selection moves
    for reasons that have nothing to do with the mechanisms.
    `TestTheLadderIsBuiltOnTheConfigurationThatShips` is where the default base
    is checked against the file, which is the other half.
    """

    def variant_env(self, suffix):
        return environment_of(variant_arms(LADDER)[rung(suffix)](toy_task(), 0))

    def variant_policy(self, suffix):
        return policy_of(variant_arms(LADDER)[rung(suffix)](toy_task(), 0))

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
        assert isinstance(self.variant_env("+terminal"), TerminalFeasibilityEnvironment)
        assert type(self.variant_env("+anchor")) is MutationEnvironment

    def test_the_anchor_arm_conditions_its_policy_on_the_anchor(self):
        policy = self.variant_policy("+anchor")
        assert isinstance(policy, AnchorConditionedPolicy)
        assert type(self.variant_policy("+terminal")) is SequencePolicy

    def test_the_both_arm_takes_both_and_the_two_do_not_interfere(self):
        campaign = variant_arms(LADDER)[rung("+terminal+anchor")](toy_task(), 0)
        assert isinstance(environment_of(campaign), TerminalFeasibilityEnvironment)
        assert isinstance(policy_of(campaign), AnchorConditionedPolicy)

    def test_the_conditioned_policy_starts_bound_to_the_graph_it_walks(self):
        # A held anchor can disagree with the environment's, and the
        # disagreement is silent: the policy conditions on a parent nobody is
        # searching from and the loss stays finite.
        campaign = variant_arms(LADDER)[rung("+anchor")](toy_task(), 0)
        assert policy_of(campaign).anchor.tolist() == environment_of(campaign).parent.tolist()

    def test_the_conditioned_policy_follows_a_moved_anchor(self):
        # The campaign prefers the sampler's own re-anchoring hook over its
        # factory, so this is the path a re-anchoring task actually takes, and
        # it never passes through the factory that built the policy.
        campaign = variant_arms(LADDER)[rung("+anchor")](toy_task(), 0)
        env = environment_of(campaign)
        moved_to = env.reanchored(one_step_from(env))
        moved = reanchored_sampler(campaign, moved_to)

        assert held_policy(moved) is policy_of(campaign), "no weight should be rebuilt"
        assert held_policy(moved).anchor.tolist() == moved_to.parent.tolist()

    def test_each_variant_changes_exactly_the_flag_it_is_named_for(self):
        # An arm that also moved the step count, the reward exponent, the
        # objective or the trunk width would be losing or winning on compute,
        # and the ladder would report that as a feasibility or conditioning
        # result. `+wide` is exempted on the capacity flag alone because capacity
        # *is* what it varies -- and on nothing else, which is the whole of its
        # claim to be a capacity control rather than a second configuration. Its
        # static `hidden_dim` is *not* exempted and must equal the base's: the
        # width it trains at is resolved per task, so a static width that differed
        # would be a second configuration hiding in the settings.
        arms = variant_arms(LADDER)
        base = settings(arms[LADDER.name])
        flags = ("terminal_feasibility", "anchor_conditioned")
        for suffix in ("+terminal", "+anchor", "+terminal+anchor"):
            resolved = settings(arms[rung(suffix)])
            assert {k: v for k, v in resolved.items() if k not in flags} == {
                k: v for k, v in base.items() if k not in flags
            }, suffix
        control = settings(arms[rung("+wide")])
        assert {k: v for k, v in control.items() if k != "match_anchor_capacity"} == {
            k: v for k, v in base.items() if k != "match_anchor_capacity"
        }

    def test_the_flags_are_off_wherever_they_are_not_named(self):
        arms = variant_arms(LADDER)
        assert settings(arms[LADDER.name])["terminal_feasibility"] is False
        assert settings(arms[LADDER.name])["anchor_conditioned"] is False
        assert settings(arms[rung("+terminal")])["anchor_conditioned"] is False
        assert settings(arms[rung("+anchor")])["terminal_feasibility"] is False
        assert settings(arms[rung("+wide")])["anchor_conditioned"] is False
        both = settings(arms[rung("+terminal+anchor")])
        assert both["terminal_feasibility"] is True
        assert both["anchor_conditioned"] is True

    def test_the_ladder_reuses_the_base_arm_rather_than_rebuilding_it(self):
        # The store keys on (task, arm), so a re-declared base is the same cell
        # only while two expressions agree. Handed the arm, they cannot disagree.
        assert variant_arms(LADDER)[LADDER.name] is LADDER.arm

    def test_the_variants_are_not_in_the_objective_sweep(self):
        # `OBJECTIVES` is resolved key-by-key to a training objective by the
        # configuration sweep. No rung is an objective -- each is the base
        # objective with a mechanism turned on -- so a key here would be looked
        # up as one and fail.
        assert not set(variant_arms(LADDER)) - {LADDER.name} & set(OBJECTIVES)

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


class TestTheLadderIsBuiltOnTheConfigurationThatShips:
    """The base rung is what the project runs, not what it inherited.

    The failure this catches had already been made. The ladder looked up
    ``gfn-tb`` as its base while the configuration selected over 3,100 campaigns
    is ``gfn-subtb``, and nothing in the resulting table would have said so: every
    rung would have been correctly one step above its base, every arm would have
    been internally consistent, and the study would simply have answered a
    question about a different method. That this repository's own beta scan
    reverses direction between the two objectives is what makes it a real error
    rather than a pedantic one -- an effect measured on one is not evidence about
    the other here.

    It is silent in the other direction too. A selection that moves and a ladder
    that does not would leave the ladder's base and the headline table's GFlowNet
    arm as two different configurations under one story, so what is asserted below
    is not "the base is sub-trajectory balance" -- which would need editing every
    time selection ran -- but "the base is whatever the file says", which cannot
    go out of date.
    """

    def test_the_default_base_is_the_arm_the_selection_recorded(self):
        choice = recorded_selection()
        if choice is None:
            pytest.skip("no selection recorded, so there is no shipped arm to check against")
        assert shipped_base().name == choice["arm"]
        assert set(variant_arms()) >= {choice["arm"], f"{choice['arm']}+terminal"}

    def test_the_base_rung_is_the_arm_the_headline_table_would_run(self):
        # Built through the selection phase's own builder, from the same file, so
        # the ladder's base cell and the headline's GFlowNet cell are the same
        # campaign. Compared on the recorded settings rather than on identity,
        # because the two are separate calls and only what they resolve to is
        # ever stored.
        choice = recorded_selection()
        if choice is None:
            pytest.skip("no selection recorded, so there is no shipped arm to check against")
        headline = _build_objective(
            choice["objective"],
            float(choice["beta"]),
            int(choice["steps"]),
            lam=None if choice["lam"] is None else float(choice["lam"]),
            mix=None if choice["mix"] is None else float(choice["mix"]),
            hidden_dim=int(choice["hidden_dim"]),
        )
        assert settings(variant_arms()[choice["arm"]]) == settings(headline)

    def test_the_base_rung_is_not_the_untuned_default_it_used_to_be(self):
        # The regression itself, stated as a regression. `gfn-tb` at the
        # inherited defaults is a configuration the selection rejected, and a
        # ladder that quietly returned to it would read exactly like one that had
        # never left.
        choice = recorded_selection()
        if choice is None:
            pytest.skip("no selection recorded, so the untuned default is what ships")
        assert settings(variant_arms()[choice["arm"]]) != settings(OBJECTIVES["gfn-tb"])

    def test_a_caller_may_hand_the_ladder_its_base(self):
        # The seam that keeps this module from owning a results file: a caller
        # with a configuration in hand passes it, and every test above builds a
        # ladder without a file on disk at all.
        assert set(variant_arms(LADDER)) == {
            "base",
            "base+terminal",
            "base+anchor",
            "base+terminal+anchor",
            "base+wide",
            # The decomposition rungs: neither trains, and they differ from each
            # other only in how much surrogate the screen is allowed to spend.
            # The oracle budget -- the plate -- is the same on all of them and on
            # the base, so the sweep varies compute against a free model rather
            # than the budget the campaign is indexed by.
            "base+untrained",
            "base+untrained@p2048",
            "base+untrained@p8192",
            "base+untrained@p32768",
        }

    def test_every_arm_in_the_ladder_carries_the_base_arms_name(self):
        # The store keys on (task, arm), and every arm here is defined *relative
        # to* a base: the two mechanism rungs by the flag they turn on, the
        # capacity control by a width derived from the base's own -- 101 matches
        # a conditioned trunk of 64 and matches nothing else. An arm named
        # without the prefix is therefore one store cell for every base it was
        # ever built against, and two selections' worth of campaigns would
        # average into a single row with nothing to say they had.
        base = LadderBase(
            name="other-base",
            arm=gflownet(TrajectoryBalance(), hidden_dim=32),
            objective=TrajectoryBalance(),
            learn_flow=False,
            beta=DEFAULT_BETA,
            steps=DEFAULT_TRAINING_STEPS,
            hidden_dim=32,
        )
        assert all(
            name == base.name or name.startswith(f"{base.name}+") for name in variant_arms(base)
        )

    def test_two_bases_share_no_arm_name_at_all(self):
        # The drift stated as the collision it produces. These two ladders differ
        # in the trunk width their control is matched against -- 32 and the
        # module default -- so the two controls are different networks, and under
        # a name that did not carry the base they were the same cell. This is the
        # regression the ladder rebuild fixed for the rungs and left open here.
        narrow = LadderBase(
            name="narrow",
            arm=gflownet(TrajectoryBalance(), hidden_dim=32),
            objective=TrajectoryBalance(),
            learn_flow=False,
            beta=DEFAULT_BETA,
            steps=DEFAULT_TRAINING_STEPS,
            hidden_dim=32,
        )
        shape = {"sequence_length": 32, "vocab_size": 20, "learn_flow": False}
        assert (
            matched_capacity(32, **shape).hidden_dim
            != matched_capacity(LADDER.hidden_dim, **shape).hidden_dim
        ), "the two bases must size their controls differently or this proves nothing"
        assert not set(variant_arms(narrow)) & set(variant_arms(LADDER))

    def test_with_no_selection_recorded_the_base_is_what_the_table_reports(self):
        # Not a placeholder. `methods_for` falls back to the untuned arms when no
        # selection exists, so `gfn-tb` is genuinely what the headline reports in
        # that state -- and looked up, so the base rung is the identical store
        # cell the objectives diagnostic already paid for.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                "evogfn.benchmark.methods.SELECTED_CONFIGURATION",
                SELECTED_CONFIGURATION.parent / "no-such-selection.json",
            )
            base = shipped_base()
        assert base.name == "gfn-tb"
        assert base.arm is OBJECTIVES["gfn-tb"]

    def test_a_selection_that_stopped_partway_refuses_rather_than_defaulting(self, tmp_path):
        # Absent and null are different claims. A file missing an axis describes
        # a configuration no rule ever chose, and building the ladder on the
        # default for that axis would report an untuned setting as selected.
        partial = tmp_path / "selected.json"
        partial.write_text(json.dumps({"objective": "gfn-subtb", "arm": "x", "beta": 0.1}))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("evogfn.benchmark.methods.SELECTED_CONFIGURATION", partial)
            with pytest.raises(ValueError, match="unfinished"):
                shipped_base()

    def test_a_selection_whose_name_and_settings_disagree_refuses(self, tmp_path):
        # The name is the store key every confirmation campaign was written
        # under. A file whose settings build something else would run one
        # configuration and read another's hundred seeds as its own.
        lying = tmp_path / "selected.json"
        lying.write_text(
            json.dumps(
                {
                    "objective": "gfn-subtb",
                    "arm": "gfn-subtb@b0.1-s300-l0.9-h64",
                    "beta": 0.2,
                    "steps": 300,
                    "lam": 0.9,
                    "mix": None,
                    "hidden_dim": 64,
                }
            )
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("evogfn.benchmark.methods.SELECTED_CONFIGURATION", lying)
            with pytest.raises(ValueError, match="never happened"):
                shipped_base()

    def test_a_breeding_selection_refuses_rather_than_dropping_to_plain_balance(self, tmp_path):
        # Genetic-GFN is built by a different factory, and that factory takes
        # neither mechanism flag. Falling back to trajectory balance would put a
        # ladder in the table under the shipped configuration's name while
        # studying an arm nobody selected.
        bred = tmp_path / "selected.json"
        bred.write_text(
            json.dumps(
                {
                    "objective": "genetic-gfn",
                    "arm": "genetic-gfn@b0.5-s300-m0.25-h64",
                    "beta": 0.5,
                    "steps": 300,
                    "lam": None,
                    "mix": 0.25,
                    "hidden_dim": 64,
                }
            )
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("evogfn.benchmark.methods.SELECTED_CONFIGURATION", bred)
            with pytest.raises(ValueError, match="neither ladder mechanism"):
                shipped_base()


#: How far the capacity control may sit from the arm it controls for, as a share
#: of that arm's parameter count. Two percent is far below any effect the ladder
#: could report and far above the residual no integer width can remove -- the
#: worst of the shipped shapes is 1.07% -- so a failure here means the
#: architecture moved rather than that the arithmetic is imprecise.
CAPACITY_TOLERANCE = 0.02

#: Every shape the headline tier puts the control on, and the widths and residuals
#: the module's docstrings quote for the shipped trunk of 64 with a flow head.
#: Written out because they are the numbers a reader is asked to trust: a
#: docstring quoting a residual that stopped being true is worse than one quoting
#: none, since nothing tells the reader which it is looking at.
#:
#: 32 is the diagnostic instance; 64 is `feasibility` and both protocol tasks;
#: four is `gb1-anchor` and `trpb-anchor`, whose Ehrlich stand-in here is exact
#: for this purpose -- a parameter count reads the length and the alphabet and
#: nothing else about a landscape.
SHIPPED_SHAPES = (
    (32, 20, 101, 179952, 179715),
    (64, 20, 103, 355728, 354435),
    (4, 20, 88, 27123, 26835),
)


class TestTheCapacityControlIsTheSizeOfTheArmItControlsFor:
    """The `+wide` control exists because `+anchor` changes two things at once.

    Conditioning widens the trunk's *input* -- state embedding, anchor embedding
    and one difference indicator per position -- so the conditioned policy is half
    again as large as the plain one at the same trunk width. Without a plain
    policy of matching size in the table, every reading of "conditioning helped"
    is equally a reading of "more parameters helped", and no column distinguishes
    them.

    Every assertion here is taken **per shape**, because a parameter count is a
    statement about one sizing and the headline tier runs three. Matched once at
    the diagnostic shape, this control carried 1.63% *fewer* parameters than the
    arm it controls for on the 64-position tasks -- an under-resourced control,
    which is the one direction that manufactures the effect the arm exists to rule
    out, on precisely the tasks the claims are drawn from.

    What would otherwise be silent is the drift. The widths are integers and the
    counts are quadratics, so the match is arithmetic that holds for one
    architecture; add a layer to either policy, change the embedding width, give
    the conditioned trunk a second head, and the control silently stops
    controlling. Nothing raises, no campaign fails, and the ladder goes on
    printing a comparison whose premise has quietly gone.
    """

    def base(self):
        choice = recorded_selection()
        if choice is None:
            pytest.skip("no selection recorded, so there is no shipped width to match")
        return shipped_base()

    def shapes(self):
        """One task per shape the headline tier runs the control on."""
        return [
            shaped_task(
                f"shape-{length}",
                sequence_length=length,
                vocab_size=20,
                n_motifs=1,
                motif_length=2,
                transition_density=0.5,
                seed=3,
            )
            for length in (32, 64, 8)
        ]

    @pytest.mark.parametrize("length", [32, 64, 8])
    def test_the_control_and_the_conditioned_arm_are_the_same_size_on_every_shape(self, length):
        base = self.base()
        arms = variant_arms()
        task = next(t for t in self.shapes() if t.name == f"shape-{length}")
        conditioned = parameter_count(arms[f"{base.name}+anchor"], task)
        control = parameter_count(arms[f"{base.name}+wide"], task)

        assert abs(control - conditioned) <= CAPACITY_TOLERANCE * conditioned

    @pytest.mark.parametrize("length", [32, 64, 8])
    def test_the_control_errs_upward_rather_than_downward_on_every_shape(self, length):
        # The direction is the whole safety property, and it is the property the
        # single-shape sizing broke: at 64 positions the control was the *smaller*
        # network. A control with slightly more capacity than the arm it controls
        # for can only understate conditioning's effect; one with slightly less
        # manufactures the effect the arm exists to rule out, and the table cannot
        # tell the two apart.
        base = self.base()
        arms = variant_arms()
        task = next(t for t in self.shapes() if t.name == f"shape-{length}")

        assert parameter_count(arms[f"{base.name}+wide"], task) >= parameter_count(
            arms[f"{base.name}+anchor"], task
        )

    @pytest.mark.parametrize("length", [32, 64, 8])
    def test_no_narrower_width_would_do(self, length):
        # The rule is the narrowest plain trunk that is not smaller, so one width
        # down must fall short. Without this the control could drift arbitrarily
        # wide and still pass the tolerance above by luck of the tolerance.
        base = self.base()
        task = next(t for t in self.shapes() if t.name == f"shape-{length}")
        width = matched_capacity_for(task, base.hidden_dim, learn_flow=base.learn_flow).hidden_dim
        conditioned = parameter_count(variant_arms()[f"{base.name}+anchor"], task)
        narrower = gflownet(
            base.objective,
            beta=base.beta,
            steps=base.steps,
            learn_flow=base.learn_flow,
            hidden_dim=width - 1,
        )

        assert parameter_count(narrower, task) < conditioned

    @pytest.mark.parametrize(("length", "vocab", "width", "plain", "conditioned"), SHIPPED_SHAPES)
    def test_the_shipped_shapes_are_the_counts_the_docstrings_state(
        self, length, vocab, width, plain, conditioned
    ):
        base = self.base()
        if base.hidden_dim != 64 or not base.learn_flow:
            pytest.skip("the stated counts are for the shipped width of 64 with a flow head")

        match = matched_capacity(
            base.hidden_dim, sequence_length=length, vocab_size=vocab, learn_flow=base.learn_flow
        )

        assert (match.hidden_dim, match.parameters, match.target) == (width, plain, conditioned)
        assert match.residual > 0

    def test_a_single_width_would_be_short_on_one_of_the_shapes(self):
        # The bug, stated as the measurement that found it. 101 matches the
        # diagnostic shape and is 1.63% *under* at 64 positions -- so this asserts
        # that resolving per task is load-bearing rather than tidy, and it fails
        # the day the architecture changes enough for one width to serve.
        base = self.base()
        if base.hidden_dim != 64 or not base.learn_flow:
            pytest.skip("the stated counts are for the shipped width of 64 with a flow head")
        shape = {"vocab_size": 20, "learn_flow": base.learn_flow}
        diagnostic = matched_capacity(base.hidden_dim, sequence_length=32, **shape)
        longer = matched_capacity(base.hidden_dim, sequence_length=64, **shape)

        assert diagnostic.hidden_dim != longer.hidden_dim
        assert (
            _parameter_count(
                diagnostic.hidden_dim,
                sequence_length=64,
                vocab_size=20,
                learn_flow=base.learn_flow,
                anchor_conditioned=False,
            )
            < longer.target
        )

    def test_the_record_carries_the_width_that_trained_and_the_residual_it_achieved(self):
        # Where a reader can find it. The width is not in the arm's name and is
        # not the one in its static settings -- that number is the *conditioned*
        # trunk being matched -- so a record reporting the nominal 64 would
        # describe a policy that never existed on any task.
        base = self.base()
        arms = variant_arms()
        task = self.shapes()[1]
        control = arms[f"{base.name}+wide"]
        match = matched_capacity_for(task, base.hidden_dim, learn_flow=base.learn_flow)

        recorded = _arm_parameters(control, task)

        assert recorded["hidden_dim"] == match.hidden_dim
        assert recorded["capacity_parameters"] == parameter_count(control, task)
        assert recorded["capacity_target"] == parameter_count(arms[f"{base.name}+anchor"], task)
        assert recorded["capacity_residual"] == pytest.approx(match.residual)
        assert float(recorded["capacity_residual"]) > 0

    def test_an_arm_that_matches_nothing_records_no_capacity_fields(self):
        # The overlay is per arm, not per table. An arm without the flag has a
        # width of its own and must not acquire a residual it never computed.
        recorded = _arm_parameters(OBJECTIVES["gfn-tb"], self.shapes()[0])

        assert recorded["hidden_dim"] == DEFAULT_HIDDEN_DIM
        assert "capacity_residual" not in recorded

    def test_matching_and_conditioning_at_once_is_refused(self):
        # The control is the conditioned arm's size *without* the conditioning.
        # An arm that is both would be its own control, and the search has no
        # target to aim at.
        with pytest.raises(ValueError, match="capacity control for itself"):
            gflownet(TrajectoryBalance(), anchor_conditioned=True, match_anchor_capacity=True)


class TestTwoRungsCannotBeMeasuredOnAnUnconstrainedTask:
    """`+terminal` defers a rule that a landscape without a matrix does not have.

    Where nothing constrains construction, `TerminalFeasibilityEnvironment` and
    `MutationEnvironment` describe the identical graph: the rung is the base arm's
    own campaign, and `+terminal+anchor` is `+anchor`'s. Running them costs the
    campaigns twice and puts two pairs of identical rows in a headline table,
    which reads as "we tested the mechanism here and it made no difference" --
    a different and false claim from "there was nothing here to test".

    The condition is read off the environment rather than off a list of task
    names, which is what makes it survive a task gaining a transition matrix. A
    list would go on naming the same rows as reproductions after the mechanism
    became measurable on them.
    """

    def unconstrained_task(self, name):
        """A task whose landscape publishes no transition matrix.

        The structural shape of `gb1-anchor` and `trpb-anchor`, which are the two
        tasks this applies to and which cannot be built here without their
        datasets. An Ehrlich instance always carries a matrix -- even at density
        1.0 -- so "no constraint" has to be modelled by removing the attribute the
        environment reads, which is exactly what the empirical landscapes do by
        never having it.
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
                sequence_length=16,
                vocab_size=20,
                n_motifs=1,
                motif_length=2,
                seed=4,
            ),
            protocol=Protocol(rounds=ROUNDS, batch_size=BATCH, max_mutations=4),
            max_mutations=4,
            attainable=None,
        )

    def test_an_unconstrained_task_reproduces_both_terminal_rungs(self):
        base = shipped_base()
        loose = self.unconstrained_task("loose")

        assert not constrains_construction(loose)
        assert reproduced_rungs(base, task=loose) == {
            f"{base.name}+terminal": base.name,
            f"{base.name}+terminal+anchor": f"{base.name}+anchor",
        }

    def test_a_constrained_task_reproduces_nothing(self):
        # The guard, in the direction that matters: a task with a matrix has a
        # mechanism to measure, so every rung must run. Derived rather than listed,
        # so a task that gains one crosses this line without an edit.
        base = shipped_base()
        constrained = shaped_task(
            "constrained",
            sequence_length=16,
            vocab_size=20,
            n_motifs=1,
            motif_length=2,
            transition_density=0.15,
            seed=5,
        )

        assert constrains_construction(constrained)
        assert reproduced_rungs(base, task=constrained) == {}

    def test_a_dense_matrix_is_still_a_matrix(self):
        # The distinction the derivation has to get right, and the one a
        # "nothing is forbidden here" reading would collapse: a landscape that
        # permits every adjacency still *has* the rule, so the two environments
        # are still different classes and the rung still runs. Reproducing there
        # would be a copy standing in for a campaign that could have differed.
        base = shipped_base()
        dense = shaped_task(
            "dense",
            sequence_length=16,
            vocab_size=20,
            n_motifs=1,
            motif_length=2,
            transition_density=1.0,
            seed=6,
        )

        assert constrains_construction(dense)
        assert reproduced_rungs(base, task=dense) == {}

    def test_the_two_environments_really_are_the_same_graph_where_it_says_so(self):
        # The premise, measured rather than asserted. If these two ever disagreed
        # on an unconstrained task, every reproduced row would be a fabrication.
        loose = self.unconstrained_task("loose-graph")
        landscape = loose.landscape()
        plain = _environment(loose, landscape)
        deferred = _environment(loose, landscape, terminal_feasibility=True)
        state = plain.initial(4)

        assert type(plain) is not type(deferred)
        assert np.array_equal(plain.forward_mask(state), deferred.forward_mask(state))
        assert plain.n_actions == deferred.n_actions
