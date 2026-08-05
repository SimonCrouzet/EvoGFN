"""Tests for the MOGFN-PC arm: what it is wired to, and what it degenerates to.

A separate file from `test_multi_objective.py` because the failures here are the
arm's own rather than the suite's, and because they are the ones that cannot be
read off any indicator column.

**The plate collapsed onto one trade-off at the last step.** If this arm were
handed a campaign-level surrogate, `Campaign._design` would score the whole
preference-diverse pool under the *single* uniform preference the acquisition
rule carries and `TopK` would take the best of that -- destroying the arm's
advantage after the pool had been built, silently, with every column looking
healthy. `surrogate=None` is what prevents it, and it is pinned here because it
reads like an omission.

**An `(n, k)` prediction reaching the ledger.** The same wiring, from the other
side: `Campaign._correlation` builds `isfinite(predicted) & isfinite(measured)`
with a `(n,)` `measured`, so an `(n, k)` prediction broadcasts and then raises
deep inside the round record -- after a plate has been measured.

**An arm that is not the ablation's twin.** The comparison MOGFN-PC exists for is
against `gfn-tb-pref{N}` at the same total budget and the *same* N trade-offs. An
arm pointed at a different set of preferences is not that comparison, and nothing
in the numbers would say so.

**A scalarisation nobody chose.** Under a weighted sum, "one model covers the
whole front" is false on any concave region at every omega (Miettinen 1999, Thm
3.1.4). Which scalarisation this arm trains under is therefore part of building
it, so it is a constructor parameter from the first commit rather than a
substitution to be made later.

The end-to-end runs are a two-round toy with a four-design plate, matching
`test_multi_objective.py`: enough to exercise every seam, cheap enough to run on
every commit.
"""

import numpy as np
import pytest

from evogfn.algorithms.gflownet.objectives import ContrastiveBalance, TrajectoryBalance
from evogfn.algorithms.gflownet.preference_sampler import PreferenceConditionedSampler
from evogfn.benchmark.multi_objective import (
    DEFAULT_POOL,
    PC_PREFERENCE_COUNT,
    PREFERENCE_COUNTS,
    MultiObjectiveTask,
    arms_for_tier,
    multi_objective_tiers,
    preference_arms,
    preference_conditioned_arm,
    preference_vectors,
    set_indicators,
)
from evogfn.benchmark.protocol import Protocol
from evogfn.landscapes.multi_ehrlich import MultiEhrlichLandscape
from evogfn.loop.campaign import Campaign, ReanchorableSampler
from evogfn.rewards.base import TemperedReward
from evogfn.rewards.scalarization import ScalarizedReward, Tchebycheff, WeightedSum

ROUNDS = 2
BATCH = 4


def toy_landscape(n_objectives=2):
    """A multi-Ehrlich instance at full conflict, so its front is not one point."""
    return MultiEhrlichLandscape.with_conflict(
        sequence_length=16,
        vocab_size=4,
        n_objectives=n_objectives,
        n_motifs=1,
        motif_length=4,
        quantization=4,
        max_spacing=2,
        transition_density=0.5,
        conflict=1.0,
        seed=3,
    )


def toy_task(*, rounds=ROUNDS, batch=BATCH, n_objectives=2):
    """A multi-objective task cheap enough to run end to end."""
    return MultiObjectiveTask(
        name="toy",
        purpose="a toy, for testing that the wiring does what the table says",
        build=lambda: toy_landscape(n_objectives),
        protocol=Protocol(rounds=rounds, batch_size=batch, max_mutations=4),
        max_mutations=4,
        reanchor=True,
        reference_point=(0.0,) * n_objectives,
        front=None,
        front_is_exact=False,
    )


def arm(**kwargs):
    """The arm at a training budget small enough for a unit test."""
    kwargs.setdefault("steps", 4)
    return preference_conditioned_arm(**kwargs)


class TestWhatTheCampaignIsWiredTo:
    def test_the_arm_runs_without_a_campaign_level_surrogate(self):
        # Not an omission. A campaign surrogate here would re-rank the whole
        # preference-diverse pool under one uniform preference and hand the plate
        # to that trade-off, which is the collapse the interleaving exists to
        # prevent -- and it would happen after the pool was built, so nothing
        # about the pool would show it.
        campaign = arm()(toy_task(), 0)
        assert isinstance(campaign, Campaign)
        assert campaign._surrogate is None

    def test_the_model_lives_on_the_sampler_instead(self):
        # The cost of `surrogate=None` is that `surrogate_correlation` is nan and
        # `screened` equals the pool for this arm and no other. That is honest --
        # there is no campaign-level surrogate -- but it is a column the other
        # arms populate, so it is pinned here rather than discovered in a table.
        campaign = arm()(toy_task(), 0)
        sampler = campaign.sampler
        assert isinstance(sampler, PreferenceConditionedSampler)
        assert sampler.surrogate.n_objectives == 2
        result = campaign.run()
        assert all(np.isnan(record.surrogate_correlation) for record in result.rounds)
        assert all(record.screened >= record.evaluated for record in result.rounds)

    def test_it_can_follow_a_moved_anchor_without_being_rebuilt(self):
        # The campaign checks the outermost object. Without this the arm goes
        # down the factory path at every move, and although the factory closes
        # over the policy and the surrogate, the sampler's accumulated
        # measurements and its counters would restart.
        campaign = arm()(toy_task(), 0)
        assert isinstance(campaign.sampler, ReanchorableSampler)

    def test_the_arm_spends_exactly_its_budget(self):
        task = toy_task()
        result = arm()(task, 0).run()
        assert result.oracle_calls == task.protocol.budget
        assert [record.evaluated for record in result.rounds] == [BATCH] * ROUNDS

    def test_the_arm_reports_both_indicators(self):
        task = toy_task()
        result = arm()(task, 0).run()
        got = set_indicators(task, result)
        assert got["best"] is not None
        assert got["best"] >= 0.0
        # No reference front on the toy task, so IGD+ is legitimately absent
        # rather than zero; what matters is that the arm does not put a spurious
        # number in a column it cannot fill.
        assert got["regret"] is None or got["regret"] >= 0.0

    def test_the_compute_it_spent_is_reported(self):
        # The amortisation claim is a compute claim as much as a coverage one, so
        # a zero here would leave half of it unmeasured while looking measured.
        campaign = arm()(toy_task(), 0)
        campaign.run()
        assert int(getattr(campaign.sampler, "proxy_calls", 0)) > 0

    def test_it_gets_the_library_pool_rather_than_the_plate(self):
        # A plate-sized pool would leave one design per preference at eight
        # trade-offs, so the per-preference ranking would rank nothing.
        campaign = arm()(toy_task(), 0)
        assert campaign._pool_size == DEFAULT_POOL


class TestItIsTheAblationsTwin:
    def test_it_is_queried_at_the_same_trade_offs_the_ablation_is_trained_at(self):
        # `gfn-tb-pref{N}` splits the budget across `preference_vectors(k, N)`.
        # MOGFN-PC must be queried at *that* grid or the two rows of F2's table
        # are covering different parts of the front and the comparison measures
        # the grids rather than the amortisation.
        campaign = arm(preferences=4)(toy_task(), 0)
        sampler = campaign.sampler
        assert isinstance(sampler, PreferenceConditionedSampler)
        np.testing.assert_allclose(sampler.preferences, preference_vectors(2, 4, seed=0))

    def test_the_default_grid_matches_the_widest_rung_of_the_ablation(self):
        # One policy against N replicas is only a comparison at the same N.
        assert max(PREFERENCE_COUNTS) == PC_PREFERENCE_COUNT

    def test_the_arm_joins_the_tier_that_holds_the_ablation(self):
        arms = preference_arms()
        assert "mogfn-pc" in arms
        assert {f"gfn-tb-pref{n}" for n in PREFERENCE_COUNTS} <= set(arms)

    def test_the_diagnostic_tier_can_reach_the_whole_three_row_decomposition(self):
        # The conditioned arm, the replicated ablation, the population baseline,
        # and the floor without which a hypervolume is a number rather than a
        # result. Before this the tier held only the middle row.
        tier = next(t for t in multi_objective_tiers(1, 1) if t.name == "preferences")
        names = set(arms_for_tier(tier))
        assert {"mogfn-pc", "gfn-tb-pref1", "nsga2", "random"} <= names

    def test_the_other_tiers_are_untouched(self):
        # Registering the arm must not change what any tier that carries results
        # runs, because those tiers' arm lists are pinned by
        # `test_multi_objective.py` and by every table read off them.
        tier = next(t for t in multi_objective_tiers(1, 1) if t.name == "main")
        assert "mogfn-pc" not in arms_for_tier(tier)


class TestTheScalarisationIsChosenRatherThanInherited:
    def test_the_default_is_the_weighted_sum_every_other_arm_ranks_under(self):
        # So that the arm and the campaign that scores it are on one trade-off
        # scale, and so that this arm is not the only one carrying a reference
        # point of its own.
        campaign = arm()(toy_task(), 0)
        sampler = campaign.sampler
        assert isinstance(sampler, PreferenceConditionedSampler)
        assert isinstance(sampler.reward.scalarization, WeightedSum)

    def test_tchebycheff_is_a_one_argument_substitution(self):
        # F5 in the hypothesis ledger: under a weighted sum no omega reaches a
        # concave region of the front, so "one model covers the whole front"
        # would be false by construction however well the conditioning worked.
        # The run under Tchebycheff is the check, and it has to be cheap.
        task = toy_task()
        campaign = arm(scalarization=Tchebycheff())(task, 0)
        sampler = campaign.sampler
        assert isinstance(sampler, PreferenceConditionedSampler)
        assert isinstance(sampler.reward.scalarization, Tchebycheff)
        assert campaign.run().oracle_calls == task.protocol.budget

    def test_the_objective_is_a_parameter_and_contrastive_balance_is_legal(self):
        # `Z` cancels exactly in a contrasted pair provided both members share
        # their omega, which one-omega-per-batch guarantees. It is the free
        # robustness check on the `Z_theta(omega)` head.
        task = toy_task()
        campaign = arm(objective=ContrastiveBalance())(task, 0)
        assert campaign.run().oracle_calls == task.protocol.budget

    def test_an_objective_that_reads_the_scalar_log_z_is_refused_when_built(self):
        # Refused where the arm is assembled, before the campaign exists and long
        # before an oracle call. A scalar log Z fitted across every omega biases
        # the policy toward one trade-off and the symptom -- a collapsed front --
        # is indistinguishable from the null this experiment is testing for.
        with pytest.raises(ValueError, match="scalar log Z"):
            arm(objective=TrajectoryBalance())(toy_task(), 0)


class TestItDegeneratesWhereItShould:
    def test_a_degenerate_preference_makes_the_reward_single_objective(self):
        # omega = e_0 under a weighted sum is objective 0 alone, so the reward the
        # policy trains against is exactly the `TemperedReward` a single-objective
        # GFlowNet uses. If this ever stopped holding, the arm's behaviour at the
        # corners of the simplex would be something nobody chose.
        values = np.array([[0.5, 4.0], [2.0, 0.25], [1.0, 1.0]])
        scalar = TemperedReward(beta=3.0)
        conditioned = ScalarizedReward(WeightedSum(), [1.0, 0.0], reward=scalar)
        np.testing.assert_allclose(conditioned.log_reward(values), scalar.log_reward(values[:, :1]))

    def test_a_single_preference_grid_leaves_one_undivided_slice(self):
        # At one trade-off the arm is structurally `gfn-tb` with a conditioning
        # input that never moves at inference. That it still runs is what makes
        # the grid size a dial rather than a rewrite -- and it is the control the
        # amortisation claim is read against at N = 1.
        campaign = arm(preferences=1)(toy_task(), 0)
        sampler = campaign.sampler
        assert isinstance(sampler, PreferenceConditionedSampler)
        assert sampler.preferences.shape == (1, 2)
        np.testing.assert_allclose(sampler.preferences[0], [0.5, 0.5])
        sampler.propose(6)
        origin = sampler.last_preference_index
        assert origin is not None
        assert set(np.unique(origin)) == {0}

    def test_a_single_objective_task_is_refused(self):
        # A simplex over one objective is the constant [1.0]. Training a policy
        # conditioned on it would produce an arm whose name claims a trade-off
        # nothing in the run varies.
        with pytest.raises(ValueError, match="at least 2 objectives"):
            arm()(toy_task(n_objectives=1), 0)


def test_the_arm_runs_at_three_objectives():
    # Two objectives make the simplex an interval, where an even grid is exact.
    # Three is where `preference_vectors` starts drawing instead, and where the
    # conditioning vector stops being a pair -- both paths the arm has to survive.
    task = toy_task(n_objectives=3)
    result = arm(preferences=3)(task, 0).run()
    assert result.oracle_calls == task.protocol.budget
    assert result.values.shape[1] == 3


def test_two_seeds_give_two_campaigns():
    # The arm is built from one seed for the policy, the surrogate and the grid.
    # A seed that reached only some of them would make paired comparisons pair
    # runs that shared less than the pairing assumes.
    first = arm()(toy_task(), 0).run()
    second = arm()(toy_task(), 1).run()
    assert not np.array_equal(first.sequences, second.sequences)
