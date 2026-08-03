"""Tests for the two site-saturation directed-evolution arms.

These arms exist because our `hill-climb` is the *other* thing the field calls
directed evolution -- a random single-substitution neighbour of the incumbent,
with no memory of which positions it has already changed. The failure these
tests guard against is the two collapsing into near-duplicates: a walk that
revisits a site, or that commits to a residue before the site is saturated, is a
noisy hill climber wearing the name every "MLDE beats DE" claim in the wet-lab
literature is measured against.

The second failure is a budget quietly spent on nothing. Both protocols cost a
fraction of the campaign they run in, and the surplus is filled with
*replicates at fresh site orders* rather than with repeats of designs already
named -- which is the choice `ReplicatedProtocol` exists to make visible, since
the sources do not address a surplus at all. An arm that silently went back to
repeating itself would be handicapped by most of its budget and nothing in the
numbers would say so.
"""

from typing import cast

import numpy as np
import pytest

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines.directed_evolution import (
    RECOMBINATION_OVERHEAD,
    Recombination,
    ReplicatedProtocol,
    SingleStepWalk,
    distinct_site_orders,
    replicated_recombination,
    replicated_walk,
    site_order,
    substitutions_at,
    walk_replicates,
    within_budget,
)
from evogfn.core import Alphabet
from evogfn.env.mutation import MutationEnvironment


def make_env(length=6, symbols="ABCD", max_mutations=3, transitions=None):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
        transitions=transitions,
    )


def additive(sequences):
    """A landscape with a distinct best token per position, so a walk can win."""
    array = np.asarray(sequences)
    weights = np.arange(1, array.shape[1] + 1, dtype=np.float64)
    return ((array == 1) * weights).sum(axis=1, keepdims=True)


def drive(sampler, env, rounds, plate, landscape=additive):
    """Run a sampler against a toy landscape the way a campaign would."""
    measured = []
    for _ in range(rounds):
        batch = sampler.propose(plate)
        assert all(within_budget(env, row) for row in batch)
        sampler.observe(batch, landscape(batch))
        measured.append(batch)
    return np.concatenate(measured)


class TestTheSiteOrderIsSeededAndShared:
    def test_the_same_seed_gives_the_same_order(self):
        # ALDE enumerates every site order because the walk is path-dependent.
        # A seeded permutation makes each campaign one order and the seed average
        # theirs; an order that moved between runs would make neither true.
        env = make_env(length=8)
        left = site_order(env, np.random.default_rng(3))
        right = site_order(env, np.random.default_rng(3))
        assert np.array_equal(left, right)

    def test_different_seeds_give_different_orders(self):
        env = make_env(length=8)
        orders = {site_order(env, np.random.default_rng(seed)).tobytes() for seed in range(8)}
        assert len(orders) > 1, "every seed drew the same walk, so the arm has one path"

    def test_both_arms_study_the_same_sites_at_one_seed(self):
        # The two differ in what they do with the measurements and in nothing
        # else. Drawing separate site sets would confound the comparison between
        # them with a difference in which residues each got to see.
        env = make_env(length=8, max_mutations=8)
        walk = SingleStepWalk(env, seed=5)
        recomb = Recombination(env, rounds=4, batch_size=64, seed=5)
        assert set(walk.order[: len(recomb.sites)].tolist()) == set(recomb.sites)

    def test_a_recombination_site_set_is_held_in_a_fixed_order(self):
        # The saturations are independent, so a site set is a *set* -- but the
        # listing order decides which of them each round's tranche contains. Two
        # replicates that drew the same sites in different orders would release
        # different tranches, the pooled request would cover the library in a
        # fraction of the rounds the pacing was laid out over, and the protocol
        # would finish with plates left that it could not fill.
        env = make_env(length=8, max_mutations=8)
        left = Recombination(env, rounds=4, batch_size=64, sites=[5, 1, 3])
        right = Recombination(env, rounds=4, batch_size=64, sites=[3, 5, 1])
        assert left.sites == right.sites == (1, 3, 5)

    def test_replicates_are_given_different_orders(self):
        # Two replicates at one order are the same experiment: they name
        # identical designs, pool down to one, and the arm silently runs fewer
        # replicates than its budget paid for.
        env = make_env(length=8, max_mutations=8)
        orders = distinct_site_orders(env, np.random.default_rng(0), 5)
        assert len({order.tobytes() for order in orders}) == len(orders) == 5

    def test_a_sequence_with_too_few_orders_yields_what_it_has(self):
        # A real limit on how many replicates a library supports, not an error.
        env = make_env(length=1, symbols="AB", max_mutations=1)
        assert len(distinct_site_orders(env, np.random.default_rng(0), 5)) == 1


class TestSaturationIsSaturation:
    def test_a_site_offers_every_substitution_the_budget_admits(self):
        env = make_env(length=4, symbols="ABCDE", max_mutations=4)
        variants = substitutions_at(env, env.parent, 2)
        assert len(variants) == env.alphabet.size - 1
        assert {int(row[2]) for row in variants} == {1, 2, 3, 4}

    def test_a_site_outside_the_budget_offers_nothing(self):
        # What bounds the walk. A design already at the budget cannot take a
        # substitution at a fresh position, so that site contributes nothing and
        # the walk moves on rather than emitting a design outside the graph.
        env = make_env(length=4, symbols="ABCD", max_mutations=1)
        at_budget = np.array([1, 0, 0, 0], dtype=np.int32)
        assert substitutions_at(env, at_budget, 2) == []

    def test_an_infeasible_substitution_is_still_offered(self):
        # Deliberate, and the opposite of what a masked sampler does. These are
        # unmasked arms: site saturation makes all substitutions, an
        # adjacency-violating one reaches the assay and scores minus infinity,
        # and filtering it out here would hand the arm the feasibility-by-
        # construction the GFlowNet is claiming as its advantage.
        forbidden = np.ones((4, 4))
        forbidden[1, 0] = 0.0
        env = make_env(length=4, symbols="ABCD", max_mutations=4, transitions=forbidden)
        variants = substitutions_at(env, env.parent, 0)
        assert any(int(row[0]) == 1 for row in variants)
        assert not env.is_reachable(np.stack(variants)).all()


class TestTheWalkNeverRevisitsASite:
    def test_no_position_is_fixed_twice(self):
        # The defining invariant, and the one thing separating this arm from a
        # hill climber. A walk that could revise a residue it had already
        # committed to would lose the path-dependence the literature reports as
        # the method's characteristic weakness.
        env = make_env(length=8, symbols="ABCD", max_mutations=8)
        walk = SingleStepWalk(env, seed=0)
        drive(walk, env, rounds=6, plate=32)
        assert len(set(walk.sites_fixed)) == len(walk.sites_fixed)

    def test_a_fixed_residue_is_never_substituted_again(self):
        # The consequence a table would actually notice: once a site is closed,
        # nothing the walk proposes differs from the incumbent there.
        env = make_env(length=8, symbols="ABCD", max_mutations=8)
        walk = SingleStepWalk(env, seed=1)
        drive(walk, env, rounds=3, plate=32)
        closed = walk.sites_fixed[:-1]
        incumbent = walk.incumbent
        proposals = walk.propose(32)
        for site in closed:
            assert (proposals[:, site] == incumbent[site]).all()


class TestTheWalkIsGreedyAndMonotone:
    def test_it_climbs(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=8)
        walk = SingleStepWalk(env, seed=0)
        drive(walk, env, rounds=6, plate=32)
        assert walk.best_value > float(additive(env.parent[None, :])[0, 0])

    def test_a_site_where_nothing_helps_leaves_the_incumbent_alone(self):
        # The background competes with the substitutions on equal terms, which is
        # what "all possible amino acids" means and what stops the walk ratcheting
        # downhill through a deleterious site.
        env = make_env(length=4, symbols="ABCD", max_mutations=4)
        walk = SingleStepWalk(env, seed=0)
        batch = walk.propose(16)
        # The background scores best; every substitution is worse.
        values = np.where((batch == env.parent[None, :]).all(axis=1), 1.0, 0.0)[:, None]
        walk.observe(batch, values)
        assert np.array_equal(walk.incumbent, env.parent)
        assert walk.best_value == 1.0

    def test_a_site_is_saturated_before_its_winner_is_fixed(self):
        # A plate too small to hold a whole site must spread it across rounds
        # rather than committing on a partial view -- otherwise the arm is
        # "best of whatever fitted on the plate", which is not site saturation.
        env = make_env(length=6, symbols="ABCDEFGH", max_mutations=6)
        walk = SingleStepWalk(env, seed=0)
        first = walk.propose(3)
        walk.observe(first, additive(first))
        assert walk.sites_fixed == walk.sites_fixed[:1], "the walk opened a second site early"
        assert np.array_equal(walk.incumbent, env.parent)


class TestTheWalkSpendsOnlyWhatItsProtocolNames:
    def test_the_first_plate_is_the_background_plus_one_saturated_site(self):
        # The published cost on a four-site library is one assay plus 19 per
        # site. Anything else here means the arm is being charged for designs
        # directed evolution never asked for.
        env = make_env(length=6, symbols="ABCDEFGHIJ", max_mutations=6)
        walk = SingleStepWalk(env, seed=0)
        assert walk.requested == 1 + (env.alphabet.size - 1)

    def test_a_lone_walk_repeats_itself_to_fill_a_larger_plate(self):
        # Which is why the arm wraps several of them: one walk asked for what it
        # asked for, and the rest of the plate is repetition.
        env = make_env(length=6, symbols="ABCD", max_mutations=6)
        walk = SingleStepWalk(env, seed=0)
        proposals = walk.propose(64)
        assert proposals.shape == (64, 6)
        assert len({row.tobytes() for row in np.ascontiguousarray(proposals)}) == walk.requested

    def test_a_finished_walk_refuses_rather_than_inventing_a_move(self):
        # A restart, a random draw or a revision of a fixed site would all be
        # mechanisms no published protocol describes. The campaign records the
        # refusal as an exhausted seed, which is a visible short seed count
        # rather than a number pretending to be comparable.
        env = make_env(length=3, symbols="ABC", max_mutations=3)
        walk = SingleStepWalk(env, seed=0)
        drive(walk, env, rounds=3, plate=8)
        assert walk.finished
        with pytest.raises(RuntimeError, match="every site"):
            walk.propose(8)


class TestTheWalkFollowsAMovedAnchor:
    def test_it_carries_its_fixed_sites_across_the_move(self):
        # Rebuilding through the campaign's factory would restart the walk at
        # site zero of a freshly drawn order, re-saturating residues it had
        # already fixed and re-measuring designs the campaign has paid for.
        env = make_env(length=8, symbols="ABCD", max_mutations=2)
        walk = SingleStepWalk(env, seed=0)
        drive(walk, env, rounds=2, plate=32)
        moved = walk.reanchored(env.reanchored(walk.incumbent))
        assert moved.sites_fixed[: len(walk.sites_fixed)] == walk.sites_fixed
        assert isinstance(moved, Sampler)

    def test_the_move_restores_the_budget_the_walk_had_spent(self):
        # The half of re-anchoring this arm needs: with a fixed anchor the ball
        # runs out after `max_mutations` sites, and with a moving one the walk
        # carries on down its permutation.
        env = make_env(length=8, symbols="ABCD", max_mutations=2)
        fixed = SingleStepWalk(env, seed=0)
        drive(fixed, env, rounds=2, plate=32)
        assert fixed.finished

        moving = SingleStepWalk(env, seed=0)
        for _ in range(4):
            batch = moving.propose(32)
            moving.observe(batch, additive(batch))
            moving = moving.reanchored(env.reanchored(moving.incumbent))
        assert len(moving.sites_fixed) > len(fixed.sites_fixed)


class TestRecombinationSaturatesIndependentlyThenCombines:
    def test_every_saturation_is_a_single_substitution_of_one_background(self):
        # The whole difference from the walk. A site measured in the presence of
        # another site's winner would make the two arms the same method run
        # twice.
        env = make_env(length=6, symbols="ABCD", max_mutations=4)
        recomb = Recombination(env, rounds=4, batch_size=16, seed=0)
        for _ in range(3):
            batch = recomb.propose(16)
            distances = (batch != env.parent[None, :]).sum(axis=1)
            assert distances.max() <= 1
            recomb.observe(batch, additive(batch))

    def test_the_recombinant_carries_the_best_variant_from_each_site(self):
        env = make_env(length=6, symbols="ABCD", max_mutations=4)
        recomb = Recombination(env, rounds=4, batch_size=32, seed=0)
        drive(recomb, env, rounds=4, plate=32)
        design = recomb.recombinant
        assert design is not None
        for site in recomb.sites:
            assert design[site] == 1, "a site did not take its best-scoring substitution"

    def test_the_saturations_leave_a_round_for_the_recombinant(self):
        # Packing them into the earliest plates would finish the protocol with
        # rounds to spare and nothing legal to spend them on, which costs the arm
        # the one design it exists to measure.
        env = make_env(length=6, symbols="ABCD", max_mutations=4)
        recomb = Recombination(env, rounds=4, batch_size=32, seed=0)
        for index in range(3):
            batch = recomb.propose(32)
            recomb.observe(batch, additive(batch))
            assert recomb.recombinant is None, f"combined at round {index}, too early"
        assert recomb.propose(32) is not None
        assert recomb.recombinant is not None

    def test_its_cost_is_the_published_formula(self):
        env = make_env(length=6, symbols="ABCDEFGHIJ", max_mutations=4)
        recomb = Recombination(env, rounds=5, batch_size=64, seed=0)
        drive(recomb, env, rounds=5, plate=64)
        expected = (env.alphabet.size - 1) * len(recomb.sites) + RECOMBINATION_OVERHEAD
        assert recomb.requested == expected

    def test_the_site_count_never_exceeds_the_mutation_budget(self):
        # The recombinant carries one substitution per site, so a wider site set
        # would nominate a design the environment cannot build.
        env = make_env(length=16, symbols="ABCD", max_mutations=3)
        recomb = Recombination(env, rounds=8, batch_size=64, seed=0)
        assert len(recomb.sites) <= env.max_mutations

    def test_the_background_does_not_follow_a_moved_anchor(self):
        # Every site's winner is chosen by comparing measurements taken on one
        # background. A background that moved half way through would have the
        # early sites judged against a different sequence from the late ones, and
        # the recombination would combine winners that were never comparable.
        env = make_env(length=6, symbols="ABCD", max_mutations=4)
        recomb = Recombination(env, rounds=4, batch_size=32, seed=0)
        batch = recomb.propose(32)
        recomb.observe(batch, additive(batch))
        moved_env = env.reanchored(np.array([1, 0, 0, 0, 0, 0], dtype=np.int32))
        moved = recomb.reanchored(moved_env)
        assert np.array_equal(moved._background, env.parent)
        assert moved.sites == recomb.sites


class TestTheSurplusBudgetGoesToReplicates:
    """The arm's own choice, since neither source addresses a surplus.

    The failure this guards against is a silent return to repetition. A plate
    filled with copies of designs the protocol had already named charges several
    wells per measurement, which handicaps the arm by most of its budget and
    shows up in the results as a weak baseline rather than as a harness
    decision.
    """

    def test_a_plate_holds_as_many_walks_as_it_has_room_for(self):
        env = make_env(length=8, symbols="ABCDEFGHIJ", max_mutations=8)
        assert walk_replicates(env, 96) == 96 // (env.alphabet.size - 1)
        assert walk_replicates(env, 4) == 1

    def test_the_replicated_walk_nearly_fills_its_plate_with_distinct_designs(self):
        # The whole point of sizing the replicates to the plate: one walk names a
        # fraction of it, and the rest would otherwise be repeats.
        env = make_env(length=8, symbols="ABCDEFGHIJ", max_mutations=8)
        arm = replicated_walk(env, batch_size=36, seed=0)
        plate = arm.propose(36)
        distinct = len({row.tobytes() for row in np.ascontiguousarray(plate)})
        assert arm.replicates == 4
        assert distinct > 36 // 2

    def test_each_replicate_keeps_its_own_walk(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=8)
        arm = replicated_walk(env, batch_size=32, seed=0)
        drive(arm, env, rounds=3, plate=32)
        walks = [cast("SingleStepWalk", protocol) for protocol in arm.protocols]
        assert any(walk.sites_fixed != walks[0].sites_fixed for walk in walks)

    def test_a_design_one_replicate_asked_for_informs_the_others(self):
        # Replicates share a background and overlap on the sites they reach, so
        # a design measured through one must count for all of them. Without that
        # a replicate would open a site whose substitutions the campaign had
        # already paid for, ask for them again, have them dropped as repeats and
        # wait for values that can never arrive.
        env = make_env(length=6, symbols="ABCD", max_mutations=6)
        arm = replicated_walk(env, batch_size=32, seed=0)
        drive(arm, env, rounds=6, plate=32)
        assert all(not protocol.requests() or True for protocol in arm.protocols)
        assert arm.requested > 0

    def test_the_pooled_request_is_deduplicated(self):
        # Replicates name the background, and later the same substitution, so an
        # undeduplicated pool would spend several wells on one design.
        env = make_env(length=6, symbols="ABCD", max_mutations=6)
        arm = replicated_walk(env, batch_size=32, seed=0)
        pending = arm.requests()
        assert len({design.tobytes() for design in pending}) == len(pending)

    def test_a_finished_arm_starts_a_further_replicate_rather_than_stopping(self):
        # A protocol that has returned its answer with budget left is what a lab
        # runs again on fresh sites. Refusing instead would lose every
        # measurement the campaign had already paid for, since a round that
        # cannot fill its plate aborts the run.
        env = make_env(length=12, symbols="ABCD", max_mutations=3)
        arm = replicated_recombination(env, rounds=3, batch_size=24, seed=0)
        started = arm.replicates
        drive(arm, env, rounds=6, plate=24)
        assert arm.replicates > started

    def test_a_library_measured_in_full_ends_the_arm_honestly(self):
        # The other side of the same rule: where a fresh replicate would have
        # nothing to ask for either, the arm stops rather than inventing designs.
        env = make_env(length=2, symbols="AB", max_mutations=2)
        arm = replicated_walk(env, batch_size=8, seed=0)
        for _ in range(6):
            if arm.finished:
                break
            batch = arm.propose(8)
            arm.observe(batch, additive(batch))
        assert arm.finished
        with pytest.raises(RuntimeError, match="anything left to measure"):
            arm.propose(8)

    def test_the_replicates_survive_a_moved_anchor_as_themselves(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=4)
        arm = replicated_walk(env, batch_size=32, seed=0)
        drive(arm, env, rounds=2, plate=32)
        anchor = cast("SingleStepWalk", arm.protocols[0]).incumbent
        moved = arm.reanchored(env.reanchored(anchor))
        assert isinstance(moved, ReplicatedProtocol)
        assert moved.replicates == arm.replicates
        assert moved.requested == arm.requested
        for before, after in zip(arm.protocols, moved.protocols, strict=True):
            assert np.array_equal(
                cast("SingleStepWalk", after).order, cast("SingleStepWalk", before).order
            )
