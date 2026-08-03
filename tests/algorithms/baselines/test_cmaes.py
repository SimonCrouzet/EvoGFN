"""Tests for separable CMA-ES over the sequence relaxation.

Three things can silently break here and none of them shows up as an exception.
The argmax decoding can leave the mutation budget, in which case the sampler is
being scored on designs no other method is allowed to propose. The update can be
attributed to the wrong sample, because the harness scores a selected *subset* of
a round's proposals rather than all of it in order, which turns the covariance
update into noise. And the step size can run away to an infinity on a round where
everything scored -inf. Each has its own test.
"""

from itertools import product

import numpy as np
import pytest

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines import CMAES
from evogfn.algorithms.baselines.cmaes import _project_onto_constructible
from evogfn.core import Alphabet
from evogfn.env.mutation import MutationEnvironment, TerminalFeasibilityEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape


def make_env(length=6, symbols="ABCD", max_mutations=3, transitions=None):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
        transitions=transitions,
    )


def constrained_transitions(vocab, forbidden):
    matrix = np.ones((vocab, vocab), dtype=np.float64)
    for a, b in forbidden:
        matrix[a, b] = 0.0
    return matrix


def toy_landscape(sequences):
    """Reward sequences for containing token 1, so improvement is detectable."""
    return (np.asarray(sequences) == 1).sum(axis=1, keepdims=True).astype(np.float64)


def unrelated_designs(env, proposals, count):
    """Reachable sequences that are definitely not among ``proposals``."""
    rng = np.random.default_rng(99)
    known = {row.tobytes() for row in np.ascontiguousarray(proposals)}
    length, budget = env.sequence_length, env.max_mutations
    picked: list[np.ndarray] = []
    while len(picked) < count:
        candidate = np.zeros(length, dtype=np.int32)
        positions = rng.choice(length, size=budget, replace=False)
        candidate[positions] = rng.integers(1, env.alphabet.size, size=budget)
        if candidate.tobytes() not in known:
            picked.append(candidate)
    return np.stack(picked)


class TestTheSharedInterface:
    def test_it_is_a_sampler(self):
        assert isinstance(CMAES(make_env()), Sampler)

    def test_proposals_have_the_right_shape(self):
        env = make_env()
        assert CMAES(env).propose(16).shape == (16, env.sequence_length)

    def test_proposals_stay_inside_the_environment_graph(self):
        env = make_env(max_mutations=2)
        sampler = CMAES(env, seed=0)
        for _ in range(6):
            proposals = sampler.propose(32)
            assert env.is_reachable(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))

    def test_proposals_are_counted(self):
        sampler = CMAES(make_env())
        sampler.propose(10)
        sampler.propose(7)
        assert sampler.proposals_made == 17

    def test_the_same_seed_gives_the_same_proposals(self):
        env = make_env()
        assert np.array_equal(CMAES(env, seed=4).propose(16), CMAES(env, seed=4).propose(16))

    def test_a_whole_campaign_is_reproducible(self):
        env = make_env(length=8, max_mutations=5)
        runs = []
        for _ in range(2):
            sampler = CMAES(env, seed=11)
            batches = []
            for _ in range(4):
                proposals = sampler.propose(24)
                sampler.observe(proposals, toy_landscape(proposals))
                batches.append(proposals)
            runs.append(np.concatenate(batches))
        assert np.array_equal(runs[0], runs[1])

    def test_the_label_says_whether_feasibility_is_enforced(self):
        env = make_env()
        assert CMAES(env).name == "CMAES"
        assert CMAES(env, feasible_only=True).name == "CMAES (feasible)"


class TestTheRelaxation:
    def test_the_covariance_is_diagonal(self):
        # A full covariance over length * vocabulary would be 26 million entries
        # at L = 256, and its eigendecomposition is what makes it intractable.
        env = make_env(length=6, symbols="ABCD")
        sampler = CMAES(env)
        assert sampler._diagonal.shape == (6 * 4,)

    def test_the_mean_is_one_logit_per_position_and_token(self):
        env = make_env(length=6, symbols="ABCD")
        assert CMAES(env).mean_logits.shape == (6, 4)

    def test_decoding_projects_onto_the_mutation_budget(self):
        # The argmax of an unconstrained Gaussian differs from the parent at
        # nearly every position; without the projection the sampler would emit
        # designs outside the graph on its very first round.
        env = make_env(length=12, symbols="ABCD", max_mutations=2)
        proposals = CMAES(env, seed=0).propose(64)
        assert (proposals != env.parent[None, :]).sum(axis=1).max() <= 2

    def test_the_distribution_moves_when_the_ranking_is_informative(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        before = sampler.mean_logits.copy()
        proposals = sampler.propose(64)
        sampler.observe(proposals, toy_landscape(proposals))
        assert not np.allclose(before, sampler.mean_logits)

    def test_it_learns_to_prefer_the_rewarded_token(self):
        # The relaxation is only doing its job if the logit for the token the
        # landscape rewards ends up above the others.
        env = make_env(length=8, symbols="ABCD", max_mutations=8)
        sampler = CMAES(env, seed=0)
        for _ in range(20):
            proposals = sampler.propose(64)
            sampler.observe(proposals, toy_landscape(proposals))
        logits = sampler.mean_logits
        assert logits[:, 1].mean() > logits[:, [0, 2, 3]].mean()

    def test_it_improves_over_rounds(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        first = last = 0.0
        for index in range(20):
            proposals = sampler.propose(64)
            values = toy_landscape(proposals)
            sampler.observe(proposals, values)
            if index == 0:
                first = float(values.max())
            last = max(last, float(values.max()))
        assert last > first


class TestAttributingScoresToDraws:
    """The harness scores a selected subset, in its own order.

    CMA-ES updates from the Gaussian draw behind each ranked sample, so lining
    scores up by row index -- which is what a naive implementation does -- would
    attribute every score to some other candidate's draw. The distribution would
    still move, and it would still look like it was working.
    """

    def test_a_subset_in_a_different_order_still_updates(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        proposals = sampler.propose(64)
        chosen = proposals[::-1][:16]
        before = sampler.mean_logits.copy()
        sampler.observe(chosen, toy_landscape(chosen))
        assert not np.allclose(before, sampler.mean_logits)

    def test_sequences_it_never_proposed_are_ignored(self):
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        proposals = sampler.propose(64)
        outsiders = unrelated_designs(env, proposals, count=8)
        before = sampler.mean_logits.copy()
        sampler.observe(outsiders, np.full((8, 1), 5.0))
        assert np.allclose(before, sampler.mean_logits)

    def test_a_single_usable_measurement_does_not_move_the_distribution(self):
        # One sample is a ranking of one: the rank-mu term is empty and the only
        # effect left is an inflated step size.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        proposals = sampler.propose(64)
        before = sampler.mean_logits.copy()
        sampler.observe(proposals[:1], np.array([[3.0]]))
        assert np.allclose(before, sampler.mean_logits)


class TestNumericalRobustness:
    def test_a_round_of_failed_assays_leaves_the_search_usable(self):
        # -inf is what an infeasible design scores, and a whole round of them is
        # routine on a sparse feasible set. A NaN in the covariance here would
        # silently poison every later round.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        for _ in range(5):
            proposals = sampler.propose(32)
            sampler.observe(proposals, np.full((32, 1), -np.inf))
        assert np.isfinite(sampler.sigma)
        assert sampler.sigma > 0.0
        assert np.isfinite(sampler.mean_logits).all()

    def test_the_step_size_stays_finite_under_a_constant_ranking(self):
        # Every candidate scoring the same is the degenerate case for a rank
        # based method: the weights are applied to an arbitrary order.
        env = make_env(length=8, symbols="ABCD", max_mutations=6)
        sampler = CMAES(env, seed=0)
        for _ in range(30):
            proposals = sampler.propose(32)
            sampler.observe(proposals, np.zeros((32, 1)))
        assert np.isfinite(sampler.sigma)
        assert np.isfinite(sampler._diagonal).all()
        assert (sampler._diagonal > 0.0).all()


def sparse_transitions(vocab, seed=0):
    """A transition matrix sparse enough that rejection sampling is hopeless.

    Built as a Hamiltonian cycle plus a few extra edges, which is how
    EhrlichLandscape builds one: the cycle keeps every token reachable, so a
    feasible sequence exists, while the density stays low enough that a
    uniformly drawn sequence is feasible with vanishing probability.
    """
    rng = np.random.default_rng(seed)
    matrix = np.zeros((vocab, vocab), dtype=np.float64)
    order = rng.permutation(vocab)
    matrix[order, np.roll(order, -1)] = 1.0
    matrix[rng.random((vocab, vocab)) < 0.1] = 1.0
    return matrix


def open_transitions(vocab, forbidden_share=0.4, seed=1):
    """A matrix that binds without emptying the construction graph.

    The counterpart of `sparse_transitions`, and needed because the two ask
    different questions. Sparse enough and the *constructible* set collapses to
    the anchor -- correctly, since almost nothing can be built -- which makes it
    the wrong instance on which to ask whether the projection searches. Here the
    Hamiltonian cycle is laid over a mostly-permissive matrix, so a design many
    substitutions from the anchor has a construction order and the projection has
    somewhere to go.
    """
    rng = np.random.default_rng(seed)
    matrix = np.ones((vocab, vocab), dtype=np.float64)
    matrix[rng.random((vocab, vocab)) < forbidden_share] = 0.0
    order = rng.permutation(vocab)
    matrix[order, np.roll(order, -1)] = 1.0
    return matrix


def feasible_start(transitions, length, seed=0):
    """A sequence the transition matrix admits, walked out of the chain itself."""
    rng = np.random.default_rng(seed)
    permitted = transitions > 0
    sequence = np.zeros(length, dtype=np.int32)
    sequence[0] = rng.integers(0, transitions.shape[0])
    for position in range(1, length):
        sequence[position] = rng.choice(np.flatnonzero(permitted[sequence[position - 1]]))
    return sequence


class TestFeasibility:
    """The relaxation cannot express an adjacency rule; the decoder must.

    A separable Gaussian over per-position logits is a product distribution and
    argmax is a product map, so nothing the search does can put its emissions
    inside a set that couples adjacent positions. Before the projection existed
    this arm scored -inf on every design of every seed of the benchmark's
    feasibility task -- not rarely, never once. These tests pin down that the
    projection fixes it, that it is exact rather than a nudge, and that it costs
    no proposals.
    """

    def sparse_env(self, length=24, vocab=8, budget=None):
        transitions = sparse_transitions(vocab)
        parent = feasible_start(transitions, length)
        return MutationEnvironment(
            parent,
            Alphabet.from_string("ABCDEFGH"[:vocab]),
            max_mutations=length - 2 if budget is None else budget,
            transitions=transitions,
        )

    def test_the_unrepaired_relaxation_emits_nothing_buildable(self):
        # The diagnosis, kept as a test so the fix cannot be mistaken for having
        # always worked. This is the behaviour that produced -inf on 100 seeds.
        env = self.sparse_env()
        proposals = CMAES(env, repair=False, seed=0).propose(64)
        assert not env.is_reachable(proposals).any()

    def test_repairing_makes_every_design_buildable(self):
        env = self.sparse_env()
        sampler = CMAES(env, seed=0)
        for _ in range(3):
            proposals = sampler.propose(64)
            assert env.is_constructible(proposals).all()
            sampler.observe(proposals, toy_landscape(proposals))

    def open_env(self, length=24, vocab=8, budget=8):
        transitions = open_transitions(vocab)
        parent = feasible_start(transitions, length)
        return MutationEnvironment(
            parent,
            Alphabet.from_string("ABCDEFGH"[:vocab]),
            max_mutations=budget,
            transitions=transitions,
        )

    def test_the_repair_does_not_collapse_onto_the_anchor(self):
        # Emitting the parent every time would also be "buildable" and would be
        # a useless search. The projection maximises the logits over the
        # constructible set, so where that set is wide it should stay well out in
        # the ball. Asked on `open_env` rather than `sparse_env`: on an instance
        # where the environment can build almost nothing, returning the anchor is
        # the right answer and the assertion would be pinning an escape from the
        # construction graph rather than a search of it.
        env = self.open_env()
        proposals = CMAES(env, seed=0).propose(64)
        distances = (proposals != env.parent[None, :]).sum(axis=1)
        assert env.is_constructible(proposals).all()
        assert distances.mean() > env.sequence_length / 4
        # Not one design repeated: a collapsed search would also travel far.
        assert len({row.tobytes() for row in np.ascontiguousarray(proposals)}) > 1

    def test_the_projection_costs_no_extra_proposals(self):
        # The cost of repair is wall clock, not proposals and not oracle calls.
        # Rejection would have cost proposals -- unboundedly many, at this
        # density -- so the distinction is the whole trade being reported.
        env = self.sparse_env()
        repairing = CMAES(env, seed=0)
        repairing.propose(64)
        assert repairing.proposals_made == 64

    def test_the_repair_rate_reports_how_much_the_relaxation_contributed(self):
        env = self.sparse_env()
        sampler = CMAES(env, seed=0)
        sampler.propose(64)
        # On a sparse instance the raw argmax is never buildable, so the
        # projection is doing all of the work and the number must say so.
        assert sampler.repaired_fraction == 1.0

    def test_an_unconstrained_landscape_needs_no_repair_at_all(self):
        sampler = CMAES(make_env(length=8, max_mutations=8), seed=0)
        sampler.propose(32)
        assert sampler.repaired_fraction == 0.0

    def test_repairing_is_exact_rather_than_approximate(self):
        # The projection returns the *highest-scoring* constructible sequence,
        # not merely a constructible one. Checked against brute force, because a
        # repair that quietly returned something worse would look identical from
        # the outside and would understate the baseline.
        transitions = constrained_transitions(3, [(0, 1), (1, 2), (2, 0)])
        env = MutationEnvironment(
            np.zeros(4, dtype=np.int32),
            Alphabet.from_string("ABC"),
            max_mutations=3,
            transitions=transitions,
        )
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((6, 4, 3))
        decoded = _project_onto_constructible(logits, env.parent, transitions > 0, 3)
        for row in range(6):
            achieved = logits[row, np.arange(4), decoded[row]].sum()
            best = max(
                logits[row, np.arange(4), np.array(candidate)].sum()
                for candidate in product(range(3), repeat=4)
                if env.is_constructible(np.array(candidate)[None, :])[0]
            )
            assert achieved == pytest.approx(best)

    def test_the_projection_lands_inside_the_enumerated_construction_graph(self):
        # The invariant, against the environment's own forward search rather
        # than against any predicate the projection might share a mistake with.
        # `reachable_terminal_states` walks `forward_mask` and `step`, so it is
        # the definition of what a trajectory can end on; a design outside it is
        # one no masked arm could have proposed, and an arm allowed to propose it
        # is being compared on a larger search space than the rest.
        for length, budget, forbidden in [
            (7, 3, [(0, 1), (1, 2), (2, 0), (1, 1)]),
            (6, 4, [(0, 2), (2, 1), (1, 0), (2, 2)]),
        ]:
            transitions = constrained_transitions(3, forbidden)
            env = MutationEnvironment(
                np.zeros(length, dtype=np.int32),
                Alphabet.from_string("ABC"),
                max_mutations=budget,
                transitions=transitions,
            )
            graph = {row.tobytes() for row in env.reachable_terminal_states().astype(np.int32)}
            rng = np.random.default_rng(1)
            logits = rng.standard_normal((48, length, 3))
            decoded = _project_onto_constructible(logits, env.parent, transitions > 0, budget)
            outside = [r for r in decoded.astype(np.int32) if r.tobytes() not in graph]
            assert not outside, (length, budget, len(graph))
            # Not vacuous: the endpoint condition alone admits strictly more, so
            # a projection targeting it would have had somewhere to escape to.
            ball = env.enumerate_terminal_states()
            assert int(env.is_reachable(ball).sum()) > len(graph)

    def test_repairing_does_not_narrow_a_terminal_feasibility_environment(self):
        # The mirror of the invariant above. Where the environment defers the
        # rule to the terminal, every feasible design in the budget is buildable
        # by any ordering, so imposing an ordering condition would confine this
        # arm to a strict subset of what that environment lets everything else
        # propose -- an error in the opposite direction and just as invisible.
        transitions = constrained_transitions(3, [(0, 1), (1, 2), (2, 0)])
        deferred = TerminalFeasibilityEnvironment(
            np.zeros(6, dtype=np.int32),
            Alphabet.from_string("ABC"),
            max_mutations=4,
            transitions=transitions,
        )
        rng = np.random.default_rng(2)
        logits = rng.standard_normal((32, 6, 3))
        decoded = _project_onto_constructible(
            logits, deferred.parent, transitions > 0, 4, ordered=False
        )
        assert deferred.is_constructible(decoded).all()
        for row in range(32):
            achieved = logits[row, np.arange(6), decoded[row]].sum()
            best = max(
                logits[row, np.arange(6), np.array(candidate)].sum()
                for candidate in product(range(3), repeat=6)
                if deferred.is_constructible(np.array(candidate)[None, :])[0]
            )
            assert achieved == pytest.approx(best)

    def test_regret_against_an_enumerated_optimum_cannot_go_negative(self):
        # What the invariant is *for*. The attainable optimum on a constrained
        # task is the maximum over the enumerated construction graph, so an arm
        # confined to that graph can match it and cannot beat it. A negative
        # regret is not a strong result, it is proof the arm left the space --
        # and it is the only symptom the harness would ever show.
        landscape = EhrlichLandscape(
            sequence_length=10,
            vocab_size=4,
            n_motifs=2,
            motif_length=2,
            quantization=2,
            max_spacing=3,
            transition_density=0.4,
            seed=3,
        )
        env = MutationEnvironment(
            landscape.feasible_sequence(3),
            landscape.alphabet,
            max_mutations=3,
            transitions=landscape.transition_matrix,
        )
        attainable = float(landscape.evaluate(env.reachable_terminal_states())[:, 0].max())

        sampler = CMAES(env, seed=0)
        best = -np.inf
        for _ in range(4):
            proposals = sampler.propose(64)
            values = landscape.evaluate(proposals)
            best = max(best, float(values[:, 0].max()))
            sampler.observe(proposals, values)
        assert best <= attainable + 1e-12
        # The task has to be able to separate arms, or the bound holds trivially
        # for a sampler that emits the anchor forever.
        assert attainable > float(landscape.evaluate(env.parent[None, :])[0, 0])

    def test_a_constraint_admitting_only_the_anchor_returns_the_anchor(self):
        # Every adjacency but (0, 0) forbidden, and the parent is all zeros, so
        # the anchor is the one design that can be built. Returning it is the
        # correct answer; the old rejection loop raised instead, which reported
        # a solvable instance as an impossible one.
        forbidden = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]
        env = make_env(
            length=8,
            symbols="ABCD",
            max_mutations=3,
            transitions=constrained_transitions(4, forbidden),
        )
        proposals = CMAES(env, seed=0).propose(16)
        assert np.array_equal(proposals, np.tile(env.parent, (16, 1)))

    def test_rejection_without_repair_still_raises(self):
        # `feasible_only` is retained as the check on `repair`, and it must keep
        # failing loudly when there is nothing to repair with.
        forbidden = [(a, b) for a in range(4) for b in range(4) if (a, b) != (0, 0)]
        env = make_env(
            length=8,
            symbols="ABCD",
            max_mutations=3,
            transitions=constrained_transitions(4, forbidden),
        )
        with pytest.raises(RuntimeError, match="feasible"):
            CMAES(env, repair=False, feasible_only=True, max_attempts=3, seed=0).propose(32)

    def test_the_label_marks_an_unrepaired_run(self):
        env = self.sparse_env()
        assert CMAES(env, repair=False).name == "CMAES (unrepaired)"


class TestValidation:
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_a_non_positive_step_size_is_refused(self, value):
        with pytest.raises(ValueError, match="initial_sigma must be positive"):
            CMAES(make_env(), initial_sigma=value)
