"""Tests for AdaLead, the sequence-design lineage's default model-guided arm.

The failure guarded against is the arm reading as a second genetic algorithm. It
differs from ours in exactly one structural place -- its rollout screens every
candidate against its own surrogate and keeps only the ones that improve on the
sequence they came from -- and an implementation that dropped that screen would
leave the suite with two blind population searches and a claim, made in the
docstring, that one of them was model-guided.

The second failure is the published rates going missing. They are the arm's whole
claim to being AdaLead rather than something AdaLead-shaped.
"""

import numpy as np
import pytest

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines.adalead import (
    DEFAULT_MUTATION_MULTIPLIER,
    DEFAULT_RECOMBINE_PROB,
    DEFAULT_THRESHOLD,
    MODEL_QUERIES_PER_DESIGN,
    AdaLead,
)
from evogfn.algorithms.baselines.directed_evolution import within_budget
from evogfn.core import Alphabet
from evogfn.env.mutation import MutationEnvironment
from evogfn.surrogate.base import Surrogate


def make_env(length=8, symbols="ABCD", max_mutations=4):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
    )


def counting(sequences):
    """A learnable target: value rises with the count of token 1."""
    return (np.asarray(sequences) == 1).sum(axis=1, keepdims=True).astype(np.float64)


class Oracle(Surrogate):
    """A model that predicts the toy landscape exactly, with a fixed spread.

    Stands in for the deep ensemble so the tests measure the search rather than
    a network's convergence, and so they do not spend a torch fit per assertion.
    """

    def __init__(self, transform=lambda values: values):
        self.transform = transform
        self.fits = 0

    def fit(self, _sequences, _values):
        self.fits += 1

    def predict(self, sequences):
        mean = self.transform(counting(sequences)[:, 0])
        return mean, np.ones_like(mean)


class Pessimist(Surrogate):
    """A model that rates every mutant below whatever it is asked about first.

    Makes the acceptance rule observable: under it, no child ever improves on its
    root, so a rollout that still returned candidates would not be screening.
    """

    def __init__(self):
        self.seen = 0

    def fit(self, _sequences, _values):
        pass

    def predict(self, sequences):
        # Falls with every call, so any child is scored below its own root.
        self.seen += 1
        mean = np.full(np.asarray(sequences).shape[0], -float(self.seen))
        return mean, np.ones_like(mean)


def seeded(env, model=None, **kwargs):
    return AdaLead(env, model=model or Oracle(), seed=0, **kwargs)


def drive(sampler, env, rounds, plate):
    """Run a sampler against the toy landscape the way a campaign would."""
    for _ in range(rounds):
        batch = sampler.propose(plate)
        assert all(within_budget(env, row) for row in batch)
        sampler.observe(batch, counting(batch))


class TestTheRatesAreTheOnesItsPaperStates:
    def test_the_published_defaults_are_the_defaults(self):
        # The elite threshold, the per-position crossover rate and a mutation
        # rate of one over the sequence length, all from the paper's appendix.
        # Comparing against anything else compares against a configuration
        # convenient to us.
        assert (DEFAULT_THRESHOLD, DEFAULT_RECOMBINE_PROB, DEFAULT_MUTATION_MULTIPLIER) == (
            0.05,
            0.2,
            1,
        )

    @pytest.mark.parametrize(("field", "value"), [("threshold", 1.5), ("recombine_prob", -0.1)])
    def test_rates_outside_the_unit_interval_are_refused(self, field, value):
        with pytest.raises(ValueError, match="must lie in"):
            AdaLead(make_env(), **{field: value})


class TestItIsAModelGuidedSearch:
    def test_it_is_a_sampler_with_the_right_proposal_shape(self):
        env = make_env()
        sampler = seeded(env)
        assert isinstance(sampler, Sampler)
        assert sampler.propose(16).shape == (16, env.sequence_length)

    def test_it_draws_at_random_until_it_has_something_to_fit(self):
        # A rollout screening against an unfitted model would be screening
        # against an initialisation, which looks like a search and is noise.
        env = make_env()
        sampler = seeded(env)
        assert not sampler.is_fitted
        sampler.propose(16)
        assert sampler.proxy_calls == 0

    def test_the_rollout_spends_model_queries_and_reports_them(self):
        # The free half of the compute. Reported under the same name every
        # surrogate-bearing arm here uses, so a table can compare what a rollout
        # costs against what training a policy costs.
        env = make_env()
        sampler = seeded(env)
        drive(sampler, env, rounds=2, plate=16)
        assert 0 < sampler.proxy_calls <= 2 * MODEL_QUERIES_PER_DESIGN * 16

    def test_a_child_the_model_rates_below_its_root_is_discarded(self):
        # The acceptance rule, and the whole structural difference from a blind
        # genetic algorithm. With a model that rates every mutant below whatever
        # it scored before, nothing survives the climb and the arm has no
        # candidates of its own to offer.
        env = make_env()
        sampler = seeded(env, model=Pessimist())
        batch = sampler.propose(16)
        sampler.observe(batch, counting(batch))
        assert sampler._rollout(16) == []

    def test_it_still_fills_the_plate_when_nothing_survives_the_climb(self):
        # A plate has to be filled with something, and the honest something is a
        # uniform draw rather than candidates the arm's own screen rejected.
        env = make_env()
        sampler = seeded(env, model=Pessimist())
        batch = sampler.propose(16)
        sampler.observe(batch, counting(batch))
        assert sampler.propose(16).shape == (16, env.sequence_length)


class TestTheEliteSetIsChosenOnMeasurements:
    def test_only_sequences_within_the_threshold_seed_a_rollout(self):
        # Seeds come from the *assay*, not from the model: it is a greedy search
        # over what is known to be good, and widening it to everything measured
        # would make the arm a random-restart search.
        env = make_env()
        sampler = seeded(env)
        good = np.zeros((1, env.sequence_length), dtype=np.int32)
        good[0, :3] = 1
        poor = np.zeros((1, env.sequence_length), dtype=np.int32)
        sampler.observe(np.concatenate([good, poor]), np.array([[3.0], [0.0]]))
        elites = sampler._elite(8)
        assert (elites == good[0]).all()

    def test_a_tie_at_the_top_seeds_every_tied_sequence(self):
        env = make_env()
        sampler = seeded(env)
        first = np.zeros((1, env.sequence_length), dtype=np.int32)
        first[0, 0] = 1
        second = np.zeros((1, env.sequence_length), dtype=np.int32)
        second[0, 1] = 1
        sampler.observe(np.concatenate([first, second]), np.array([[1.0], [1.0]]))
        elites = sampler._elite(8)
        assert len({row.tobytes() for row in np.ascontiguousarray(elites)}) == 2


class TestRecombination:
    def test_a_zero_rate_leaves_the_pool_alone(self):
        # The rate is a rate: at zero there is no crossover, which is what makes
        # the published 0.2 a setting rather than a switch.
        env = make_env()
        sampler = seeded(env, recombine_prob=0.0)
        pool = np.tile(np.arange(env.sequence_length, dtype=np.int32) % 4, (8, 1))
        assert np.array_equal(sampler._recombine(pool), pool)

    def test_crossover_produces_designs_neither_parent_carried(self):
        env = make_env(length=16, max_mutations=16)
        sampler = seeded(env, recombine_prob=0.5)
        pool = np.zeros((8, env.sequence_length), dtype=np.int32)
        pool[1::2] = 1
        offspring = sampler._recombine(pool)
        mixed = [row for row in offspring if 0 < int((row == 1).sum()) < env.sequence_length]
        assert mixed, "no offspring mixed its two parents"

    def test_offspring_beyond_the_mutation_budget_are_refused(self):
        # Every proposal has to stay inside the graph the rest of the suite
        # searches; a design outside it is one no other arm could have produced.
        env = make_env(length=16, max_mutations=2)
        sampler = seeded(env, recombine_prob=0.5)
        pool = np.zeros((8, env.sequence_length), dtype=np.int32)
        pool[1::2, :2] = 1
        offspring = sampler._recombine(pool)
        assert all(within_budget(env, row) for row in offspring)


class TestItCarriesItsStateAcrossAMovedAnchor:
    def test_the_measurements_and_the_model_survive(self):
        # A rebuild would empty the elite set and refit from nothing every round,
        # which on this arm is the difference between a screened search and a
        # random one.
        env = make_env()
        sampler = seeded(env)
        drive(sampler, env, rounds=2, plate=16)
        anchor = np.zeros(env.sequence_length, dtype=np.int32)
        anchor[0] = 1
        moved = sampler.reanchored(env.reanchored(anchor))
        assert moved.training_examples == sampler.training_examples
        assert moved.proxy_calls == sampler.proxy_calls
        assert moved._model is sampler._model


class TestReproducibility:
    def test_the_same_seed_gives_the_same_proposals(self):
        env = make_env()
        left, right = seeded(env), seeded(env)
        batch = left.propose(16)
        assert np.array_equal(batch, right.propose(16))
        left.observe(batch, counting(batch))
        right.observe(batch, counting(batch))
        assert np.array_equal(left.propose(16), right.propose(16))
