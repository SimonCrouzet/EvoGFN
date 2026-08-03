"""What action masking excludes, as a property rather than as a table of numbers.

``experiments/feasible_reachable_sweep.py`` measures the gap between the feasible
set and the set a masked policy can construct. These tests pin the *claim* that
experiment supports:

* at full transition density the two sets coincide, so the gap is caused by the
  constraint and not by the enumeration;
* as the constraint tightens the excluded share does not fall;
* on some instances the excluded part contains the best feasible design, which
  is what makes this a measurement about ceilings rather than about set sizes.

Deliberately not pinned: any particular percentage. The excluded share at a fixed
density ranges from 0% to over 80% depending only on which adjacencies the
instance happened to forbid, so a test asserting "68.8% at density 0.5" would be
asserting a seed, and would break the moment anything upstream of the RNG moved.
The averages here are over enough instances to make the ordering stable while
still running in a fraction of a second.
"""

import statistics

import pytest

from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape

#: Densities from unconstrained to severe. Coarse on purpose: adjacent densities
#: can invert on a finite sample, and the claim is about the trend.
DENSITIES = (1.0, 0.7, 0.5, 0.3, 0.15)

#: Instances per density. Enough that the mean is not one instance's accident.
SEEDS = 25

#: A small graph: 8 positions over 4 tokens within 2 substitutions is a Hamming
#: ball of 277, so every reachable set can be found by exhaustive forward search.
GEOMETRY = {
    "sequence_length": 8,
    "vocab_size": 4,
    "n_motifs": 2,
    "motif_length": 2,
    "quantization": 2,
    "max_spacing": 3,
}
BUDGET = 2


def measure(density, seed):
    """Sizes and ceilings of the feasible and reachable sets for one instance."""
    landscape = EhrlichLandscape(transition_density=density, seed=seed, **GEOMETRY)
    env = MutationEnvironment(
        landscape.feasible_sequence(seed),
        landscape.alphabet,
        max_mutations=BUDGET,
        transitions=landscape.transition_matrix,
    )
    ball = env.enumerate_terminal_states()
    feasible = ball[landscape.is_feasible(ball)]
    reachable = env.reachable_terminal_states()
    constructible = ball[env.is_constructible(ball)]
    return {
        "feasible": {row.tobytes() for row in feasible},
        "reachable": {row.tobytes() for row in reachable},
        "constructible": {row.tobytes() for row in constructible},
        "best_feasible": float(landscape.evaluate(feasible)[:, 0].max()),
        "best_reachable": float(landscape.evaluate(reachable)[:, 0].max()),
    }


@pytest.fixture(scope="module")
def sweep():
    """Every instance, measured once and shared by every test in the module."""
    return {density: [measure(density, seed) for seed in range(SEEDS)] for density in DENSITIES}


def excluded(result):
    return 1.0 - len(result["reachable"]) / len(result["feasible"])


class TestTheSetsCoincideWhenNothingIsForbidden:
    def test_full_density_leaves_no_design_unreachable(self, sweep):
        # The control. Without it, a gap at low density could just as well be an
        # enumeration bug: this says the two sets are computed compatibly, and
        # that everything below is caused by the constraint.
        for result in sweep[1.0]:
            assert result["reachable"] == result["feasible"]

    def test_full_density_loses_nothing_at_the_top(self, sweep):
        for result in sweep[1.0]:
            assert result["best_reachable"] == result["best_feasible"]


class TestExclusionGrowsAsTheConstraintTightens:
    def test_the_reachable_set_is_always_a_subset_of_the_feasible_one(self, sweep):
        # Structural, and the reason "excluded" is a share rather than a
        # difference that could point either way.
        for results in sweep.values():
            for result in results:
                assert result["reachable"] <= result["feasible"]

    def test_mean_exclusion_never_falls_as_density_falls(self, sweep):
        means = [statistics.fmean([excluded(r) for r in sweep[d]]) for d in DENSITIES]
        assert means == sorted(means), dict(zip(DENSITIES, means, strict=True))

    def test_the_tightest_and_loosest_are_separated_by_more_than_noise(self, sweep):
        # Guards the ordering test from passing on five near-identical means,
        # which would make "non-decreasing" true and uninformative.
        loosest = statistics.fmean([excluded(r) for r in sweep[DENSITIES[0]]])
        tightest = statistics.fmean([excluded(r) for r in sweep[DENSITIES[-1]]])
        assert tightest > loosest + 0.05


class TestTheLocalConstructibilityTestIsExact:
    """`is_constructible` must agree with the forward search, not approximate it.

    The search is the definition -- it walks the environment's own masks -- but it
    is exponential and cannot be run on a real task, so anything that needs to
    know whether a design is in the search space uses the local test instead. If
    the two ever disagree, the local test is admitting designs no trajectory can
    build or excluding ones it can, and every caller that trusted it is deciding
    membership of the wrong set.
    """

    def test_it_reproduces_the_forward_search_on_every_instance(self, sweep):
        # Set equality rather than counts: two sets of the same size that differ
        # in their members would pass a size check and still be wrong.
        for density, results in sweep.items():
            for index, result in enumerate(results):
                assert result["constructible"] == result["reachable"], (density, index)

    def test_it_is_strictly_stronger_than_endpoint_feasibility(self, sweep):
        # Without this the agreement above could be vacuous: a test that simply
        # returned `is_reachable` would satisfy it on any instance where the two
        # sets happen to coincide.
        narrowed = [
            result
            for results in sweep.values()
            for result in results
            if result["constructible"] < result["feasible"]
        ]
        assert narrowed


class TestTheExcludedSetCanContainTheOptimum:
    def test_some_instance_cannot_construct_its_best_feasible_design(self, sweep):
        # The claim that matters. Excluding designs is a fact about set sizes;
        # excluding the *best* one means a masked sampler is optimising against
        # a lower ceiling than every report of it assumes, and no amount of
        # training closes the gap because the design is outside its support.
        stranded = [
            (density, index)
            for density, results in sweep.items()
            for index, result in enumerate(results)
            if result["best_reachable"] < result["best_feasible"]
        ]
        assert stranded

    def test_no_instance_reaches_past_what_is_feasible(self, sweep):
        # The other direction, which would mean the walk had left the feasible
        # set -- a far worse failure than excluding part of it.
        for results in sweep.values():
            for result in results:
                assert result["best_reachable"] <= result["best_feasible"]
