"""Tests for the attainable-optimum audit.

Three things are under test, and only the first is about this module's code.

**The suite's budgets.** ``test_a_task_out_of_reach_of_its_own_optimum_declares_it``
is the check whose absence let every Ehrlich task in ``MAIN`` report regret
against a target outside its own search space. It is the reason this file
exists, and it is deliberately the cheapest test here so nothing tempts anyone
to skip it. Its shape has changed with the suite's -- the radius is per round
now and the anchor moves, so the question is no longer whether the optimum is
inside one Hamming ball but whether the task states what it can reach. What is
being prevented is unchanged.

**The bracket.** A bound that is wrong is worse than no bound, because it reads
like a fact. Where the reachable set is small enough to enumerate, the exact
answer is known, and every bound is checked against it rather than against
another bound.

**The distinction.** ``exact`` must stay ``None`` unless the value was measured
or the interval closed, since the entire point of the return type is that a
caller cannot mistake one for the other.
"""

from typing import Any

import numpy as np
import pytest

from evogfn.benchmark.attainable import (
    AttainableOptimum,
    _certified_upper_bound,
    _searched_lower_bound,
    attainable_optimum,
    per_round_budget,
    planted_distance,
    planted_optimum_reachable,
    reanchored_attainable,
)
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.suite import CAPPED, MAIN, budget_gradient, objective_task, rounds_curve
from evogfn.benchmark.tasks import Task
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape

TOY: dict[str, Any] = {
    "sequence_length": 12,
    "vocab_size": 4,
    "n_motifs": 2,
    "motif_length": 2,
    "max_spacing": 2,
    "transition_density": 0.5,
}

#: Large enough that the Hamming ball is far past enumeration, small enough that
#: a beam search over it is a unit test rather than a benchmark run.
UNENUMERABLE: dict[str, Any] = {
    "sequence_length": 32,
    "vocab_size": 20,
    "n_motifs": 2,
    "motif_length": 4,
    "transition_density": 0.6,
}


def ehrlich_task(name="toy", *, budget=2, params=None, seed=0):
    settings = {**(params or TOY), "seed": seed}
    return Task(
        name=name,
        purpose="a toy, for testing bounds against enumeration",
        build=lambda: EhrlichLandscape(**settings),
        protocol=Protocol(rounds=1, batch_size=8, max_mutations=budget),
        max_mutations=budget,
    )


def environment(task):
    landscape = task.landscape()
    return landscape, MutationEnvironment(
        task.parent(landscape),
        landscape.alphabet,
        max_mutations=task.max_mutations,
        transitions=landscape.transition_matrix,
    )


# --------------------------------------------------------------------------
# The regression test: a task measured against a target it cannot reach.
# --------------------------------------------------------------------------
#
# The question this asks has moved, and the reason is worth stating. It used to
# be "is the planted optimum inside `max_mutations`", and the honest answer under
# the suite's current shape is *no, on every Ehrlich task, by design*: the radius
# is one round of mutagenesis and the planted optimum is 32 to 248 substitutions
# away. What clears that distance is the campaign re-anchoring, so the radius
# alone can no longer settle anything.
#
# What replaces it is the same failure caught one level up. A task may report
# regret against a target outside its reach only if it *says so* -- by declaring
# what it can reach, and by carrying the phrase `experiments/audit_optima.py`
# greps for. A task that carries neither is the original defect exactly.


@pytest.mark.parametrize(
    "task",
    [*MAIN, *budget_gradient(), *rounds_curve(), objective_task()],
    ids=lambda t: t.name,
)
def test_a_task_out_of_reach_of_its_own_optimum_declares_it(task):
    distance = planted_distance(task)
    if distance is None:
        pytest.skip(f"{task.name} has no planted optimal sequence")
    assert task.attainable is not None, (
        f"{task.name} declares no attainable optimum, so every regret reported on it is "
        f"against a target nobody has checked is reachable"
    )
    if distance <= task.search_budget:
        return
    assert CAPPED in task.purpose, (
        f"{task.name} plants its optimum {distance} substitutions from the parent and can "
        f"reach {task.search_budget}, and its purpose does not say the radius is capped on "
        f"purpose; that gap is a defect in the task rather than a property of it"
    )


#: Tasks whose per-round radius deliberately *equals* the sequence length, so
#: the mutation budget restricts nothing and is not meant to. Both are four-site
#: empirical libraries: the assay varies four positions and the sequence is
#: those four positions, so the ball is the whole published table -- and no
#: narrower radius would do, since the best measured variant differs from the
#: wild type at every one of the four on both datasets.
#:
#: Named rather than derived from ``max_mutations == length``, which is the
#: condition the assertion below tests: deriving the exemption from it would
#: excuse exactly the tasks that failed and leave the check asserting nothing.
UNCONSTRAINED_BY_DESIGN = ("gb1-anchor", "trpb-anchor")


@pytest.mark.parametrize("task", MAIN, ids=lambda t: t.name)
def test_the_per_round_radius_is_below_the_sequence_length(task):
    # A radius at or above the sequence length is not a constraint, and a task
    # claiming to measure constrained search under one measures nothing.
    length = task.landscape().sequence_length
    assert task.max_mutations <= length
    if task.name not in UNCONSTRAINED_BY_DESIGN:
        assert task.protocol.constrains_search(length), (
            f"{task.name} allows {task.max_mutations} of {length} positions per round, so its "
            f"mutation budget does not restrict anything"
        )


@pytest.mark.parametrize("task", MAIN, ids=lambda t: t.name)
def test_task_budget_matches_its_protocol(task):
    # The environment reads the task's, `constrains_search` reports the
    # protocol's, and only the protocol's survives into a stored record.
    assert task.max_mutations == task.protocol.max_mutations


# --------------------------------------------------------------------------
# The bracket, checked against enumeration rather than against itself.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_certified_upper_bound_is_never_exceeded(seed):
    task = ehrlich_task(seed=seed)
    landscape, env = environment(task)
    reachable = env.reachable_terminal_states()
    values = np.asarray(landscape.evaluate(reachable))

    upper, _ = _certified_upper_bound(landscape, env.parent, task.max_mutations)
    assert values.max() <= upper + 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_searched_lower_bound_is_actually_attained(seed):
    task = ehrlich_task(seed=seed)
    landscape, env = environment(task)
    reachable = env.reachable_terminal_states()
    values = np.asarray(landscape.evaluate(reachable))

    lower, _ = _searched_lower_bound(
        landscape,
        env,
        env.parent,
        task.max_mutations,
        beam_width=8,
        placements=3,
        patience=4,
        ceiling=1.0,
    )
    # Not merely "no larger than the optimum": the bound claims a construction
    # exists, so some reachable design must actually carry that value.
    assert np.isclose(values, lower).any()


@pytest.mark.parametrize("seed", range(4))
def test_exact_answer_lies_inside_the_bracket(seed):
    task = ehrlich_task(seed=seed)
    landscape, env = environment(task)
    exact = float(np.asarray(landscape.evaluate(env.reachable_terminal_states())).max())

    upper, _ = _certified_upper_bound(landscape, env.parent, task.max_mutations)
    lower, _ = _searched_lower_bound(
        landscape,
        env,
        env.parent,
        task.max_mutations,
        beam_width=8,
        placements=3,
        patience=4,
        ceiling=upper,
    )
    assert lower <= exact <= upper


# --------------------------------------------------------------------------
# Exact where it can be measured, bracketed where it cannot, never confused.
# --------------------------------------------------------------------------


def test_small_instance_is_reported_as_exact():
    result = attainable_optimum(ehrlich_task())
    assert result.is_exact
    assert result.exact == result.lower == result.upper
    assert "exact" in result.method


def test_large_instance_is_reported_as_an_interval(monkeypatch):
    # Force the bounded walk to give up, which is what happens on the real
    # tasks; the toy is otherwise the only thing standing between this test and
    # a benchmark-sized run.
    monkeypatch.setattr("evogfn.benchmark.attainable.MAX_REACHABLE_CELLS", 64)
    result = attainable_optimum(
        ehrlich_task(budget=4, params=UNENUMERABLE), beam_width=8, placements=3
    )
    assert result.lower <= result.upper
    if result.is_exact:
        assert "pinned" in result.method
    else:
        assert "bounded" in result.method
        assert result.solvable_headroom > 0


def test_gb1_anchor_has_no_regret_floor():
    # The control. Four sites and a budget of four means everything measured is
    # reachable, so a floor here would mean the audit itself was wrong.
    anchor = next(task for task in MAIN if task.name == "gb1-anchor")
    result = attainable_optimum(anchor)
    assert result.is_exact
    assert result.regret_floor == (0.0, 0.0)


# --------------------------------------------------------------------------
# Reachability is not implied by budget.
# --------------------------------------------------------------------------


def test_planted_optimum_out_of_budget_is_unreachable():
    task = ehrlich_task(budget=1, params=UNENUMERABLE)
    distance = planted_distance(task)
    assert distance is not None
    assert distance > 1
    assert planted_optimum_reachable(task) is False


def test_planted_optimum_reachability_is_undefined_without_a_planted_one():
    anchor = next(task for task in MAIN if task.name == "gb1-anchor")
    assert planted_optimum_reachable(anchor) is None
    assert planted_distance(anchor) is None


# --------------------------------------------------------------------------
# Re-anchoring: cumulative reach, priced without turning it on anywhere.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", MAIN, ids=lambda t: t.name)
def test_per_round_budget_reaches_the_planted_distance(task):
    bound = per_round_budget(task)
    distance = planted_distance(task)
    if bound is None:
        assert distance is None
        pytest.skip(f"{task.name} has no planted optimal sequence")
    assert distance is not None
    # The whole claim of the counting bound: rounds of it clear the distance,
    # and one fewer per round would not.
    assert bound * task.protocol.rounds >= distance
    assert (bound - 1) * task.protocol.rounds < distance


def test_re_anchoring_reaches_at_least_as_much_as_one_ball():
    task = ehrlich_task(budget=2, params=UNENUMERABLE)
    fixed = attainable_optimum(task, beam_width=16, placements=6)
    chained = reanchored_attainable(task, per_round=2, rounds=6, beam_width=16, placements=6)
    # Six rounds of two is a superset of one round of two, so the searched
    # bound cannot fall and the certified bound cannot tighten.
    assert chained.lower >= fixed.lower
    assert chained.upper >= fixed.upper
    assert chained.budget == 12


def test_re_anchoring_is_bounded_at_the_cumulative_budget():
    task = ehrlich_task(budget=2, params=UNENUMERABLE)
    chained = reanchored_attainable(task, per_round=3, rounds=5, beam_width=16, placements=6)
    ceiling, _ = _certified_upper_bound(task.landscape(), task.parent(task.landscape()), 15)
    assert chained.upper <= ceiling + 1e-12
    assert chained.lower <= chained.upper


def test_re_anchoring_rejects_a_non_positive_shape():
    with pytest.raises(ValueError, match="must be positive"):
        reanchored_attainable(ehrlich_task(), per_round=0, rounds=3)


# --------------------------------------------------------------------------
# The return type refuses to describe an impossible quantity.
# --------------------------------------------------------------------------


def test_lower_above_upper_is_rejected():
    with pytest.raises(ValueError, match="bound derivation is wrong"):
        AttainableOptimum(
            task="t", budget=4, nominal=1.0, lower=0.9, upper=0.5, exact=None, method="x"
        )


def test_exact_outside_the_bracket_is_rejected():
    with pytest.raises(ValueError, match="lies outside"):
        AttainableOptimum(
            task="t", budget=4, nominal=1.0, lower=0.2, upper=0.5, exact=0.7, method="x"
        )


def test_regret_floor_and_rescoring_agree_with_the_bracket():
    result = AttainableOptimum(
        task="t", budget=4, nominal=1.0, lower=0.25, upper=0.5, exact=None, method="x"
    )
    assert result.regret_floor == (0.5, 0.75)
    assert result.solvable_headroom == pytest.approx(0.25)
    assert result.regret_against_attainable(0.25) == (0.0, 0.25)
    assert result.solved_by(0.25)
    assert not result.solved_by(0.2)


def test_budget_must_be_positive():
    with pytest.raises(ValueError, match="budget must be at least 1"):
        attainable_optimum(ehrlich_task(), budget=0)


def test_landscape_without_an_optimum_has_no_floor_to_report():
    class Unmeasured(EhrlichLandscape):
        @property
        def optimum(self):
            return None

    task = Task(
        name="unmeasured",
        purpose="a landscape that cannot say what perfect is",
        build=lambda: Unmeasured(**TOY, seed=0),
        protocol=Protocol(rounds=1, batch_size=8, max_mutations=2),
        max_mutations=2,
    )
    with pytest.raises(ValueError, match="no optimum"):
        attainable_optimum(task)
