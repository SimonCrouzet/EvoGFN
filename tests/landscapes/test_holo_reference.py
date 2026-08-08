"""Guard the bridge to holo-bench's own Ehrlich implementation.

Two kinds of test live here, and they catch different things.

The tests that need holo assert the claim the port exists to establish: that
this package's evaluator, attainability audit and CMA-ES repairs all run against
a reference instance and agree with the reference where they must. They skip
when no holo environment is installed, because a CI machine without one should
say "not checked here" rather than "passed".

The tests that do *not* need holo guard the bridge itself, and they matter more
than they look. ``HoloEhrlichLandscape`` bypasses ``EhrlichLandscape.__init__``
and installs the attributes that constructor would have set; if ``ehrlich.py``
gains or renames one, nothing in the type system notices and the adapter goes on
scoring against a stale field. Likewise, an interpreter search that silently
fell back to something that is not holo would turn the entire cross-validation
into a comparison of this package with itself, and still pass.

The sweep's arithmetic is guarded here too, and for the same reason the sweep
exists: the port's findings are *rates over instances*, and a rate reported
without an interval, or with an interval that collapses to a point when every
draw agrees, is what let "4 of 5" read as a settled number.
"""

import re

import numpy as np
import pytest

from evogfn.benchmark.holo_port import (
    LARGE_SPACE_AUDIT_SEEDS,
    REFERENCE_SEEDS,
    REFERENCE_SHAPES,
    InstanceOutcome,
    ReferenceShape,
    _banded_fraction,
    audit_reference,
    compare_instances,
    compare_rewards,
    log10_feasible_count,
    matched_parameters,
    mean_estimate,
    rate_estimate,
    reference_task,
    repair_reference,
    summarise,
    sweep_shape,
)
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.holo_reference import (
    INTERPRETER_ENV_VAR,
    HoloEhrlichLandscape,
    HoloUnavailableError,
    ReferenceInstance,
    ReferenceParameters,
    find_interpreter,
    load_reference_instance,
    reference_scores,
    require_interpreter,
)

HOLO = find_interpreter()
needs_holo = pytest.mark.skipif(HOLO is None, reason="no interpreter can import holo")

# A reference instance small enough to reason about by hand, fabricated rather
# than fetched so the bridge's own invariants can be tested without holo. One
# motif [0, 1] at offsets [0, 2] is satisfied at placement 0 by this sequence.
HAND_BUILT = ReferenceInstance(
    parameters=ReferenceParameters(num_states=3, dim=6, num_motifs=1, motif_length=2),
    transition_matrix=np.full((3, 3), 1.0 / 3.0),
    motifs=np.array([[0, 1]], dtype=np.int64),
    gaps=np.array([[2]], dtype=np.int64),
    optimal_solution=np.array([0, 2, 1, 0, 2, 1], dtype=np.int64),
    initial_solutions=np.array([[2, 2, 2, 2, 2, 2], [1, 0, 1, 0, 1, 0]], dtype=np.int64),
    optimal_value=1.0,
)


def test_adapter_installs_every_attribute_the_constructor_would_have():
    """The bypassed constructor's fields and the adapter's must stay in step.

    ``HoloEhrlichLandscape`` cannot call ``EhrlichLandscape.__init__`` -- that
    constructor generates an instance and takes no injection point -- so it sets
    the attributes itself. A field added to ``ehrlich.py`` would then be missing
    on every reference landscape, and would surface as an ``AttributeError``
    from somewhere deep in the audit, or worse as a default that scores.
    """
    generated = EhrlichLandscape(
        sequence_length=16, vocab_size=5, n_motifs=1, motif_length=2, max_spacing=2
    )
    assert set(vars(generated)) == set(HoloEhrlichLandscape.installed_attributes)


def test_adapter_sets_them_all():
    """And the adapter really does set each one, not merely list it."""
    adapter = HoloEhrlichLandscape(HAND_BUILT)
    assert set(HoloEhrlichLandscape.installed_attributes) <= set(vars(adapter))


def test_gaps_become_cumulative_offsets():
    """holo spaces motifs by gaps, this package by offsets from the placement.

    Getting this backwards would place every motif element at the wrong absolute
    position, and on a dense transition matrix the resulting landscape would
    still look perfectly well formed -- just a different function than holo's.
    """
    instance = ReferenceInstance(**{**vars(HAND_BUILT), "gaps": np.array([[2, 3]], dtype=np.int64)})
    assert instance.offsets.tolist() == [[0, 2, 5]]


def test_planted_optimum_is_rescored_rather_than_trusted():
    """The adapter re-scores holo's own optimum through this package's evaluator.

    That check is the port's tripwire: a mistranscribed spacing, motif or
    alphabet size shows up here as an optimum that no longer reaches 1.0,
    instead of as a silent disagreement in a comparison two steps later.
    """
    landscape = HoloEhrlichLandscape(HAND_BUILT)
    assert landscape.evaluate(HAND_BUILT.optimal_solution[None, :])[0, 0] == pytest.approx(1.0)

    broken = ReferenceInstance(**{**vars(HAND_BUILT), "motifs": np.array([[2, 2]], dtype=np.int64)})
    with pytest.raises(RuntimeError, match="transcribed wrongly"):
        HoloEhrlichLandscape(broken)


def test_quantisation_that_does_not_divide_is_refused():
    """holo and this package quantise differently when q does not divide k.

    holo uses ``count // ceil(k/q) / (k / ceil(k/q))`` and this package
    ``count // (k // q) / q``. Scoring such an instance would compare two
    genuinely different functions and report the gap as a generator
    discrepancy, which is the one conclusion this whole module must not reach
    by accident.
    """
    instance = ReferenceInstance(
        **{
            **vars(HAND_BUILT),
            "parameters": ReferenceParameters(
                num_states=3, dim=6, num_motifs=1, motif_length=3, quantization=2
            ),
        }
    )
    with pytest.raises(ValueError, match="does not divide"):
        HoloEhrlichLandscape(instance)


def test_wild_type_comes_from_holos_own_draws():
    """A campaign's anchor must be holo's draw, not ours from holo's chain.

    ``Task.parent`` reaches the wild type through ``feasible_sequence``, and the
    inherited implementation would walk the reference chain with this package's
    sampler -- putting our sampling back into a comparison that exists to keep
    it out.
    """
    landscape = HoloEhrlichLandscape(HAND_BUILT)
    pool = HAND_BUILT.initial_solutions
    assert landscape.feasible_sequence(0).tolist() == pool[0].tolist()
    assert landscape.feasible_sequence(3).tolist() == pool[3 % pool.shape[0]].tolist()


def test_feasible_count_matches_brute_force():
    """The closed-form feasible-set size must equal an enumeration.

    The count is a matrix power with a renormalisation on every step, so an
    error in the scaling would be invisible: it would still return a plausible
    number, just a wrong one, and every density comparison in the port would
    inherit it.
    """
    rng = np.random.default_rng(0)
    transitions = (rng.random((3, 3)) < 0.6).astype(float)
    transitions[np.arange(3), np.arange(3)] = 1.0  # keep the chain non-degenerate

    length = 6
    grid = np.array(np.meshgrid(*[np.arange(3)] * length, indexing="ij")).reshape(length, -1).T
    feasible = np.all(transitions[grid[:, :-1], grid[:, 1:]] > 0, axis=1).sum()
    assert 10 ** log10_feasible_count(transitions, length) == pytest.approx(float(feasible))


def test_banded_rows_are_recognised_when_they_wrap():
    """A wrap-around band must count as banded, not as scattered.

    holo's mask is ``banded_square_matrix``, whose rows wrap by construction, so
    most rows of a real instance are exactly the case a linear left-to-right
    scan gets wrong -- it sees ``{v-1, 0, 1}`` as two runs. Getting this wrong
    reports holo's chain as unstructured, which reverses the one qualitative
    finding this metric exists to state.
    """
    size = 8
    wrapped = np.zeros((size, size), dtype=float)
    for row in range(size):
        for offset in (-1, 0, 1):
            wrapped[row, (row + 3 + offset) % size] = 1.0
    assert _banded_fraction(wrapped) == 1.0

    scattered = np.zeros((size, size), dtype=float)
    scattered[:, [0, 3, 6]] = 1.0
    assert _banded_fraction(scattered) < 1.0


def test_no_interpreter_raises_rather_than_substituting(monkeypatch):
    """A missing holo must fail, never fall back.

    Every conclusion the port draws rests on the other side of the comparison
    being code this project did not write. A fallback -- to our own generator,
    to a transcription, to anything -- would let the port report agreement it
    never measured, and that failure would look exactly like success.
    """
    monkeypatch.setenv(INTERPRETER_ENV_VAR, "/nonexistent/python")
    monkeypatch.setattr("evogfn.landscapes.holo_reference.DEFAULT_INTERPRETERS", ())
    monkeypatch.setattr("evogfn.landscapes.holo_reference.sys.executable", "/nonexistent/python")
    assert find_interpreter() is None
    with pytest.raises(HoloUnavailableError, match=re.escape("numpy<2.0")):
        require_interpreter()


def _outcome(seed, *, unreachable, pinned, repaired):
    """One sweep row, with only the fields the summary reads."""
    return InstanceOutcome(
        shape="toy",
        seed=seed,
        planted_distance=10,
        reachable_at_budget=False,
        reachable_unbounded=not unreachable,
        attainable_lower=0.5,
        attainable_upper=0.5 if pinned else 1.0,
        pinned=pinned,
        repaired={"greedy": repaired, "exact": repaired},
        all_constructible=True,
    )


def test_a_unanimous_rate_does_not_report_certainty():
    """Five of five must not come back as "1.000, no interval".

    This is the arithmetic behind the whole sweep. The port's headline was a
    rate over five draws, and the textbook normal interval on 5/5 is exactly
    ``[1, 1]`` -- a claim of certainty from five instances. Wilson's does not
    collapse, and the gap between the two is the difference between "we saw it
    five times" and "it always happens".
    """
    five = rate_estimate("unreachable", 5, 5)
    assert five.point == 1.0
    assert five.low < 0.6
    assert five.high == 1.0

    hundred = rate_estimate("unreachable", 100, 100)
    assert hundred.low > five.low, "more draws must buy a tighter interval"


def test_the_interval_is_reported_at_the_ends_too():
    """Zero of n is a rate with an interval, not a proven impossibility."""
    none = rate_estimate("unreachable", 0, 40)
    assert none.point == 0.0
    assert none.low == 0.0
    assert 0.0 < none.high < 0.2


def test_five_draws_could_never_have_been_significant():
    """The sign test on the port's original five draws, stated as a number.

    ``unanimity_floor()`` is six: at five instances even a unanimous result does
    not reach 0.05, so "4 of 5" was below the count at which its own strongest
    possible form would have been evidence.
    """
    assert rate_estimate("unreachable", 5, 5).sign_p > 0.05
    assert rate_estimate("unreachable", 4, 5).sign_p > 0.05
    assert rate_estimate("unreachable", 100, 100).sign_p < 0.05


def test_a_single_draw_carries_an_infinite_interval():
    """One instance says nothing about spread, and must not claim to.

    A zero-width interval on one draw is how a single-instance number becomes a
    reported constant, which is the failure the sweep was asked to correct.
    """
    single = mean_estimate("repaired_fraction", [0.672])
    assert single.mean == pytest.approx(0.672)
    assert single.low == -float("inf")
    assert single.high == float("inf")


def test_the_mean_carries_the_range_beside_it():
    """A claim of the form "at least x" is read against the minimum, not the mean."""
    estimate = mean_estimate("repaired_fraction", [0.5, 0.75, 1.0])
    assert estimate.minimum == pytest.approx(0.5)
    assert estimate.maximum == pytest.approx(1.0)
    assert estimate.low < estimate.mean < estimate.high


def test_the_summary_counts_unreachability_the_way_the_finding_states_it():
    """``reachable_unbounded is False`` is the finding; ``None`` is not.

    ``planted_optimum_reachable`` returns ``None`` for a landscape with no
    planted optimum, and counting that as unreachable would inflate the
    headline with instances that were never measured.
    """
    rows = [
        _outcome(0, unreachable=True, pinned=True, repaired=0.7),
        _outcome(1, unreachable=False, pinned=False, repaired=0.9),
    ]
    unknown = InstanceOutcome(**{**vars(rows[0]), "seed": 2, "reachable_unbounded": None})
    summary = summarise("toy", [*rows, unknown])
    assert summary.unreachable.successes == 1
    assert summary.unreachable.trials == 3
    assert summary.pinned.successes == 2
    assert summary.repaired["greedy"].mean == pytest.approx((0.7 + 0.9 + 0.7) / 3)


def test_an_empty_sweep_is_refused_rather_than_summarised():
    """Zero instances would produce a table of NaNs that still prints."""
    with pytest.raises(ValueError, match="no instance was drawn"):
        summarise("toy", [])


def test_every_shape_draws_the_suites_own_seed_count():
    """The reduction at L=256 must not reach the findings that cost nothing.

    ``run_suite.py`` states the same thing about campaigns -- cost differs by an
    order of magnitude between L=4 and L=256, not because the claims differ. Here
    only the *audit* is expensive, so only the audit is reduced: every shape,
    L=256 included, draws the full hundred instances, and the unreachability rate
    and both repaired fractions are estimated from all of them.
    """
    assert all(shape.seeds == REFERENCE_SEEDS for shape in REFERENCE_SHAPES)


def test_only_the_audit_is_reduced_and_the_shape_carries_the_reduction():
    """A reduced count belongs to the shape that needs it, not to a call site.

    A caller that forgot it would silently over-run a sweep that takes minutes
    per instance, or -- worse -- under-run one that does not and report a
    hundred-draw claim from thirty.
    """
    by_name = {shape.name: shape for shape in REFERENCE_SHAPES}
    assert by_name["large-space"].audit_seeds == LARGE_SPACE_AUDIT_SEEDS
    assert LARGE_SPACE_AUDIT_SEEDS < REFERENCE_SEEDS
    assert all(
        shape.audit_seeds is None for shape in REFERENCE_SHAPES if shape.name != "large-space"
    )

    large = by_name["large-space"]
    assert large.audits(0)
    assert large.audits(LARGE_SPACE_AUDIT_SEEDS - 1)
    assert not large.audits(LARGE_SPACE_AUDIT_SEEDS)
    assert by_name["diagnostic"].audits(REFERENCE_SEEDS - 1)


def test_an_unaudited_draw_is_not_a_bracket():
    """``pinned is None`` must not be counted as an audit that failed to pin.

    The reduced audit count at L=256 would otherwise report itself as a shape
    whose bracket almost never closes -- a made-up finding produced entirely by
    how many instances someone could afford to audit.
    """
    audited = _outcome(0, unreachable=True, pinned=True, repaired=0.9)
    skipped = InstanceOutcome(
        **{
            **vars(audited),
            "seed": 1,
            "pinned": None,
            "attainable_lower": None,
            "attainable_upper": None,
        }
    )
    summary = summarise("toy", [audited, skipped])
    assert summary.instances == 2
    assert summary.audited == 1
    assert summary.pinned.trials == 1
    assert summary.pinned.point == 1.0
    assert summary.attainable.n == 1
    assert summary.repaired["greedy"].n == 2, "the cheap findings keep every draw"


def test_two_shapes_map_onto_one_reference_configuration():
    """holo has no density knob, so ``feasibility`` and ``evolvepro`` coincide.

    Our two tasks differ only in ``transition_density`` (0.15 against 0.5) and
    in the campaign shape. holo derives density from the alphabet size, so its
    draws for the two are from the *same* distribution -- and reading their
    swept answers as two independent shapes would double-count one.
    """
    by_name = {shape.name: shape for shape in REFERENCE_SHAPES}
    feasibility = by_name["feasibility"].at(7)
    evolvepro = by_name["evolvepro"].at(7)
    assert feasibility == evolvepro
    assert by_name["feasibility"].rounds != by_name["evolvepro"].rounds


def test_a_draw_seeds_the_instance_and_nothing_else():
    """``at`` must move the instance seed and leave the shape alone."""
    shape = REFERENCE_SHAPES[0]
    assert shape.at(11).random_seed == 11
    assert shape.at(11).dim == shape.parameters.dim
    assert shape.at(11).num_motifs == shape.parameters.num_motifs


@pytest.fixture(scope="module")
def reference_instance():
    """One reference instance, shared: each fetch is a subprocess."""
    return load_reference_instance(
        ReferenceParameters(num_states=20, dim=32, num_motifs=2, motif_length=4, random_seed=0)
    )


@needs_holo
def test_rewards_agree_exactly_with_the_reference(reference_instance):
    """The two implementations must score holo's instance identically.

    This is the one place a difference would be a bug rather than a design
    choice: the instance is holo's, so only the arithmetic differs. The reward
    is quantised to multiples of 1/q^c, so any disagreement here is a whole
    level -- a placement window off by one, or a quantisation floor applied at
    the wrong point -- and not floating-point noise.
    """
    agreement = compare_rewards(reference_instance, n_random=64, seed=0)
    assert agreement.n_feasible > 0, "an all-infeasible batch would agree at -inf and prove nothing"
    assert agreement.feasibility_mismatches == 0
    assert agreement.max_absolute_difference == 0.0


@needs_holo
def test_the_sweep_measures_every_draw_and_audits_only_the_declared_ones():
    """A reduced audit count must not reduce anything else.

    The failure this catches is quiet and would look like a finding: if the
    audit's cost were allowed to set the draw count, the unreachability rate --
    which costs nothing to measure and is the claim most likely to be challenged
    -- would be reported from the same handful of instances as the beam search,
    and the note would be back to quoting a rate from too few draws.
    """
    shape = ReferenceShape(
        name="sweep-probe",
        parameters=ReferenceParameters(num_states=6, dim=12, num_motifs=1, motif_length=2),
        max_mutations=3,
        rounds=2,
        batch_size=8,
        seeds=2,
        audit_seeds=1,
    )
    outcomes = sweep_shape(shape)
    assert [outcome.seed for outcome in outcomes] == [0, 1]
    assert outcomes[0].pinned is not None
    assert outcomes[1].pinned is None, "the second draw is past the audit count"
    assert all(outcome.reachable_unbounded is not None for outcome in outcomes)
    assert all(outcome.repaired for outcome in outcomes)

    summary = summarise(shape.name, outcomes)
    assert summary.instances == 2
    assert summary.audited == 1
    assert summary.unreachable.trials == 2
    assert summary.repaired["greedy"].n == 2


@needs_holo
def test_reference_scores_are_holos_own(reference_instance):
    """The reference values come from holo, not from a matrix we shipped back.

    ``reference_scores`` rebuilds the instance in the other interpreter from the
    same parameters rather than sending our transcription across, so nothing
    about how the bridge reads an instance can influence the numbers it is
    checked against.
    """
    values = reference_scores(
        reference_instance.parameters, reference_instance.optimal_solution[None, :]
    )
    assert values.tolist() == [1.0]


@needs_holo
def test_instance_comparison_reports_both_sides(reference_instance):
    """The generator comparison must measure both, not assert they are equal."""
    ours = EhrlichLandscape(sequence_length=32, vocab_size=20, n_motifs=2, motif_length=4, seed=0)
    assert matched_parameters(ours).dim == 32
    quantities = {row.quantity for row in compare_instances(ours, reference_instance)}
    assert "transition density" in quantities
    assert "log10 |feasible set|" in quantities


@needs_holo
def test_attainability_audit_runs_against_the_reference(reference_instance):
    """The audit must run unmodified on a landscape holo generated.

    Its whole claim is that regret is reported against what a protocol can
    reach. If that claim only holds on instances we generate it is a property of
    our generator, not a methodological contribution, which is exactly what the
    manuscript's limitation says.
    """
    task = reference_task(reference_instance, max_mutations=8, rounds=4, batch_size=32)
    audit = audit_reference(task)
    assert audit.planted_distance is not None
    assert audit.planted_reachable is not None
    assert audit.optimum.lower <= audit.optimum.upper
    assert audit.optimum.upper <= audit.optimum.nominal


@needs_holo
@pytest.mark.parametrize("policy", ["greedy", "exact"])
def test_both_repairs_run_against_the_reference(reference_instance, policy):
    """Both decoders must produce constructible designs on holo's instance.

    The reported ``repaired_fraction`` is a benchmarking claim about the
    published relaxation baselines, and it currently rests on our instances. A
    repair that only worked against our transition matrices would make that
    claim about our generator instead.
    """
    task = reference_task(reference_instance, max_mutations=8, rounds=4, batch_size=32)
    (measurement,) = repair_reference(task, n_designs=32, seed=0, policies=(policy,))
    assert measurement.constructible
    assert 0.0 <= measurement.repaired_fraction <= 1.0
