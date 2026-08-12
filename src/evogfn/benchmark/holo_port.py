r"""Run this package's procedures against holo-bench's own Ehrlich instances.

The manuscript carries a limitation that reduces to one sentence: our Ehrlich
generator is ours, so the numbers it produces cannot be quoted beside Stanton et
al.'s. Arguing about the limitation is not worth anything. Running the
procedures on the reference instance is, and that is all this module does.

It answers three questions, in the order they have to be answered:

**Do the two generators agree, and where do they not?** They cannot agree on the
*instance*, since neither draws from the other's distribution -- so
[compare_instances][evogfn.benchmark.holo_port.compare_instances] measures the
distributions rather than asserting equality, on quantities that decide how hard
an instance is: the density of the transition matrix, whether repeats are
permitted, the exact size of the feasible set, how far the motifs are spread,
and how far the planted optimum sits from the wild type. What *must* agree is
the reward, because both sides claim to implement the same closed form; that is
checked separately by
[compare_rewards][evogfn.benchmark.holo_port.compare_rewards], which scores one
reference instance twice -- once through holo, once through this package.

**Does the attainability audit run on the reference instance?** This is the
substance. The audited attainable optimum -- what a protocol's search space
actually contains, as against the landscape's own maximum -- currently rests on
instances we generated, which is exactly the position the limitation describes.
[audit_reference][evogfn.benchmark.holo_port.audit_reference] runs the
unmodified audit on a task whose landscape is holo's.

**Do both repairs run on the reference instance?** The reported result that
relaxation-based baselines owe their designs to an unspecified repair is
measured by ``repaired_fraction``, and it too rests on our instances.
[repair_reference][evogfn.benchmark.holo_port.repair_reference] measures it on
holo's, for the greedy and exact decoders both.

**And how often?** The three questions above are answered per *instance*, and an
instance is a draw. The port's first pass drew one per shape, which makes its
headline -- that holo's own verified planted optimum has no legal construction
order -- a rate estimated from five draws.
[sweep_shape][evogfn.benchmark.holo_port.sweep_shape] repeats the whole
per-instance measurement across seeds and
[summarise][evogfn.benchmark.holo_port.summarise] reports each answer as a rate
with an interval. Nothing here is a campaign: no sampler is trained, no oracle
budget is spent, and the cost is holo's constructor plus the audit's beam. So the
seed count is set by what the audit costs, not by what a campaign would, and it
is stated per shape wherever it falls below
[REFERENCE_SEEDS][evogfn.benchmark.holo_port.REFERENCE_SEEDS].

What is deliberately *not* swept is the instance comparison of
`compare_instances`. Every difference it reports -- the transition density, the
banding, the forced diagonal, the size of the feasible set -- is a deterministic
consequence of holo's construction rather than a draw from it: the bandwidth is
``int(0.4 * num_states)`` with no seed in it, ``banded_square_matrix`` is banded
for every permutation of its rows, and ``repeats_always_possible`` forces the
diagonal on every instance. Sweeping those for symmetry would imply they were
uncertain, which would be a worse report than the single instance is.

Nothing here is imported by
[RESULT_DEPENDENCIES][evogfn.benchmark.suite.RESULT_DEPENDENCIES] or anything
reachable from it, which is deliberate: this module can grow without
invalidating a single stored campaign record.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from evogfn.algorithms.baselines.cmaes import CMAES
from evogfn.benchmark.attainable import (
    attainable_optimum,
    planted_distance,
    planted_optimum_reachable,
    reanchored_attainable,
)
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.statistics import t_critical, unanimity_floor, unanimity_p
from evogfn.benchmark.tasks import Task
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.holo_reference import (
    HoloEhrlichLandscape,
    ReferenceParameters,
    load_reference_instance,
    reference_scores,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    import numpy.typing as npt

    from evogfn.benchmark.attainable import AttainableOptimum
    from evogfn.core.types import Tokens
    from evogfn.landscapes.holo_reference import ReferenceInstance

#: Repair policies the CMA-ES decoder offers beyond doing nothing. Both are run,
#: because the reported ``repaired_fraction`` is a claim about the relaxation
#: and a single decoder could be hiding a decoder-specific artefact.
REPAIR_POLICIES: tuple[str, ...] = ("greedy", "exact")


@dataclass(frozen=True)
class InstanceComparison:
    """How a reference instance and one of ours differ at matched parameters.

    Every field is a property of the *instance*, not of a search over it, so a
    difference here is a difference in the generators rather than in anything
    this package does with them.

    Attributes:
        reference: The reference instance's value.
        ours: Ours.
        quantity: What is being compared, for a report to quote.
    """

    quantity: str
    reference: float
    ours: float

    @property
    def ratio(self) -> float:
        """Reference over ours, or ``inf`` when ours is zero."""
        return self.reference / self.ours if self.ours != 0.0 else float("inf")

    def __repr__(self) -> str:
        """Name the quantity and both sides."""
        return f"{self.quantity}: holo={self.reference:.4g} ours={self.ours:.4g}"


@dataclass(frozen=True)
class RewardAgreement:
    r"""Whether the two implementations score the same sequences the same way.

    This is the one comparison where disagreement is a bug rather than a design
    difference: both sides claim to implement Stanton et al.'s closed form, and
    they are given the *same* instance -- holo's -- so nothing but the
    arithmetic differs.

    Attributes:
        n_sequences: How many sequences were scored through both.
        n_feasible: How many the reference judged feasible. A comparison run
            entirely on infeasible sequences would agree at $-\\infty$ and prove
            nothing, so this is reported rather than assumed non-zero.
        feasibility_mismatches: Sequences one side called feasible and the other
            did not.
        max_absolute_difference: Largest gap on the sequences both called
            feasible, in reward units. The reward is quantised to multiples of
            ``1/q^c``, so anything non-zero here is a whole level, not rounding.
    """

    n_sequences: int
    n_feasible: int
    feasibility_mismatches: int
    max_absolute_difference: float

    @property
    def agrees(self) -> bool:
        """Whether the two implementations are indistinguishable on this batch."""
        return self.feasibility_mismatches == 0 and self.max_absolute_difference == 0.0

    def __repr__(self) -> str:
        """State the verdict and what it rests on."""
        verdict = "exact agreement" if self.agrees else "DISAGREEMENT"
        return (
            f"{verdict} over {self.n_sequences} sequences ({self.n_feasible} feasible): "
            f"{self.feasibility_mismatches} feasibility mismatches, "
            f"max |Δreward| = {self.max_absolute_difference:g}"
        )


@dataclass(frozen=True)
class RepairMeasurement:
    """What a repair policy had to do on the reference instance.

    Attributes:
        policy: ``"greedy"`` or ``"exact"``.
        repaired_fraction: Share of decoded designs whose raw argmax the
            environment could not build. At 1.000 every design credited to the
            arm is the repair's work.
        n_decoded: Designs decoded, which is what the fraction is over.
        constructible: Whether every returned design was constructible. A repair
            that leaves one behind has failed at its only job, so this is
            checked rather than assumed.
    """

    policy: str
    repaired_fraction: float
    n_decoded: int
    constructible: bool

    def __repr__(self) -> str:
        """State the policy and its repaired fraction."""
        return (
            f"{self.policy}: repaired_fraction = {self.repaired_fraction:.3f} "
            f"over {self.n_decoded} designs, all constructible = {self.constructible}"
        )


def matched_parameters(landscape: EhrlichLandscape, *, random_seed: int = 0) -> ReferenceParameters:
    """The reference parameters that describe the same problem shape as ours.

    "Matched" is as close as the two generators come: holo has no density knob
    (its bandwidth is fixed at ``int(0.4 * num_states)``) and no ``max_spacing``
    (its spacings come from a simplex draw over the slack), so those two
    arguments of ours have no counterpart and their difference is itself one of
    the findings.

    Args:
        landscape: The instance whose shape is being matched.
        random_seed: Seeds the reference instance.

    Returns:
        Parameters for a reference instance of the same length, alphabet, motif
        count, motif length and quantisation.
    """
    return ReferenceParameters(
        num_states=landscape.alphabet.size,
        dim=landscape.sequence_length,
        num_motifs=landscape.n_motifs,
        motif_length=landscape.motif_length,
        quantization=landscape.quantization,
        random_seed=random_seed,
    )


def log10_feasible_count(transitions: npt.NDArray[np.floating], length: int) -> float:
    r"""Exact size of the feasible set, as a base-10 logarithm.

    The feasible set is the set of walks of length ``length`` in the graph whose
    edges are the transition matrix's non-zeros, so its size is
    $\mathbf{1}^\top A^{L-1} \mathbf{1}$ with $A$ the 0/1 adjacency. That is a
    measurement rather than the Monte-Carlo estimate a sampler would give, and
    at the lengths used here the count runs past $10^{200}$ -- hence the
    logarithm and the per-step renormalisation, without which the power
    overflows long before it is finished.

    Args:
        transitions: A square matrix whose zeros mark forbidden adjacencies.
        length: Sequence length.

    Returns:
        $\log_{10}$ of the number of feasible sequences.
    """
    adjacency = (np.asarray(transitions) > 0.0).astype(np.float64)
    vector = np.ones(adjacency.shape[0], dtype=np.float64)
    log_scale = 0.0
    for _ in range(length - 1):
        vector = adjacency @ vector
        norm = float(vector.max())
        vector /= norm
        log_scale += np.log10(norm)
    return log_scale + float(np.log10(vector.sum()))


def _is_contiguous(row: npt.NDArray[np.bool_]) -> bool:
    """Whether a boolean row's true positions form one interval modulo its length.

    Counted cyclically rather than by rotating to the first true index, because
    holo's bands wrap: a set like ``{v-1, 0, 1}`` starts at index 0 and would
    look like two runs to a linear scan, which would report the most banded rows
    as the least.
    """
    changes = int(np.count_nonzero(row != np.roll(row, -1)))
    return changes <= 2  # noqa: PLR2004 - one interval has one rise and one fall


def _banded_fraction(transitions: npt.NDArray[np.floating]) -> float:
    """Fraction of tokens whose permitted successors are contiguous modulo ``v``.

    holo masks its transition matrix with a *banded* matrix whose rows are then
    permuted, so every token's permitted successors are an interval of token
    indices; ours draws each entry independently, so they are scattered. That is
    a difference in kind rather than degree -- a banded chain makes the token
    index itself a metric a sampler can exploit -- and a density comparison
    alone would hide it completely.

    The self-transition is treated as a wildcard, present or absent, because
    holo forces the diagonal on top of the band (``repeats_always_possible``).
    Counting that forced entry as a break would report holo's chain as
    unstructured, which is the opposite of what it is.
    """
    allowed = np.asarray(transitions) > 0.0
    size = allowed.shape[0]
    contiguous = 0
    for index, row in enumerate(allowed):
        without_self = row.copy()
        without_self[index] = not row[index]
        contiguous += int(_is_contiguous(row) or _is_contiguous(without_self))
    return float(contiguous) / float(size)


def _motif_adjacency_fraction(
    motifs: npt.NDArray[np.integer], transitions: npt.NDArray[np.floating]
) -> float:
    """Fraction of consecutive motif elements the chain would permit as neighbours.

    holo draws a motif as one contiguous walk of the chain, so every consecutive
    pair inside a motif is a permitted adjacency; we carve motifs out of *spaced*
    positions of a feasible sequence, so consecutive elements are unrelated by
    the chain. This is the sharpest statement of how the two generators differ,
    and it is what makes holo's planted optimum a sequence of runs.
    """
    allowed = np.asarray(transitions) > 0.0
    pairs = np.asarray(motifs)
    if pairs.shape[1] < 2:  # noqa: PLR2004 - a one-token motif has no internal adjacency
        return float("nan")
    return float(np.mean(allowed[pairs[:, :-1], pairs[:, 1:]]))


def compare_instances(
    ours: EhrlichLandscape,
    reference: ReferenceInstance,
    *,
    parent_seed: int = 0,
) -> tuple[InstanceComparison, ...]:
    """Measure the two generators on the quantities that set an instance's difficulty.

    Args:
        ours: An instance from this package's generator.
        reference: An instance from holo's.
        parent_seed: Selects the wild type each side is measured from.

    Returns:
        One comparison per quantity.
    """
    reference_landscape = HoloEhrlichLandscape(reference)
    our_parent = ours.feasible_sequence(parent_seed)
    reference_parent = reference_landscape.feasible_sequence(parent_seed)

    our_offsets = np.asarray(ours.spacings)
    reference_offsets = reference.offsets
    return (
        InstanceComparison(
            "transition density",
            reference.transition_density,
            float(np.mean(np.asarray(ours.transition_matrix) > 0.0)),
        ),
        InstanceComparison(
            "self-transitions permitted (fraction of tokens)",
            float(np.mean(np.diag(reference.transition_matrix) > 0.0)),
            float(np.mean(np.diag(np.asarray(ours.transition_matrix)) > 0.0)),
        ),
        InstanceComparison(
            "log10 |feasible set|",
            log10_feasible_count(reference.transition_matrix, reference_landscape.sequence_length),
            log10_feasible_count(np.asarray(ours.transition_matrix), ours.sequence_length),
        ),
        InstanceComparison(
            "mean motif span (positions)",
            float(np.mean(reference_offsets[:, -1] + 1)),
            float(np.mean(our_offsets[:, -1] + 1)),
        ),
        InstanceComparison(
            "max gap between motif elements",
            float(np.max(reference.gaps)),
            float(np.max(np.diff(our_offsets, axis=1))),
        ),
        InstanceComparison(
            "out-degree spread (std of permitted successors per token)",
            float(np.std(np.count_nonzero(reference.transition_matrix > 0.0, axis=1))),
            float(np.std(np.count_nonzero(np.asarray(ours.transition_matrix) > 0.0, axis=1))),
        ),
        InstanceComparison(
            "tokens whose permitted successors are a contiguous band (self aside)",
            _banded_fraction(reference.transition_matrix),
            _banded_fraction(np.asarray(ours.transition_matrix)),
        ),
        InstanceComparison(
            "consecutive motif elements that are a permitted adjacency",
            _motif_adjacency_fraction(reference.motifs, reference.transition_matrix),
            _motif_adjacency_fraction(np.asarray(ours.motifs), np.asarray(ours.transition_matrix)),
        ),
        InstanceComparison(
            "distinct tokens in the planted optimum",
            float(np.unique(reference.optimal_solution).size),
            float(np.unique(np.asarray(ours.optimal_sequence)).size),
        ),
        InstanceComparison(
            "planted optimum distance from wild type",
            float(np.count_nonzero(reference_parent != reference.optimal_solution)),
            float(np.count_nonzero(our_parent != np.asarray(ours.optimal_sequence))),
        ),
    )


def compare_rewards(
    reference: ReferenceInstance,
    *,
    n_random: int = 256,
    seed: int = 0,
    interpreter: Path | None = None,
) -> RewardAgreement:
    """Score one reference instance through both implementations.

    The batch deliberately mixes three populations, because each catches a
    different failure: the planted optimum and holo's own initial draws are
    feasible and high-scoring, uniform draws are almost all infeasible at any
    useful length, and mutants of the optimum sit at the partial-match levels
    where a quantisation off-by-one lives. A batch of only one kind would agree
    for reasons that have nothing to do with the arithmetic being right.

    Args:
        reference: The instance to score both ways.
        n_random: Uniform and mutated sequences drawn per population.
        seed: Seeds those draws.
        interpreter: Python that can import holo, located automatically if
            omitted.

    Returns:
        The agreement, or the exact shape of its absence.
    """
    landscape = HoloEhrlichLandscape(reference)
    rng = np.random.default_rng(seed)
    length = landscape.sequence_length
    vocabulary = landscape.alphabet.size

    uniform = rng.integers(vocabulary, size=(n_random, length))
    mutants = np.repeat(reference.optimal_solution[None, :], n_random, axis=0)
    sites = rng.integers(length, size=(n_random, max(1, length // 8)))
    for row in range(n_random):
        mutants[row, sites[row]] = rng.integers(vocabulary, size=sites.shape[1])
    batch = np.concatenate(
        [
            reference.optimal_solution[None, :],
            reference.initial_solutions,
            uniform,
            mutants,
        ],
        axis=0,
    ).astype(np.int64)

    theirs = reference_scores(reference.parameters, batch, interpreter=interpreter)
    ours = landscape.evaluate(batch.astype(np.int32))[:, 0]

    their_feasible = np.isfinite(theirs)
    our_feasible = np.isfinite(ours)
    both = their_feasible & our_feasible
    difference = np.abs(theirs[both] - ours[both]) if both.any() else np.zeros(0)
    return RewardAgreement(
        n_sequences=int(batch.shape[0]),
        n_feasible=int(their_feasible.sum()),
        feasibility_mismatches=int(np.count_nonzero(their_feasible != our_feasible)),
        max_absolute_difference=float(difference.max()) if difference.size else 0.0,
    )


def reference_task(  # noqa: PLR0913 - a task is defined by what it declares
    reference: ReferenceInstance,
    *,
    name: str = "holo-reference",
    max_mutations: int = 8,
    rounds: int = 8,
    batch_size: int = 32,
    reanchor: bool = False,
    parent_seed: int = 0,
) -> Task:
    """A benchmark task whose landscape is holo's instance.

    Built as a plain `Task` so that every procedure taking one -- the audit, the
    environment, the samplers -- runs unmodified. The task is not registered in
    the suite: it exists to be audited, not to be campaigned on, and adding it
    to the suite would put this module in the fingerprinted closure.

    Args:
        reference: The instance to wrap.
        name: Task name, which the audit stamps on its result.
        max_mutations: Substitutions permitted per round.
        rounds: Design-build-test-learn cycles.
        batch_size: Designs measured per round.
        reanchor: Whether the anchor moves to each round's best design.
        parent_seed: Selects the wild type from holo's own draws.

    Returns:
        The task.
    """
    return Task(
        name=name,
        purpose="run this package's procedures against holo-bench's own generator",
        build=lambda: HoloEhrlichLandscape(reference),
        protocol=Protocol(
            rounds=rounds,
            batch_size=batch_size,
            max_mutations=max_mutations,
            label="holo reference",
        ),
        max_mutations=max_mutations,
        reanchor=reanchor,
        parent_seed=parent_seed,
    )


@dataclass(frozen=True)
class ReferenceAudit:
    """What the attainability audit found on a reference instance.

    Attributes:
        optimum: The audit's own result, bounds and method text included.
        planted_distance: Substitutions between the wild type and holo's planted
            optimum.
        planted_reachable: Whether any legal ordering of substitutions builds
            that optimum inside the budget. ``False`` with a distance inside the
            budget is the failure mode the audit was written to catch.
    """

    optimum: AttainableOptimum
    planted_distance: int | None
    planted_reachable: bool | None


def audit_reference(task: Task, *, budget: int | None = None) -> ReferenceAudit:
    """Run the unmodified attainability audit against a reference-instance task.

    Args:
        task: A task from `reference_task`.
        budget: Mutation budget to audit under. Defaults to the task's own.

    Returns:
        The audit's result and the two planted-optimum diagnostics beside it.
    """
    return ReferenceAudit(
        optimum=attainable_optimum(task, budget=budget),
        planted_distance=planted_distance(task),
        planted_reachable=planted_optimum_reachable(task, budget=budget),
    )


def repair_reference(
    task: Task,
    *,
    n_designs: int = 128,
    seed: int = 0,
    policies: tuple[str, ...] = REPAIR_POLICIES,
) -> tuple[RepairMeasurement, ...]:
    """Measure each repair policy's ``repaired_fraction`` on a reference instance.

    One proposal round is enough and more would confound the measurement: the
    fraction is a property of the relaxation's geometry against the environment,
    and letting CMA-ES adapt first would mix in how fast it learns to avoid the
    infeasible region, which is a different question.

    Args:
        task: A task from `reference_task`.
        n_designs: Designs decoded per policy.
        seed: Seeds the CMA-ES draws.
        policies: Repair policies to measure.

    Returns:
        One measurement per policy.

    Raises:
        TypeError: If the task's landscape is not an Ehrlich instance, since
            then there is no feasibility constraint for a repair to act on and
            a fraction of 0.0 would be reported as though it were a finding.
    """
    landscape = task.landscape()
    if not isinstance(landscape, EhrlichLandscape):
        raise TypeError(
            f"repair fractions are only meaningful under a feasibility constraint, and "
            f"{type(landscape).__name__} has none"
        )
    parent = task.parent(landscape)

    measurements: list[RepairMeasurement] = []
    for policy in policies:
        env = MutationEnvironment(
            parent,
            landscape.alphabet,
            max_mutations=min(task.max_mutations, landscape.sequence_length),
            transitions=np.asarray(landscape.transition_matrix),
        )
        sampler = CMAES(env, repair=policy, seed=seed)  # type: ignore[arg-type]
        designs: Tokens = sampler.propose(n_designs)
        sampler.observe(designs, landscape.evaluate(designs))
        measurements.append(
            RepairMeasurement(
                policy=policy,
                repaired_fraction=sampler.repaired_fraction,
                n_decoded=n_designs,
                constructible=bool(env.is_constructible(designs).all()),
            )
        )
    return tuple(measurements)


# --------------------------------------------------------------------------
# Sweeping the port's findings over seeds.
# --------------------------------------------------------------------------

#: Instances drawn per shape. This project's standard, from
#: ``experiments/run_suite.py``'s ``MAIN_SEEDS``, and the same number for the
#: same reason: a rate estimated from a handful of draws is not a rate. Nothing
#: about a sweep here is a campaign -- no policy is trained and no oracle budget
#: is spent -- so the count is not set by campaign cost and there is no reason
#: for it to be smaller than the suite's.
REFERENCE_SEEDS = 100

#: Instances the *attainability audit* runs on at the L=256 shape. Everything
#: else is measured on all `REFERENCE_SEEDS` draws there too -- the audit is the
#: only expensive step, at roughly 340 seconds of beam search per instance
#: against 11 at L=32 and 15 at L=64, measured before this number was chosen.
#:
#: The precedent is ``run_suite.LARGE_SPACE_SEEDS = 30``, whose justification
#: transfers verbatim: cost differs by an order of magnitude between the small
#: shapes and L=256, not because the claims differ. What is deliberately *not*
#: reduced with it is the unreachability rate, which is the finding most likely
#: to be challenged and which costs nothing to measure -- so it is estimated from
#: the full hundred at every shape, including this one.
LARGE_SPACE_AUDIT_SEEDS = 30


@dataclass(frozen=True)
class ReferenceShape:
    """One suite task's problem shape, as holo's generator would draw it.

    A *shape*, not a task: holo has no density knob, so this package's
    ``feasibility`` (density 0.15) and ``evolvepro`` (density 0.5) tasks map onto
    the **same** reference configuration and differ only in the campaign wrapped
    round it. Both are kept, named as the tasks they stand for, so the sweep's
    rows line up with the suite's -- but their instance-level answers are two
    samples from one distribution rather than from two, and a report must not
    read them as independent shapes.

    Attributes:
        name: The suite task this stands for.
        parameters: What holo is asked for. Its ``random_seed`` is replaced per
            draw and the value carried here is ignored.
        max_mutations: Substitutions per round, from the suite task's protocol.
        rounds: Rounds, likewise.
        batch_size: Designs per round, likewise. Unused by the audit, carried so
            the task this builds is the task the suite runs.
        seeds: Instances to draw.
        audit_seeds: Instances the attainability audit runs on, or ``None`` for
            all of them. It is separate from `seeds` because the audit is the
            only expensive measurement here, and a reduced count that applied to
            everything would drag the cheap findings down with the costly one --
            including the unreachability rate, which is the one most likely to
            be challenged. A shape that reduces it carries the reduction *here*,
            beside its own name, rather than in a caller's argument where the
            justification would not travel with it.
    """

    name: str
    parameters: ReferenceParameters
    max_mutations: int
    rounds: int
    batch_size: int
    seeds: int = REFERENCE_SEEDS
    audit_seeds: int | None = None

    def at(self, seed: int) -> ReferenceParameters:
        """This shape's parameters at one draw."""
        return replace(self.parameters, random_seed=seed)

    def audits(self, seed: int) -> bool:
        """Whether this draw is one the attainability audit runs on.

        The audited draws are a **prefix** of the seeds rather than a sample of
        them, so the audited subset is the same set every time the sweep runs and
        adding seeds never re-rolls which instances carry an audit.

        Args:
            seed: The draw.

        Returns:
            Whether to audit it.
        """
        return self.audit_seeds is None or seed < self.audit_seeds


#: The suite shapes that have a reference counterpart at all. ``gb1-anchor`` and
#: ``trpb-anchor`` are empirical landscapes with no Ehrlich instance behind them,
#: so holo has nothing to draw for them.
REFERENCE_SHAPES: tuple[ReferenceShape, ...] = (
    ReferenceShape(
        name="diagnostic",
        parameters=ReferenceParameters(num_states=20, dim=32, num_motifs=2, motif_length=4),
        max_mutations=4,
        rounds=4,
        batch_size=96,
    ),
    ReferenceShape(
        name="feasibility",
        parameters=ReferenceParameters(num_states=20, dim=64, num_motifs=2, motif_length=4),
        max_mutations=4,
        rounds=4,
        batch_size=96,
    ),
    ReferenceShape(
        name="alde",
        parameters=ReferenceParameters(num_states=20, dim=64, num_motifs=2, motif_length=4),
        max_mutations=21,
        rounds=3,
        batch_size=132,
    ),
    ReferenceShape(
        name="evolvepro",
        parameters=ReferenceParameters(num_states=20, dim=64, num_motifs=2, motif_length=4),
        max_mutations=4,
        rounds=8,
        batch_size=48,
    ),
    ReferenceShape(
        name="large-space",
        parameters=ReferenceParameters(
            num_states=20, dim=256, num_motifs=4, motif_length=8, quantization=4
        ),
        max_mutations=62,
        rounds=4,
        batch_size=96,
        audit_seeds=LARGE_SPACE_AUDIT_SEEDS,
    ),
)


@dataclass(frozen=True)
class InstanceOutcome:
    """Everything one reference instance answers, in one row.

    Attributes:
        shape: The shape that drew it.
        seed: The draw.
        planted_distance: Substitutions between holo's wild type and holo's
            planted optimum.
        reachable_at_budget: Whether any legal ordering builds the planted
            optimum inside the task's own per-round budget. Almost never, and
            uninteresting on its own -- the budget is far below the distance.
        reachable_unbounded: Whether any legal ordering builds it **at all**,
            audited at ``budget = L`` so every position may move. ``False`` is
            the finding: holo's own ``optimal_solution()`` verifies at reward
            1.0, and no single-substitution path arrives at it. This is a
            property of the instance alone -- it does not depend on the campaign
            wrapped round it -- so two shapes that differ only in protocol
            answer it identically per draw.
        attainable_lower: Best value the audit's beam actually constructed, or
            ``None`` where this draw carried no audit.
        attainable_upper: The certified bound above it, or ``None`` likewise.
        pinned: Whether the audit closed the bracket to a point. ``None`` is
            **not** a bracket: it says no audit was run on this draw, and a
            summary that counted it as one would report a reduced audit count as
            a shape that fails to pin.
        reanchored_lower: Best value the **re-anchored** audit constructed --
            the chain of Hamming balls a real campaign searches, each centred on
            the last round's best design. ``None`` where this draw carried no
            audit.

            This is the other half of the headline. ``reachable_unbounded``
            being ``False`` says holo's own verified optimum has no legal
            construction order from a fixed anchor; this says what a campaign
            that *moves* its anchor recovers, which is the difference between
            an indictment of the benchmark and a statement about how it is used.
            It rested on one draw per shape until it was swept.
        reanchored_upper: The certified bound above it, or ``None`` likewise.
        reanchored_pinned: Whether the re-anchored audit closed its bracket to a
            point. ``None`` means no audit ran on this draw, which is not the
            same as a bracket that failed to close.
        repaired: ``repaired_fraction`` per policy, keyed by policy name.
        all_constructible: Whether every repaired design was buildable, for
            every policy. A repair that leaves one behind has failed at its
            only job.
    """

    shape: str
    seed: int
    planted_distance: int | None
    reachable_at_budget: bool | None
    reachable_unbounded: bool | None
    attainable_lower: float | None
    attainable_upper: float | None
    pinned: bool | None
    reanchored_lower: float | None
    reanchored_upper: float | None
    reanchored_pinned: bool | None
    repaired: dict[str, float]
    all_constructible: bool


def sweep_instance(
    shape: ReferenceShape,
    seed: int,
    *,
    interpreter: Path | None = None,
    n_designs: int = 128,
    audit: bool | None = None,
) -> InstanceOutcome:
    """Draw one reference instance and answer every per-instance question on it.

    One instance is loaded once and reused by every measurement, which is the
    whole saving: holo's constructor is the fixed cost at the small shapes, and
    the audit's beam is the fixed cost at the large one.

    Args:
        shape: What to draw.
        seed: Which draw.
        interpreter: Python that can import holo, located automatically if
            omitted.
        n_designs: Designs decoded per repair policy.
        audit: Whether to run the attainability audit on this draw, or ``None``
            to let the shape decide through
            [ReferenceShape.audits][evogfn.benchmark.holo_port.ReferenceShape.audits].
            Skipping it leaves the audit fields ``None`` rather than filling
            them with a cheaper substitute.

    Returns:
        One row of the sweep.
    """
    instance = load_reference_instance(shape.at(seed), interpreter=interpreter)
    task = reference_task(
        instance,
        name=f"holo-{shape.name}",
        max_mutations=shape.max_mutations,
        rounds=shape.rounds,
        batch_size=shape.batch_size,
        reanchor=True,
    )
    audited = shape.audits(seed) if audit is None else audit
    optimum = attainable_optimum(task) if audited else None
    # The re-anchored audit chains one beam per round, so it costs the
    # fixed-anchor audit times the round count -- which is why it rides on the
    # same `audits(seed)` gate rather than running on every draw.
    reanchored = reanchored_attainable(task) if audited else None
    repairs = repair_reference(task, n_designs=n_designs)
    return InstanceOutcome(
        shape=shape.name,
        seed=seed,
        planted_distance=planted_distance(task),
        reachable_at_budget=planted_optimum_reachable(task),
        # At the sequence length, which is the largest budget that means
        # anything: every position may be substituted once. `False` here is
        # "no legal construction order at any budget", not "not at this one".
        reachable_unbounded=planted_optimum_reachable(task, budget=instance.parameters.dim),
        attainable_lower=None if optimum is None else optimum.lower,
        attainable_upper=None if optimum is None else optimum.upper,
        pinned=None if optimum is None else optimum.is_exact,
        reanchored_lower=None if reanchored is None else reanchored.lower,
        reanchored_upper=None if reanchored is None else reanchored.upper,
        reanchored_pinned=None if reanchored is None else reanchored.is_exact,
        repaired={measurement.policy: measurement.repaired_fraction for measurement in repairs},
        all_constructible=all(measurement.constructible for measurement in repairs),
    )


def sweep_shape(
    shape: ReferenceShape,
    *,
    seeds: Sequence[int] | None = None,
    workers: int = 1,
    interpreter: Path | None = None,
) -> tuple[InstanceOutcome, ...]:
    """Repeat the per-instance measurement across draws of one shape.

    Args:
        shape: What to draw.
        seeds: Which draws, or ``None`` for ``range(shape.seeds)`` -- the count
            the shape itself declares, which is where the justification for a
            reduced count lives.
        workers: Instances measured in parallel. Every measurement is
            independent and each spends most of its time in holo's subprocess or
            in the beam, so this is close to linear. Leave at 1 for a
            reproducible profile.
        interpreter: Python that can import holo.

    Returns:
        One outcome per draw, in seed order.
    """
    draws = list(range(shape.seeds)) if seeds is None else list(seeds)
    if workers <= 1:
        return tuple(sweep_instance(shape, seed, interpreter=interpreter) for seed in draws)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(sweep_instance, shape, seed, interpreter=interpreter) for seed in draws
        ]
        return tuple(future.result() for future in futures)


@dataclass(frozen=True)
class RateEstimate:
    """A share of instances, with the interval that says how well it is known.

    Attributes:
        quantity: What is being counted, for a report to quote.
        successes: Instances the property held on.
        trials: Instances drawn.
        low: Lower end of the 95% interval.
        high: Upper end.
        sign_p: Two-sided sign-test p-value against a rate of one half, from
            [unanimity_p][evogfn.benchmark.statistics.unanimity_p]. It is the
            one statement about a count of draws that assumes no distribution,
            and it is what says whether the draw count could support the claim
            at all: below
            [unanimity_floor][evogfn.benchmark.statistics.unanimity_floor]
            instances, even a unanimous result does not reach 0.05.
    """

    quantity: str
    successes: int
    trials: int
    low: float
    high: float
    sign_p: float

    @property
    def point(self) -> float:
        """The observed share."""
        return self.successes / self.trials if self.trials else float("nan")

    def __repr__(self) -> str:
        """State the share, its interval and what it rests on."""
        return (
            f"{self.quantity}: {self.successes}/{self.trials} = {self.point:.3f} "
            f"[{self.low:.3f}, {self.high:.3f}] (sign test p = {self.sign_p:.3g})"
        )


def rate_estimate(quantity: str, successes: int, trials: int) -> RateEstimate:
    """A Wilson score interval for a share of instances.

    Wilson rather than the textbook normal interval because the shares this
    reports sit near the ends: 5 of 5 has a normal interval of exactly
    ``[1, 1]``, which asserts certainty from five draws and is the specific
    misreading the sweep exists to remove.

    The critical value comes from
    [t_critical][evogfn.benchmark.statistics.t_critical] rather than a hard-coded
    1.96, so this uses the repository's own table and is slightly conservative at
    small counts. There is no proportion helper in
    [evogfn.benchmark.statistics][] to call instead; adding one there is a
    separate change to a module in the fingerprinted closure, which this port
    does not touch.

    Args:
        quantity: What is being counted.
        successes: Instances the property held on.
        trials: Instances drawn.

    Returns:
        The estimate.

    Raises:
        ValueError: If more instances succeeded than were drawn.
    """
    if successes > trials:
        raise ValueError(f"{successes} instances cannot succeed out of {trials}")
    if trials == 0:
        return RateEstimate(quantity, 0, 0, float("nan"), float("nan"), 1.0)
    z = t_critical(max(trials - 1, 1))
    share = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (share + z**2 / (2 * trials)) / denominator
    half = z * math.sqrt(share * (1.0 - share) / trials + z**2 / (4 * trials**2)) / denominator
    return RateEstimate(
        quantity=quantity,
        successes=successes,
        trials=trials,
        low=max(0.0, centre - half),
        high=min(1.0, centre + half),
        sign_p=unanimity_p(trials, successes),
    )


@dataclass(frozen=True)
class MeanEstimate:
    """A mean over instances, with the interval that says how well it is known.

    Attributes:
        quantity: What was averaged.
        n: Instances drawn.
        mean: The mean.
        low: Lower end of the 95% Student-t interval.
        high: Upper end.
        minimum: Smallest value observed, which is what a claim of the form
            "at least x" has to be read against.
        maximum: Largest.
    """

    quantity: str
    n: int
    mean: float
    low: float
    high: float
    minimum: float
    maximum: float

    def __repr__(self) -> str:
        """State the mean, its interval and its range."""
        return (
            f"{self.quantity}: {self.mean:.3f} [{self.low:.3f}, {self.high:.3f}] "
            f"over {self.n} instances, range [{self.minimum:.3f}, {self.maximum:.3f}]"
        )


def mean_estimate(quantity: str, values: Iterable[float]) -> MeanEstimate:
    """A Student-t interval for a mean over instances.

    Args:
        quantity: What is being averaged.
        values: One value per instance.

    Returns:
        The estimate. A single value carries an infinite interval rather than a
        zero-width one, which is the honest reading of one draw.
    """
    sample = np.asarray(list(values), dtype=np.float64)
    n = int(sample.size)
    if n == 0:
        nan = float("nan")
        return MeanEstimate(quantity, 0, nan, nan, nan, nan, nan)
    mean = float(sample.mean())
    if n == 1:
        return MeanEstimate(quantity, 1, mean, -float("inf"), float("inf"), mean, mean)
    half = t_critical(n - 1) * float(sample.std(ddof=1)) / math.sqrt(n)
    return MeanEstimate(
        quantity=quantity,
        n=n,
        mean=mean,
        low=mean - half,
        high=mean + half,
        minimum=float(sample.min()),
        maximum=float(sample.max()),
    )


@dataclass(frozen=True)
class ShapeSummary:
    """One shape's swept answers.

    Attributes:
        shape: The shape.
        instances: Draws behind the cheap figures -- the unreachability rate and
            both repaired fractions.
        audited: Draws behind `pinned` and `attainable`. Equal to `instances`
            except where the audit's cost forced a reduction, and carried
            separately so a report cannot quote one count over both.
        unreachable: Share of draws whose planted optimum has no legal
            construction order at any budget.
        pinned: Share of *audited* draws whose attainability audit closed its
            bracket to a point.
        attainable: The audited lower bound, averaged over audited draws. Read
            beside `pinned`: where the bracket did not close this is the searched
            value and not the optimum.
        reanchored: The **re-anchored** audit's lower bound, averaged over
            audited draws. The other half of the headline: `unreachable` says
            the planted optimum has no construction order from a fixed anchor,
            and this says what a campaign that moves its anchor recovers. Quoted
            without it, the unreachability rate reads as a defect in the
            benchmark rather than as a property of how the benchmark is used.
        reanchored_pinned: Share of audited draws whose re-anchored audit closed
            its bracket, so the mean above is read as a value and not a floor.
        repaired: ``repaired_fraction`` per policy.
        constructible_everywhere: Whether every repair returned buildable
            designs on every draw.
    """

    shape: str
    instances: int
    audited: int
    unreachable: RateEstimate
    pinned: RateEstimate
    attainable: MeanEstimate
    reanchored: MeanEstimate
    reanchored_pinned: RateEstimate
    repaired: dict[str, MeanEstimate]
    constructible_everywhere: bool

    def __repr__(self) -> str:
        """One line per swept quantity."""
        audited = "" if self.audited == self.instances else f", {self.audited} audited"
        lines = [
            f"{self.shape} ({self.instances} instances{audited})",
            f"  {self.unreachable!r}",
            f"  {self.pinned!r}",
            f"  {self.attainable!r}",
            # Printed directly under the fixed-anchor pair, because the finding
            # is the *difference* between them and a reader who has to fetch the
            # second number from elsewhere will quote the first alone.
            f"  {self.reanchored!r}",
            f"  {self.reanchored_pinned!r}",
            *(f"  {estimate!r}" for estimate in self.repaired.values()),
        ]
        return "\n".join(lines)


def summarise(shape: str, outcomes: Sequence[InstanceOutcome]) -> ShapeSummary:
    """Turn one shape's rows into rates and means with intervals.

    Args:
        shape: The shape's name.
        outcomes: Its rows.

    Returns:
        The summary.

    Raises:
        ValueError: If no instance was drawn, since every figure below would be
            a fabrication rather than an estimate.
    """
    if not outcomes:
        raise ValueError(f"{shape}: nothing to summarise; no instance was drawn")
    trials = len(outcomes)
    # The audited subset, and *only* it, is what the pin rate is over. A draw
    # that carried no audit is not a draw whose bracket failed to close.
    audited = [outcome for outcome in outcomes if outcome.pinned is not None]
    policies = sorted({policy for outcome in outcomes for policy in outcome.repaired})
    return ShapeSummary(
        shape=shape,
        instances=trials,
        audited=len(audited),
        unreachable=rate_estimate(
            "planted optimum unconstructible at any budget",
            # `is False` and not `not`: `None` means the landscape reported no
            # planted optimum to audit, which is not the same finding as one
            # that cannot be built.
            sum(1 for outcome in outcomes if outcome.reachable_unbounded is False),
            trials,
        ),
        pinned=rate_estimate(
            "attainability audit pinned exactly",
            sum(1 for outcome in audited if outcome.pinned),
            len(audited),
        ),
        attainable=mean_estimate(
            "attainable lower bound",
            [
                outcome.attainable_lower
                for outcome in audited
                if outcome.attainable_lower is not None
            ],
        ),
        repaired={
            policy: mean_estimate(
                f"repaired_fraction ({policy})",
                [outcome.repaired[policy] for outcome in outcomes if policy in outcome.repaired],
            )
            for policy in policies
        },
        reanchored=mean_estimate(
            "re-anchored attainable lower bound",
            [
                outcome.reanchored_lower
                for outcome in audited
                if outcome.reanchored_lower is not None
            ],
        ),
        reanchored_pinned=rate_estimate(
            "re-anchored audit pinned exactly",
            sum(1 for outcome in audited if outcome.reanchored_pinned),
            len(audited),
        ),
        constructible_everywhere=all(outcome.all_constructible for outcome in outcomes),
    )


def sweep_report(summaries: Sequence[ShapeSummary]) -> str:
    """Every shape's summary, plus what the draw count itself permits.

    The trailing line is not decoration. `unanimity_floor` is the fewest
    instances at which a unanimous result reaches significance at all, and the
    port's original five draws sat below it: "4 of 5" could not have been
    evidence whatever the four said.

    Args:
        summaries: One per shape.

    Returns:
        A printable report.
    """
    floor = unanimity_floor()
    body = "\n".join(repr(summary) for summary in summaries)
    return f"{body}\n\nfewest instances a unanimous result could be significant at: {floor}"
