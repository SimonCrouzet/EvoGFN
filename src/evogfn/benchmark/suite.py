"""The suite: main tests that carry claims, diagnostics that inform them.

Two tiers, because they answer different kinds of question and should not be
read the same way.

**Main tests** are the headline. Each is a landscape and a protocol chosen so
that a result on it means something a lab would recognise, and each is sized so
the claim survives the strongest control available. These go in the paper's main
table.

**Diagnostics** vary one axis on a fixed, cheap landscape. They decide things --
which objective to carry into the main table, whether the ranking survives a
change of budget, whether rounds matter at fixed total. They are how choices get
made, not what gets claimed.

Sequence lengths follow published practice rather than convenience
------------------------------------------------------------------

Stanton et al.'s own base configuration is ``L = 256, c = 4, k = 8, q = 4``, and
they sweep around it. HDBO uses ``L = 5, 15, 64``, and reports two published
Bayesian-optimisation methods running out of memory at 64. holo-bench ships
``dim = 7`` defaults for quick enumerable use. So:

* the flagship large-space task uses **Stanton's base configuration**, which
  makes our numbers directly comparable to the benchmark's authors;
* the mid-size tasks use **L = 64**, the setting where the published field
  degrades;
* diagnostics use **L = 32**, cheap enough to sweep an axis at 50 seeds.

The search radius is per round, and the anchor moves
----------------------------------------------------

A single fixed Hamming ball makes an Ehrlich task unwinnable by construction.
Reaching reward 1.0 on an Ehrlich instance means placing every residue of every
motif, and the parent is drawn independently of the planted optimum, so the two
differ in roughly ``L * (1 - 1/v)`` positions -- far outside a ball of the
radius a round of mutagenesis buys. Against such a ball the answer is not merely
hard to find, it is absent, and a regret reported against it is mostly a
constant that says nothing about any method.

The fix is not a wider radius. Real directed evolution keeps the radius small --
four or five substitutions is what a round of site-saturation mutagenesis buys --
and moves the *anchor*: round two starts from the best variant round one
produced. Distance from the wild type then accumulates while the per-round
budget does not. So every Ehrlich task here re-anchors, and its radius is the
smallest one the audit put its own optimum inside the campaign's reach at.

Three tasks are not like the others, and all three for stated reasons. The two
empirical anchors, ``gb1-anchor`` and ``trpb-anchor``, each have four measured
sites and a budget of four, so each ball is that landscape's whole space and
there is nothing for an anchor to move towards. That is a property to be checked
per landscape and never inherited: it holds for both of these because both
libraries vary four positions and nothing else, and it would fail the moment a
task's sites outnumbered its budget -- at which point the task must re-anchor and
its bound becomes a bracket. ``feasibility`` keeps a fixed anchor because what
binds there is the transition matrix, not the radius. Leaving the anchor still is
what keeps its attainable optimum an *enumerated* answer rather than a bracket,
which is the difference between reporting a fact and reporting a search.

The round-varying tasks are the ones this matters most for. Comparing 3x132
against 8x48 asks whether many small rounds beat few large ones, and with a
fixed anchor neither shape can move at all -- so the comparison would be between
two identically stranded campaigns.

Regret is against what is attainable, not against the nominal optimum
---------------------------------------------------------------------

Even with the anchor moving, the reachable set is not the landscape. Each task
therefore declares what [evogfn.benchmark.attainable][] audited it to contain,
and `run_task` stores regret against *that*. The declarations are constants
rather than computations because the audit costs minutes per task, and
``tests/benchmark/test_suite_tasks.py`` re-derives them rather than trusting
them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
from evogfn.benchmark.determinism import is_deterministic
from evogfn.benchmark.protocol import PLATE, Protocol, round_sweep
from evogfn.benchmark.tasks import Attainable, Task
from evogfn.env.feasibility import (
    BudgetBandPredicate,
    CodebookPredicate,
    ConjunctionPredicate,
    ContactPredicate,
)
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.gb1 import GB1Landscape
from evogfn.landscapes.trpb import TRPB_POSITIONS, TrpBLandscape
from evogfn.metrics.diversity import diversity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from evogfn.algorithms.base import Sampler
    from evogfn.benchmark.attainable import AttainableOptimum
    from evogfn.benchmark.methods import Methodology
    from evogfn.benchmark.store import ResultStore, RunRecord
    from evogfn.env.feasibility import FeasibilityPredicate
    from evogfn.env.mutation import MutationEnvironment
    from evogfn.landscapes.base import FitnessLandscape
    from evogfn.loop.campaign import Campaign
    from evogfn.loop.ledger import CampaignResult, RoundRecord

#: GB1's four measured sites. Equal to the sequence length, so every variant in
#: the published table is reachable and the anchor exercises no search radius.
GB1_MUTATIONS = 4

#: TrpB's four assayed active-site positions, and the per-round radius on
#: ``trpb-anchor``. Taken from the landscape rather than written as ``4`` so that
#: the claim the task rests on -- that the radius *equals* the number of sites,
#: hence the ball is the whole 160,000-sequence library -- is structural rather
#: than two literals happening to agree.
TRPB_MUTATIONS = len(TRPB_POSITIONS)

#: Per-round radius on the flagship task, at L=256. Four re-anchored rounds of
#: this reach 248 substitutions -- the distance to the planted optimum -- while
#: any single round still searches a ball a lab would recognise.
LARGE_SPACE_MUTATIONS = 62

#: Per-round radius on the feasibility task at L=64, and the whole radius: this
#: task keeps a fixed anchor. A sparse transition matrix makes only a small
#: fraction of the Hamming ball at this radius constructible, and that ratio is
#: what the task exists to measure. Widening it would measure something else.
FEASIBILITY_MUTATIONS = 4

#: Per-round radius for ALDE's three rounds. Wide enough that a re-anchored
#: chain reaches this task's own optimum in the fewest rounds any protocol here
#: runs, which a round of mutagenesis on its own does not buy.
ALDE_MUTATIONS = 21

#: Per-round radius for EVOLVEpro's eight rounds. Eight rounds accumulate from a
#: radius a lab would recognise the reach that three rounds have to buy in one
#: go, which is the whole content of the protocol comparison: a shape with more
#: rounds buys reach at the same per-round cost.
EVOLVEPRO_MUTATIONS = 4

#: Per-round radius on the shared diagnostic landscape at L=32. The original
#: shared constant, kept because re-anchoring already makes it sufficient over
#: the rounds a diagnostic runs.
DIAGNOSTIC_MUTATIONS = 4

#: A task whose per-round radius is deliberately smaller than the distance to
#: its own planted optimum has to say so, in these words, so
#: ``experiments/audit_optima.py`` reports the gap as a property of the design
#: rather than as a defect in it. Every Ehrlich task here is in that position by
#: intent: the radius is a round of mutagenesis and the reach is cumulative.
CAPPED = (
    "The per-round mutation budget is deliberately capped below the distance to "
    "the planted optimum."
)

#: What the capped radius buys back, on a task whose anchor moves.
CUMULATIVE = f"{CAPPED} The campaign re-anchors, so its reach is cumulative."

#: Where a campaign's code actually starts, for staleness. Everything a run
#: touches is reachable from these two: the methodology table pulls in every
#: sampler, surrogate, acquisition rule and landscape, and the loop pulls in the
#: ledger and its metrics. Declaring them rather than hashing the whole package
#: tree is what stops an unrelated addition -- a new Pareto indicator, say --
#: invalidating a genetic-algorithm result it cannot possibly have influenced.
#: What a stored campaign's result can depend on.
#:
#: The rule this encodes: a result is invalid if it would change under the
#: current code, and valid if it would not. Hashing the import closure is a
#: conservative approximation of that -- it cannot know whether an edit alters
#: an outcome without re-running, so it assumes the worst. Where a change
#: provably cannot alter a completed run,
#: [bless][evogfn.benchmark.store.ResultStore.bless] is the escape hatch, and it
#: requires naming the modules being vouched for precisely because a blanket
#: restamp is dozens of independent assertions made in one call.
#:
#: ``benchmark.methods`` and ``loop.campaign`` cover how a campaign is built and
#: run. The other two are here because neither is reachable by import from those
#: two, and both decide what a run is:
#:
#: * ``benchmark.suite`` holds the task definitions, the per-task mutation
#:   budgets, and the line that writes ``proxy_calls`` into the record. Editing
#:   it changes what runs, or what gets recorded, and neither is reachable by
#:   import from the two entry points above.
#: * ``benchmark.selection`` builds the swept arms and holds their
#:   hyperparameters, so an edit there changes what a swept arm *is*.
#:
#: Declaring them rather than hashing the whole package tree is still what stops
#: an unrelated addition -- a new Pareto indicator, say -- invalidating a
#: genetic-algorithm result it cannot possibly have influenced.
RESULT_DEPENDENCIES = (
    "evogfn.benchmark.methods",
    "evogfn.benchmark.selection",
    "evogfn.benchmark.suite",
    "evogfn.loop.campaign",
)


def _ehrlich(**kwargs: object) -> Callable[[], FitnessLandscape]:
    """A factory for an Ehrlich instance with fixed parameters."""

    def build() -> FitnessLandscape:
        return EhrlichLandscape(**kwargs)  # type: ignore[arg-type]

    return build


def _task(  # noqa: PLR0913 - a task is defined by what it declares
    name: str,
    purpose: str,
    build: Callable[[], FitnessLandscape],
    protocol: Protocol,
    *,
    reanchor: bool,
    attainable: Attainable | None,
) -> Task:
    """A task whose search radius, anchor rule and reachable optimum are all stated.

    Taking the radius from the protocol rather than accepting it separately is
    what stops the two disagreeing. They are read by different code -- the
    environment uses the task's, `Protocol.constrains_search` reports the
    protocol's -- and a task that searched at one radius while reporting another
    would be undetectable from a stored record.

    ``reanchor`` and ``attainable`` are keyword-only and have **no defaults**,
    which is the enforcement this module exists to apply. A task that searches
    one fixed ball, or whose regret is taken against a nominal optimum, is
    making a decision that has to be visible in the suite's definition, and a
    default would let the next task added inherit it in silence.

    Args:
        name: Short identifier.
        purpose: What this task decides that the others cannot.
        build: Makes the landscape.
        protocol: Rounds, batch size and the per-round mutation budget.
        reanchor: Whether the anchor follows the best design measured so far.
        attainable: What an audit found this task's search space to contain, or
            ``None`` where no audit covers it -- which stores no regret at all.
            Statable, never defaulted: a diagnostic that varies the *shape of
            the reachable set* has no audited target until somebody audits each
            rung, and declaring a neighbouring task's number would put a floor
            no method could clear into every regret on it.

    Returns:
        The task.

    Raises:
        ValueError: If the protocol states no mutation budget. A task without
            one searches the whole space, which is a decision worth writing
            down rather than defaulting into.
    """
    if protocol.max_mutations is None:
        raise ValueError(f"{name}: protocol must state a mutation budget")
    return Task(
        name=name,
        purpose=purpose,
        build=build,
        protocol=protocol,
        max_mutations=protocol.max_mutations,
        reanchor=reanchor,
        attainable=attainable,
    )


#: Stanton et al.'s base configuration, so our numbers sit beside theirs.
STANTON_BASE = _ehrlich(
    sequence_length=256,
    vocab_size=20,
    n_motifs=4,
    motif_length=8,
    quantization=4,
    transition_density=0.5,
    seed=0,
)

MAIN: tuple[Task, ...] = (
    _task(
        "gb1-anchor",
        "Do the numbers hold on real measurements? The empirical anchor, and "
        "the easiest geometry here: four sites, no feasibility constraint, and "
        "a mutation budget that reaches every sequence.",
        GB1Landscape,
        Protocol(rounds=4, batch_size=PLATE, max_mutations=GB1_MUTATIONS, label="four plates"),
        # Nothing to move towards: the ball of radius four over four sites is
        # the entire published table, so the first round already sees every
        # design a later one could be anchored at.
        reanchor=False,
        attainable=Attainable.whole_optimum(
            "exact: four mutations over four sites reach every measured variant"
        ),
    ),
    _task(
        "trpb-anchor",
        "Does the empirical result survive a second protein? Johnston et al.'s "
        "four active-site positions of tryptophan synthase, a different assay "
        "and a different reward geometry from GB1's -- and a harder one: 99.3% "
        "of the 159,129 measured variants score below the wild type, against "
        "90% for GB1, so there is far less gradient to climb. Same geometry as "
        "gb1-anchor: four sites, no feasibility constraint, and a mutation "
        "budget that reaches every sequence, so the bound below is an "
        "enumeration rather than a search bracket.",
        TrpBLandscape,
        Protocol(rounds=4, batch_size=PLATE, max_mutations=TRPB_MUTATIONS, label="four plates"),
        # GB1's special property, checked for this landscape rather than
        # inherited from it: the four assayed positions *are* the sequence
        # length, so the ball of radius four around ``VFVS`` is the entire
        # 160,000-sequence library and there is nothing an anchor could move
        # towards. One mutation per position is the environment's rule and it
        # binds nothing here, since four positions admit four mutations.
        reanchor=False,
        # Audited for this landscape and not borrowed from ``gb1-anchor``:
        # ``attainable_optimum`` enumerated all 160,000 reachable terminal states
        # from ``VFVS`` at four mutations and found the maximum to be the
        # landscape's own optimum, 2.4505 at ``AIKG`` -- which, like ``FWAA``,
        # differs from the wild type at all four positions and so is exactly the
        # design a narrower budget would have put out of reach. The regret floor
        # is therefore genuinely zero, and ``whole_optimum`` is the declaration
        # that says so by deferring to the landscape rather than by restating a
        # number that would silently stop matching if the dataset were repinned.
        attainable=Attainable.whole_optimum(
            "exact: enumerated all 160,000 reachable terminal states at 4 mutations "
            "over 4 sites, which is the whole TrpB library"
        ),
    ),
    _task(
        "large-space",
        "Can the method search a space it cannot enumerate? Stanton et al.'s "
        "base configuration, L=256 with four motifs of length eight. The budget "
        "is 384 designs against a reachable set with no useful upper digit -- "
        "62 substitutions a round of 256 positions over an alphabet of 20. "
        f"{CUMULATIVE}",
        STANTON_BASE,
        Protocol(
            rounds=4, batch_size=PLATE, max_mutations=LARGE_SPACE_MUTATIONS, label="four plates"
        ),
        reanchor=True,
        # The one task whose bracket the audit cannot close, and the reason an
        # interval is carried rather than a point: the lower end is witnessed by
        # a design the beam actually built, the upper is what the reward's
        # structure permits at 248 cumulative substitutions, and nothing settles
        # which. A comparison here is read against the interval.
        attainable=Attainable.between(
            0.2812,
            1.0,
            "bounded: 4 re-anchored rounds of 62, beam search below the budget-split bound",
        ),
    ),
    _task(
        "feasibility",
        "Can the method stay inside the constructible set? A sparse transition "
        "matrix makes most sequences unbuildable, so rejection sampling spends "
        "the budget on designs that cannot be made while masking cannot. "
        f"{CAPPED} The anchor is held fixed: what binds here is the transition "
        "matrix rather than the radius, so moving outward would change what the "
        "task measures.",
        _ehrlich(
            sequence_length=64,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.15,
            seed=1,
        ),
        Protocol(
            rounds=4, batch_size=PLATE, max_mutations=FEASIBILITY_MUTATIONS, label="four plates"
        ),
        reanchor=False,
        # Enumerated, not searched: the constructible subset of this ball can be
        # listed exhaustively, so the maximum over the reachable set is a
        # measurement rather than the best a search happened to reach.
        attainable=Attainable.exactly(
            0.375, "exact: enumerated the 26,580 reachable terminal states at 4 mutations"
        ),
    ),
    _task(
        "protocol-alde",
        "Does the ranking survive the shape a real campaign takes? Three rounds "
        "of 132, after ALDE's six 96-well plates over three rounds. "
        f"{CUMULATIVE} Three rounds is the fewest here, so it needs the widest "
        "radius to put its own optimum inside reach.",
        _ehrlich(
            sequence_length=64,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.5,
            seed=2,
        ),
        Protocol(rounds=3, batch_size=132, max_mutations=ALDE_MUTATIONS, label="ALDE"),
        reanchor=True,
        attainable=Attainable.exactly(
            1.0, "pinned: 3 re-anchored rounds of 21 reach the budget-split bound"
        ),
    ),
    _task(
        "protocol-evolvepro",
        "The opposite shape at a comparable budget: eight rounds of 48, after "
        "EVOLVEpro. Many small rounds against few large ones, on the same "
        "landscape as protocol-alde so only the shape differs. "
        f"{CUMULATIVE} Eight rounds accumulate from a narrow per-round radius "
        "the reach three rounds have to buy in one wide one, which is the point "
        "of running both.",
        _ehrlich(
            sequence_length=64,
            vocab_size=20,
            n_motifs=2,
            motif_length=4,
            transition_density=0.5,
            seed=2,
        ),
        Protocol(
            rounds=8, batch_size=48, max_mutations=EVOLVEPRO_MUTATIONS, label="EVOLVEpro-like"
        ),
        reanchor=True,
        attainable=Attainable.exactly(
            1.0, "pinned: 8 re-anchored rounds of 4 reach the budget-split bound"
        ),
    ),
)

#: Fraction of token pairs the shared diagnostic instance permits as adjacent.
#: Named because a family that sweeps this axis has to be able to say which of
#: its rungs is the instance every other diagnostic already runs on -- and then
#: reuse that task instead of defining a twin of it.
DIAGNOSTIC_DENSITY = 0.5

#: The shared diagnostic instance, as parameters rather than as a built factory.
#: A sweep over one of them rebuilds this mapping with that one replaced, so
#: every rung is provably the same instance in every other respect. Written out
#: a second time by hand it would drift on the first edit to either copy, and a
#: sweep whose rungs differ in a parameter nobody meant to vary measures
#: something other than its own axis.
DIAGNOSTIC_INSTANCE: dict[str, object] = {
    "sequence_length": 32,
    "vocab_size": 20,
    "n_motifs": 2,
    "motif_length": 4,
    "transition_density": DIAGNOSTIC_DENSITY,
    "seed": 7,
}

#: The cheap landscape every diagnostic varies an axis on.
DIAGNOSTIC_LANDSCAPE = _ehrlich(**DIAGNOSTIC_INSTANCE)

#: The protocol a diagnostic runs at when it is not the protocol being varied.
#: One frozen instance shared by every such task, so that two tasks meant to
#: differ in one axis cannot differ in the budget as well -- and so that a task
#: which *is* configurationally identical to another is identical by
#: construction rather than by two literals happening to agree.
DIAGNOSTIC_PROTOCOL = Protocol(rounds=4, batch_size=PLATE, max_mutations=DIAGNOSTIC_MUTATIONS)

#: What every diagnostic can reach. One landscape and one per-round radius, so
#: one audited answer -- taken at the fewest rounds any diagnostic runs, and
#: therefore valid for all of them, since rounds only add reach.
DIAGNOSTIC_ATTAINABLE = Attainable.exactly(
    1.0, "pinned: 4 re-anchored rounds of 4 reach the budget-split bound at L=32"
)


#: Landscape draws the protocol comparison is replicated across.
#:
#: The two protocol tasks deliberately share one instance so that only campaign
#: shape differs between them, which is what makes that comparison clean -- and
#: which leaves every constrained result in the suite resting on a single draw
#: from the generator. A hundred seeds vary the wild type and the surrogate's
#: initialisation; not one of them varies the landscape, so "does this ordering
#: hold on a different instance" is a question the headline tables cannot answer
#: at any seed count.
#:
#: Three further draws, chosen as small odd integers distinct from the shared
#: instance's seed and fixed before anything was run. Three is enough to see an
#: ordering break and too few to characterise a distribution over instances,
#: which is the honest scope: this replicates a result, it does not estimate how
#: it varies.
#:
#: **Three was chosen a priori and has not been shown to be enough.**
#: ``experiments/variance_pilot.py`` is what measures whether it is: it draws
#: more instances at fewer seeds, separates the between-instance variance from
#: the within-instance one, and applies a rule declared before the numbers.
#: Until that has run this tuple is an assumption, and the report says so rather
#: than letting the count pass for a design. Note that three draws cannot clear
#: the sign-test floor at any seed count --
#: [unanimity_floor][evogfn.benchmark.statistics.unanimity_floor] -- so the one
#: thing already known about this value is that it is too small.
REPLICATION_SEEDS: tuple[int, ...] = (3, 5, 7)

#: How a replicate task's name is built, and the only place it is built. Read
#: back by `replicate_instance`, so a rename cannot leave a parser matching the
#: old shape -- which would silently report every replicate as belonging to no
#: instance family, and a per-instance analysis over no families is a pooled one.
_REPLICATE_NAME = "replicate-{shape}-i{draw}"


def replicate_instance(task: str) -> tuple[str, int] | None:
    """Split a replicate task's name back into its protocol shape and its draw.

    The inverse of the one format string `replication` builds names with, kept
    beside it for that reason. Every reader that has to treat instance draws as
    the unit of replication needs to know which tasks are draws *of the same
    thing*, and a reader that re-derives it from a literal of its own is a
    reader that stops agreeing with the suite the first time a name moves.

    Args:
        task: A task name.

    Returns:
        The ``(shape, draw)`` pair, or ``None`` when the name is not a
        replicate's -- which is a real answer and not a failure: most tasks are
        not replicates.
    """
    prefix, _, rest = _REPLICATE_NAME.partition("{shape}")
    if not task.startswith(prefix):
        return None
    middle, _, _ = rest.partition("{draw}")
    shape, separator, draw = task[len(prefix) :].rpartition(middle)
    if not separator or not shape or not draw.isdigit():
        return None
    return shape, int(draw)


def replication(draws: Sequence[int] = REPLICATION_SEEDS) -> tuple[Task, ...]:
    """The two protocol shapes, on landscape draws other than the shared one.

    Everything except the draw is held at the headline tasks' own settings --
    same length, alphabet, motif structure, density, protocol and anchor rule --
    so a difference between a replicate and its headline task is a difference
    between instances and nothing else.

    Args:
        draws: Generator seeds to replicate across. Defaults to
            `REPLICATION_SEEDS`, which is what the tier runs. A parameter rather
            than a constant read inside, because the variance pilot has to build
            tasks at *other* draws and a second builder written for it would
            differ from this one in whatever the next edit here touched -- which
            would leave the pilot sizing a design for tasks the tier does not
            run.

    Returns:
        One task per (shape, draw), in a fixed order.

    Raises:
        ValueError: If a draw repeats, which would file two campaigns of the
            same instance under one store key and read as two draws agreeing.
    """
    if len(set(draws)) != len(tuple(draws)):
        raise ValueError(f"landscape draws must be distinct, got {list(draws)}")
    shapes = (
        ("alde", Protocol(rounds=3, batch_size=132, max_mutations=ALDE_MUTATIONS, label="ALDE")),
        (
            "evolvepro",
            Protocol(
                rounds=8, batch_size=48, max_mutations=EVOLVEPRO_MUTATIONS, label="EVOLVEpro-like"
            ),
        ),
    )
    return tuple(
        _task(
            _REPLICATE_NAME.format(shape=shape, draw=seed),
            f"Does the {shape} ordering survive a different landscape draw? Identical "
            f"to its headline task in every respect but the generator's seed.",
            _ehrlich(
                sequence_length=64,
                vocab_size=20,
                n_motifs=2,
                motif_length=4,
                transition_density=0.5,
                seed=seed,
            ),
            protocol,
            reanchor=True,
            # The same declaration the headline task carries, and it transfers
            # for a stated reason rather than by resemblance: the bound is an
            # argument about the *budget* -- that this many re-anchored rounds
            # at this radius reach the budget-split bound -- and the budget,
            # length, radius and round count are all held fixed here. The draw
            # decides where the optimum sits, not whether the protocol can walk
            # far enough to reach one.
            attainable=Attainable.exactly(
                1.0,
                f"pinned: {protocol.rounds} re-anchored rounds of {protocol.max_mutations} "
                f"reach the budget-split bound; the bound is a property of the budget, "
                f"which this replicate holds at its headline task's value",
            ),
        )
        for shape, protocol in shapes
        for seed in draws
    )


def budget_gradient() -> tuple[Task, ...]:
    """Tasks spanning the wet-lab regime to the machine-learning convention.

    The budget axis as a curve rather than an assertion: if the ranking of
    methods flips somewhere between 96 assays and 10,000, that location is the
    finding.

    Returns:
        One task per budget, on the shared diagnostic landscape.
    """
    return tuple(
        _task(
            f"budget-{rounds * batch}",
            f"Budget gradient at {rounds * batch} calls. {CUMULATIVE}",
            DIAGNOSTIC_LANDSCAPE,
            Protocol(rounds=rounds, batch_size=batch, max_mutations=DIAGNOSTIC_MUTATIONS),
            reanchor=True,
            attainable=DIAGNOSTIC_ATTAINABLE,
        )
        for rounds, batch in ((8, 12), (4, PLATE), (10, 100), (10, 1000))
    )


def rounds_curve(budget: int = 384) -> tuple[Task, ...]:
    """Tasks splitting one budget across different numbers of rounds.

    The diagnostic re-anchoring matters most for: with the anchor fixed, every
    shape in this sweep searches the identical ball, so the curve would be flat
    by construction and about something other than rounds.

    Args:
        budget: Total oracle calls to hold fixed.

    Returns:
        One task per split, on the shared diagnostic landscape.
    """
    return tuple(
        _task(
            f"rounds-{protocol.rounds}x{protocol.batch_size}",
            f"Response curve: {protocol.rounds} rounds of {protocol.batch_size}. {CUMULATIVE}",
            DIAGNOSTIC_LANDSCAPE,
            # `round_sweep` varies the shape and says nothing about the search
            # radius, so the diagnostic's own is filled in here rather than
            # letting the sweep decide an axis it is not varying.
            replace(protocol, max_mutations=DIAGNOSTIC_MUTATIONS),
            reanchor=True,
            attainable=DIAGNOSTIC_ATTAINABLE,
        )
        for protocol in round_sweep(budget)
    )


def objective_task() -> Task:
    """The single task the GFlowNet objectives are compared on."""
    return _task(
        "objectives",
        "Which training objective, at equal budget? GFlowNet-only, since a "
        f"classical baseline has no objective to vary. {CUMULATIVE}",
        DIAGNOSTIC_LANDSCAPE,
        DIAGNOSTIC_PROTOCOL,
        reanchor=True,
        attainable=DIAGNOSTIC_ATTAINABLE,
    )


#: Transition densities the constructibility diagnostic sweeps, from the
#: sparsest instance that still discriminates up to no constraint at all. What
#: each rung is for:
#:
#: * **1.0** -- every adjacency permitted, so the feasible set is the whole
#:   space and nothing bred can fail to be constructible. It is the axis's
#:   origin and its own control: a non-zero unconstructible share here is a
#:   fault in the measurement rather than a property of any landscape, and
#:   without it a small share at 0.5 cannot be told from a small bug.
#: * **`DIAGNOSTIC_DENSITY`** -- the instance every other diagnostic runs on, so
#:   the curve passes through the configuration the rest of the diagnostics are
#:   read at. This rung is `objective_task` itself, not a copy of it.
#: * **0.25** -- a rung between the diagnostic and the headline setting, so
#:   those two are not adjacent points with nothing between them to say whether
#:   the axis bends.
#: * **0.15** -- the density the headline ``feasibility`` task runs at. Putting
#:   the curve through it is what lets a share measured on that task be read
#:   against a curve rather than as an isolated number.
#: * **0.05** -- about as sparse as this vocabulary can be while the constraint
#:   still discriminates between designs. The instance is built around a
#:   Hamiltonian cycle so that every token keeps a successor; as the density
#:   approaches zero that cycle becomes the whole of the permitted adjacency,
#:   and feasibility stops being a property of the design and becomes a property
#:   of its first token.
CONSTRAINT_DENSITIES: tuple[float, ...] = (0.05, 0.15, 0.25, DIAGNOSTIC_DENSITY, 1.0)


def constraint_density() -> tuple[Task, ...]:
    """Tasks varying how much of the sequence space is constructible at all.

    The axis the feasible/reachable distinction needs and does not have. A
    design can satisfy the transition constraint and sit inside the mutation
    budget and still have no construction order in which every intermediate is
    feasible, so masking can only build the reachable part of the feasible set.
    ``unconstructible_fraction`` measures that gap on designs a run actually
    produced -- but it is currently collected on tasks designed to measure
    something else, which makes it telemetry rather than a result: one number,
    at one density, with nothing to read it against.

    Varying the density is what turns it into a curve. Everything else about
    the instance is held at `DIAGNOSTIC_INSTANCE` and the protocol is
    `DIAGNOSTIC_PROTOCOL`, so a share that moves across the family moves with
    the constraint and with nothing else.

    Only an arm that *breeds* has a share to report: the quantity counts
    offspring a genetic teacher produced and the policy could not construct, so
    an arm with no teacher stores a share of nothing, which is zero. Run this
    family with a Genetic-GFN arm in it or the whole curve is a column of
    zeros, and a column of zeros reads as "no gap" rather than as "nothing
    bred".

    Returns:
        One task per entry in `CONSTRAINT_DENSITIES`, in that order. The rung at
        `DIAGNOSTIC_DENSITY` **is** `objective_task`, returned rather than
        copied: the store keys on ``(task, arm)``, so a renamed twin of a task
        already defined pays for the same campaigns twice and files the two
        under keys nothing can compare. The names are therefore not uniform
        across the family, which is the intended shape and not an oversight.

        Every other rung declares no attainable optimum. Lowering the density
        shrinks the reachable set, so the audited value at
        `DIAGNOSTIC_DENSITY` is not this task's, and carrying it across would
        put an unclearable floor into every regret on the sparser rungs. This
        family is read on the constructibility columns, not on regret.
    """
    return tuple(
        objective_task()
        if density == DIAGNOSTIC_DENSITY
        else _task(
            f"density-{density:g}",
            f"How much of the feasible set is reachable at transition density "
            f"{density:g}? The shared diagnostic instance in every parameter but "
            f"that one, so the unconstructible share moves with the constraint "
            f"alone. {CUMULATIVE}",
            _ehrlich(**{**DIAGNOSTIC_INSTANCE, "transition_density": density}),
            DIAGNOSTIC_PROTOCOL,
            reanchor=True,
            attainable=None,
        )
        for density in CONSTRAINT_DENSITIES
    )


#: The support study's instance. Two properties are load-bearing and neither is
#: a tuning choice.
#:
#: **Enumerable.** At ten positions over four tokens the Hamming ball is
#: 1,048,576, inside `MAX_ENUMERABLE_SIZE`, so ``|feasible|`` and ``|reachable|``
#: are both *exact* rather than estimated. That is the whole reason this study
#: can be run here and essentially nowhere else. Twelve positions would be
#: 16.7 million and would put both quantities out of reach.
#:
#: **Its own constraint is vacuous.** ``transition_density=1.0`` leaves zero
#: forbidden adjacencies, which matters because an Ehrlich landscape scores a
#: sequence violating *its* transition matrix at minus infinity. Without this
#: the reward would carry a second feasibility rule that the declared predicate
#: knows nothing about, and a contrast meant to vary one thing would vary two.
SUPPORT_INSTANCE: dict[str, object] = {
    "sequence_length": 10,
    "vocab_size": 4,
    "n_motifs": 2,
    "motif_length": 2,
    "transition_density": 1.0,
    "seed": 7,
}

#: Positions in the support instance, named because the predicates are sized to
#: it and the budget is set equal to it.
SUPPORT_LENGTH = 10

#: Tokens per position in the support instance.
SUPPORT_TOKENS = 4

#: Which anchor the predicate factories must agree with. `Task.parent` resolves
#: an Ehrlich anchor as ``feasible_sequence(parent_seed)``, and a predicate that
#: forbade the anchor's own contacts would start every trajectory infeasible --
#: so the factories read the same seed rather than assuming a default.
SUPPORT_PARENT_SEED = 0

#: Induced widths the contact family sweeps. One is a path, and therefore the
#: shape `AdjacencyPredicate` already describes -- the rung that says the dial's
#: easy end reproduces the predicate this project shipped. Above it the
#: completion oracle costs ``O(L * v**(w+1) * B)``, so the axis is the exponent.
SUPPORT_WIDTHS: tuple[int, ...] = (1, 2, 3)

#: Share of the sequence space each rung's predicate admits, held **equal
#: across rungs**. This is the control that makes the family readable: the
#: question is what fraction of a feasible set the mask can reach, and a rung
#: whose feasible set is a different size is answering a different question.
#:
#: Calibrated rather than set. Per-contact density compounds over the contact
#: count, so a fixed density collapses the feasible set as the width grows --
#: measured, a shared 0.6 gave 21,188 feasible at width 1 and 61 at width 3,
#: which is a family varying its own difficulty rather than its structure.
#: Each rung therefore solves for the density that hits this share.
SUPPORT_FEASIBLE_TARGET = 0.02

#: Magnitude of the budget band's integer weights. Large on purpose, twice
#: over: it spreads the distribution of totals so a narrow band is genuinely
#: selective (at +/-4 even the single-point band admits ~6% of the space), and
#: the NP-hardness of the band is **weak**, so instances with small weights
#: admit a pseudo-polynomial dynamic program and are easy. Range is what makes
#: the reduction bite.
SUPPORT_WEIGHT = 16

#: Stream for the second band's weights. Independent of the first, which is
#: what stops one substitution repairing both totals at once.
SUPPORT_SECOND_SEED = 101

#: Separations the codebook family sweeps. The dial is the least Hamming
#: distance between two legal designs, and therefore the depth of lookahead a
#: sound mask needs: moving between neighbours costs that many substitutions and
#: every intermediate is illegal, so a mask seeing fewer moves ahead admits
#: nothing. Two is the smallest separation at which a local mask is already
#: stuck; four is deep enough that the cost of soundness, `alphabet**depth`
#: continuations per move, is visibly growing.
SUPPORT_SEPARATIONS: tuple[int, ...] = (2, 3, 4)

#: Designs per codebook, held equal across separations so the feasible set does
#: not change size with the dial -- the same control the contact family applies
#: by solving for its per-contact density.
SUPPORT_CODEBOOK = 800

#: Lengths the codebook family is built at. Ten positions is where the *support*
#: table lives, because measuring the reachable share exactly needs the Hamming
#: ball enumerated and 4^10 is inside the limit. The longer rungs exist because
#: that requirement does not apply to the method evaluation: coverage and
#: precision are taken against the codebook, which is known exactly by
#: construction at any length, so they can be measured where a shallow search
#: genuinely fails rather than only where it is cheap.
#:
#: Longer is also *easier* to construct, which is worth stating because it reads
#: as though it should be harder. Random codewords over four tokens differ in
#: about three quarters of their positions, so separation is nearly free once
#: the sequence is long: reaching 800 designs at separation four took 6,246
#: draws at ten positions and 799 -- one per design -- at twenty.
SUPPORT_LENGTHS: tuple[int, ...] = (10, 20, 30)


def _contact_pairs(width: int, length: int = SUPPORT_LENGTH) -> np.ndarray:
    """Coupled positions whose constraint graph has the requested induced width.

    Built by adding skips to a path: the path alone is width 1, adding every
    ``(i, i+2)`` makes it 2, and ``(i, i+3)`` makes it 3. The point is that the
    *predicate* is the same kind of object at every rung -- a set of pairwise
    constraints -- and only how densely the positions couple changes.

    Args:
        width: Induced width wanted. Must be at least 1.
        length: Sequence length.

    Returns:
        An ``(m, 2)`` array of position pairs.

    Raises:
        ValueError: If ``width`` is below 1.
    """
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    pairs = [(i, i + skip) for skip in range(1, width + 1) for i in range(length - skip)]
    return np.asarray(pairs, dtype=np.int64)


def _contact_predicate(width: int) -> Callable[[FitnessLandscape], FeasibilityPredicate]:
    """A factory for the contact predicate at one rung of the width dial.

    A factory rather than a predicate because the permitted token pairs are
    drawn *around the anchor*: every contact is forced to permit the pair the
    anchor already carries there, which is what stops the campaign beginning at
    an infeasible state. The anchor is not known until the landscape is built,
    which is exactly the reason `Task.feasibility` takes a factory.

    Args:
        width: Which rung.

    Returns:
        A callable taking the landscape and returning the predicate.
    """

    def build(landscape: FitnessLandscape) -> FeasibilityPredicate:
        pairs = _contact_pairs(width)
        # Solve for the per-contact density that leaves the same share of the
        # space feasible as every other rung: independently, m contacts each
        # admitting d leave d**m, so d is the m-th root of the target.
        density = SUPPORT_FEASIBLE_TARGET ** (1.0 / len(pairs))
        rng = np.random.default_rng(SUPPORT_INSTANCE["seed"])  # type: ignore[arg-type]
        permitted = rng.random((len(pairs), SUPPORT_TOKENS, SUPPORT_TOKENS)) < density
        anchor = np.asarray(landscape.feasible_sequence(SUPPORT_PARENT_SEED))  # type: ignore[attr-defined]
        for index, (left, right) in enumerate(pairs):
            permitted[index, anchor[left], anchor[right]] = True
        return ContactPredicate(
            pairs, permitted, length=SUPPORT_LENGTH, alphabet_size=SUPPORT_TOKENS
        )

    return build


def _band_bounds(weights: np.ndarray, target: float = SUPPORT_FEASIBLE_TARGET) -> int:
    """The half-width admitting `SUPPORT_FEASIBLE_TARGET` of the space.

    Computed exactly rather than sampled. A sequence's total is a sum of one
    weight per position, so the distribution over a uniform sequence is the
    convolution of the ten per-position distributions -- cheap over an integer
    range, deterministic, and it makes the band a *calibrated* quantity rather
    than a constant somebody picked.

    Args:
        weights: The ``(length, tokens)`` integer weights, already shifted so
            the anchor totals zero.
        target: Share of the space the band should admit. Defaults to the shared
            target; a conjunction passes the square root, so that two bands
            together admit the shared target.

    Returns:
        The smallest half-width whose band admits at least the target share.
    """
    low, high = int(weights.min(axis=1).sum()), int(weights.max(axis=1).sum())
    distribution = np.zeros(high - low + 1)
    distribution[-low] = 1.0
    for position in range(weights.shape[0]):
        convolved = np.zeros_like(distribution)
        for token in range(weights.shape[1]):
            convolved += np.roll(distribution, int(weights[position, token]))
        distribution = convolved / weights.shape[1]

    zero = -low
    total = distribution.sum()
    for half in range(high - low + 1):
        share = distribution[max(0, zero - half) : zero + half + 1].sum() / total
        if share >= target:
            return half
    return high - low


def _band_predicate() -> Callable[[FitnessLandscape], FeasibilityPredicate]:
    """A factory for the budget band -- the rung where no tractable mask exists.

    Every position enters one running total, so the constraint graph is complete
    and the induced width is the sequence length. Deciding whether a legal
    construction order exists is **NP-complete**, by reduction from PARTITION.

    The weights are shifted so the anchor's own tokens weigh nothing, which puts
    the anchor at a total of zero and therefore inside any band containing it.
    That is the same device the reduction uses, and it is what makes the anchor
    feasible without weakening the constraint anywhere else.

    Returns:
        A callable taking the landscape and returning the predicate.
    """

    def build(landscape: FitnessLandscape) -> FeasibilityPredicate:
        rng = np.random.default_rng(SUPPORT_INSTANCE["seed"])  # type: ignore[arg-type]
        shape = (SUPPORT_LENGTH, SUPPORT_TOKENS)
        weights = rng.integers(-SUPPORT_WEIGHT, SUPPORT_WEIGHT + 1, shape)
        anchor = np.asarray(landscape.feasible_sequence(SUPPORT_PARENT_SEED))  # type: ignore[attr-defined]
        # Zero the anchor's own tokens, so its total is zero and it is legal.
        weights = weights - weights[np.arange(SUPPORT_LENGTH), anchor][:, None]
        half = _band_bounds(weights)
        return BudgetBandPredicate(weights, low=-half, high=half, length=SUPPORT_LENGTH)

    return build


def _two_band_predicate() -> Callable[[FitnessLandscape], FeasibilityPredicate]:
    """A factory for two simultaneous bands on independently drawn weights.

    The rung that a one-step lookahead cannot repair. With a single band, a
    state whose running total has left it is restored by any substitution whose
    weight is large enough to carry the total back, and because the weights span
    more than the band's width such a substitution almost always exists -- so a
    mask looking one move ahead recovers the whole feasible set, and the
    measured reachable share went from 0.0005 under a strictly local mask to
    1.000 under depth-one lookahead.

    Drawing a second set of weights independently removes that repair. Each
    substitution moves both totals at once, and a move chosen to carry the first
    back inside its band moves the second by an unrelated amount, so restoring
    feasibility generally requires several substitutions chosen together. Each
    band is calibrated to the square root of the shared target, so that the two
    together admit the same share of the space as every other rung.

    Returns:
        A callable taking the landscape and returning the predicate.
    """

    def build(landscape: FitnessLandscape) -> FeasibilityPredicate:
        anchor = np.asarray(landscape.feasible_sequence(SUPPORT_PARENT_SEED))  # type: ignore[attr-defined]
        shape = (SUPPORT_LENGTH, SUPPORT_TOKENS)
        bands = []
        for stream in (SUPPORT_INSTANCE["seed"], SUPPORT_SECOND_SEED):
            rng = np.random.default_rng(stream)  # type: ignore[arg-type]
            weights = rng.integers(-SUPPORT_WEIGHT, SUPPORT_WEIGHT + 1, shape)
            weights = weights - weights[np.arange(SUPPORT_LENGTH), anchor][:, None]
            half = _band_bounds(weights, target=SUPPORT_FEASIBLE_TARGET**0.5)
            bands.append(BudgetBandPredicate(weights, low=-half, high=half, length=SUPPORT_LENGTH))
        return ConjunctionPredicate(bands)

    return build


def _codebook_predicate(
    separation: int, length: int = SUPPORT_LENGTH
) -> Callable[[FitnessLandscape], FeasibilityPredicate]:
    """A factory for a codebook whose designs are mutually well separated.

    Built greedily around the anchor: candidates are drawn from the shared
    stream and accepted when they lie at least ``separation`` substitutions from
    every design already accepted, so the guarantee the predicate records is
    established by construction rather than asserted. The anchor is accepted
    first, which is what makes the campaign legal at its own starting point.

    Args:
        separation: Least Hamming distance between two legal designs.
        length: Sequence length the codebook is drawn at.

    Returns:
        A callable taking the landscape and returning the predicate.
    """

    def build(landscape: FitnessLandscape) -> FeasibilityPredicate:
        anchor = np.asarray(landscape.feasible_sequence(SUPPORT_PARENT_SEED))  # type: ignore[attr-defined]
        rng = np.random.default_rng(SUPPORT_INSTANCE["seed"])  # type: ignore[arg-type]
        book = [anchor]
        accepted = anchor[None, :]
        for candidate in rng.integers(0, SUPPORT_TOKENS, (400_000, length)):
            if int((accepted != candidate).sum(axis=1).min()) >= separation:
                book.append(candidate)
                accepted = np.asarray(book)
                if len(book) >= SUPPORT_CODEBOOK:
                    break
        return CodebookPredicate(
            accepted,
            length=length,
            alphabet_size=SUPPORT_TOKENS,
            separation=separation,
        )

    return build


def support_study() -> tuple[Task, ...]:
    """Tasks whose axis is whether a *sound* mask can be built at all.

    Every other constrained task in this suite poses the same feasibility rule:
    permitted adjacent token pairs, whose constraint graph is a path. That rule
    is the tractable case by construction -- the completion oracle is a dynamic
    program, an exact projection exists, and a learned sampler can buy no
    support advantage over it. So the condition this project states is measured
    only on the half where it holds, and the half where it fails is instantiated
    nowhere.

    This family instantiates it. The rungs share one landscape, one anchor and
    one protocol, and differ **only** in the predicate:

    * ``support-w1`` -- contacts along a path. Induced width 1, and therefore the
      shape `AdjacencyPredicate` already describes; the rung that says the dial's
      easy end reproduces what ships.
    * ``support-w2``, ``support-w3`` -- the same kind of predicate with positions
      coupled more densely. The oracle costs ``O(L * v**(w+1) * B)``, so this is
      the exponent moving, and the tractable side degrades continuously rather
      than falling off a cliff.
    * ``support-band`` -- a weighted running total confined to a band. Every
      position couples to every other, and deciding constructibility is
      NP-complete by reduction from PARTITION. The endpoint the dial points at.

    **Unbudgeted, and fixed-anchor, both deliberately.** The per-round budget is
    the sequence length, so it never binds; and the anchor does not move. A
    design unreachable here is unreachable because of the *predicate*, which is
    the only one of the three causes this family exists to isolate. Adding a
    budget or a moving anchor would reintroduce the other two and make the
    contrast unreadable.

    Returns:
        One task per rung, widths first and the band last. None declares an
        attainable optimum: the reachable set differs at every rung by
        construction, so a neighbouring rung's audited value would be a floor no
        method could clear. This family is read on support, not on regret.
    """
    protocol = Protocol(rounds=4, batch_size=PLATE, max_mutations=SUPPORT_LENGTH)
    shared = (
        "Ten positions over four tokens, so the feasible and reachable sets are "
        "both enumerable exactly. The instance's own transition constraint is "
        "vacuous, so the declared predicate is the only feasibility rule. "
        "Unbudgeted and fixed-anchor, so unreachability is the predicate's doing "
        "and not the budget's or the anchor's."
    )
    tasks = [
        _task(
            f"support-w{width}",
            f"Does a sound mask exist when the constraint graph has induced width "
            f"{width}? {shared}",
            _ehrlich(**SUPPORT_INSTANCE),
            protocol,
            reanchor=False,
            attainable=None,
        )
        for width in SUPPORT_WIDTHS
    ]
    band = _task(
        "support-band",
        "Does a sound mask exist when deciding constructibility is NP-complete? "
        "A weighted running total confined to a band, so every position couples "
        f"to every other. {shared}",
        _ehrlich(**SUPPORT_INSTANCE),
        protocol,
        reanchor=False,
        attainable=None,
    )
    two_band = _task(
        "support-band2",
        "Does a sound mask exist when two independent totals must both stay in "
        "band? A single band is repaired by one substitution of sufficient "
        "weight, so a mask looking one move ahead recovers all of it; two "
        "totals drawn independently are not repaired together by any single "
        f"move. {shared}",
        _ehrlich(**SUPPORT_INSTANCE),
        protocol,
        reanchor=False,
        attainable=None,
    )
    separated = [
        _task(
            f"support-sep{separation}",
            f"How far ahead must a mask see when legal designs are {separation} "
            f"substitutions apart? Every intermediate between two of them is "
            f"illegal, so a mask seeing fewer than {separation - 1} moves ahead "
            f"admits nothing at all. {shared}",
            _ehrlich(**SUPPORT_INSTANCE),
            protocol,
            reanchor=False,
            attainable=None,
        )
        for separation in SUPPORT_SEPARATIONS
    ]
    # The same codebook family at lengths where a shallow search genuinely
    # fails. Only the method evaluation runs here: coverage and precision are
    # taken against the codebook, which is exact by construction at any length,
    # where the support table needs the Hamming ball enumerated and so stays at
    # ten positions.
    longer = [
        _task(
            f"support-sep{separation}-L{length}",
            f"Does a learned flow recover the support at {length} positions, "
            f"where legal designs are {separation} substitutions apart and a "
            f"depth-two search cannot reach them? The feasible set is the "
            f"codebook and is known exactly without enumerating the ball, so "
            f"coverage and precision are measurable here even though the "
            f"reachable share is not. {shared}",
            _ehrlich(**{**SUPPORT_INSTANCE, "sequence_length": length}),
            Protocol(rounds=4, batch_size=PLATE, max_mutations=length),
            reanchor=False,
            attainable=None,
        )
        for length in SUPPORT_LENGTHS
        if length != SUPPORT_LENGTH
        for separation in (4,)
    ]
    return (*tasks, band, two_band, *separated, *longer)


def support_predicates() -> dict[str, Callable[[FitnessLandscape], FeasibilityPredicate]]:
    """The predicate each support task declares, by task name.

    Kept beside `support_study` rather than inside it because `Task` is frozen
    and the tasks are built by the shared `_task` helper, which knows nothing
    about predicates. `support_tasks` is what puts the two together.

    Returns:
        Factories by task name.
    """
    return {
        **{f"support-w{width}": _contact_predicate(width) for width in SUPPORT_WIDTHS},
        "support-band": _band_predicate(),
        "support-band2": _two_band_predicate(),
        **{
            f"support-sep{separation}": _codebook_predicate(separation)
            for separation in SUPPORT_SEPARATIONS
        },
        **{
            f"support-sep4-L{length}": _codebook_predicate(4, length)
            for length in SUPPORT_LENGTHS
            if length != SUPPORT_LENGTH
        },
    }


def support_tasks() -> tuple[Task, ...]:
    """The support study with each task's predicate attached.

    Returns:
        The tasks of `support_study`, each carrying the factory
        `support_predicates` names for it.
    """
    declared = support_predicates()
    return tuple(replace(task, feasibility=declared[task.name]) for task in support_study())


def support_tier(seeds: Sequence[int]) -> Tier:
    """The support study as a tier.

    Args:
        seeds: Seeds per arm.

    Returns:
        A `Purpose.DIAGNOSTIC` tier. Diagnostic and not benchmark: what it
        varies is the *shape of the reachable set*, which explains where a
        method can help rather than establishing that one did.
    """
    return Tier("support", support_tasks(), tuple(seeds), Purpose.DIAGNOSTIC)


#: Per-round mutation radii the rejection diagnostic sweeps. The axis is how
#: many substitutions a design is *permitted* to carry, which is what the
#: feasibility filter is ultimately refusing: each substitution is another
#: chance to break an adjacency the transition matrix forbids.
#:
#: What each rung is for:
#:
#: * **1** and **2** -- below the radius every other diagnostic runs at, where
#:   the ceiling binds even the published mutation kernel and the acceptance
#:   rate is at its highest. Two rungs rather than one, because a single point
#:   below the shared radius cannot say whether the axis is a line or a knee.
#:   The rung at **1** is also the axis's own control: a kernel matched to a
#:   radius of one *is* ``p_m = 1/L`` on the diagnostic instance, so the two
#:   columns `rejection_arms` builds are the same configuration there and must
#:   report the same acceptance rate. A gap at that rung is a fault in the
#:   measurement rather than a property of either kernel.
#: * **`DIAGNOSTIC_MUTATIONS`** -- the radius every other diagnostic runs at, so
#:   the curve passes through the configuration the rest are read at. This rung
#:   is `objective_task` itself, not a copy of it.
#: * **8** and **16** -- above it, where the published kernel has essentially no
#:   proposal mass and the curve therefore *stops moving*. That flattening is
#:   the confound rather than a result, and having two rungs in it is what makes
#:   the flattening visible as flattening instead of as one point that happened
#:   to land near its neighbour.
#:
#: The top rung is half the diagnostic instance's length, so a design there may
#: differ from its anchor at half its positions and the ceiling is still a real
#: constraint rather than a formality.
REJECTION_RADII: tuple[int, ...] = (1, 2, DIAGNOSTIC_MUTATIONS, 8, 16)


def rejection_curve() -> tuple[Task, ...]:
    """Tasks varying how many substitutions a design may carry.

    The axis the three ``draws_*`` counters need and do not have. They are
    campaign totals -- so many offspring bred, so many refused as unreachable,
    so many of the survivors the anchor unchanged -- measured at one radius on
    tasks designed to measure something else, which makes them telemetry rather
    than a result: three numbers, at one setting, with nothing to read them
    against.

    The claim they are wanted for is about **acceptance as a function of
    mutation count**. Every substitution an offspring carries is another chance
    to break an adjacency the transition matrix forbids, so the share the filter
    admits should fall as the permitted count rises, and *how fast* it falls is
    what decides whether rejection sampling is a cost a method absorbs or a wall
    it hits. Varying the radius is what turns the counters into that curve.
    Everything else is held at `DIAGNOSTIC_INSTANCE` and `DIAGNOSTIC_PROTOCOL`,
    so a rate that moves across the family moves with the radius and nothing
    else.

    The confound, which travels with any claim drawn from this
    ---------------------------------------------------------

    The published mutation rate is ``p_m = 1/L`` -- Stanton et al.'s setting,
    and what `BASELINES`'s rejection arm runs -- so the number of substitutions
    an offspring carries is **Poisson(1) whatever this axis says**. About 37% of
    draws carry none at all, 98% carry four or fewer, and the mass above that is
    a fraction of a percent. Widening the radius past three or four therefore
    changes almost nothing about what is *proposed*: the designs the filter
    would have refused were never bred in the first place.

    So confinement near the anchor is rejection **and** the published kernel
    together, and this family cannot attribute it to rejection alone. Claiming
    the whole gap for rejection is overreach. `rejection_arms` is what makes the
    separation possible rather than assumed: a second column whose kernel is
    scaled to each rung's radius, so the designs the axis permits are actually
    drawn. Read as a pair, the difference *between* columns at one radius is the
    kernel's share and the slope *within* a column is what the filter costs once
    the proposal mass is there. Read alone, either column is the overreach.

    The matched column is expected to give up on the widest rungs, and that is
    a result rather than a gap in the family: an exhausted campaign still stores
    its ``draws_*`` counters, which are the very quantities this curve is read
    on, so a rung that could not fill a plate reports the acceptance rate that
    explains why.

    Returns:
        One task per entry in `REJECTION_RADII`, in that order. The rung at
        `DIAGNOSTIC_MUTATIONS` **is** `objective_task`, returned rather than
        copied, for the reason `constraint_density` returns it: the store keys
        on ``(task, arm)``, so a renamed twin of a task already defined pays for
        the same campaigns twice and files the two under keys nothing can
        compare.

        Every other rung declares no attainable optimum, in both directions and
        for two different reasons. `DIAGNOSTIC_ATTAINABLE` is what four
        re-anchored rounds *of four* reach: the narrow rungs reach a strictly
        smaller set, so carrying that value across would put a floor no method
        could clear into every regret on them; the wide rungs reach a larger set
        that no audit has covered, and a bound declared on some rungs of a family
        and not others gives a regret column that cannot be read down its own
        length. This family is read on the ``draws_*`` columns, not on regret.
    """
    return tuple(
        objective_task()
        if radius == DIAGNOSTIC_MUTATIONS
        else _task(
            f"radius-{radius}",
            f"How much does the feasibility filter refuse when a design may "
            f"carry {radius} substitutions? The shared diagnostic instance in "
            f"every parameter but the per-round radius, so the acceptance rate "
            f"moves with the permitted mutation count alone. {CUMULATIVE}",
            DIAGNOSTIC_LANDSCAPE,
            # Rebuilt from the shared protocol with one field replaced, so the
            # rungs cannot differ in the budget as well as in the radius.
            replace(DIAGNOSTIC_PROTOCOL, max_mutations=radius),
            reanchor=True,
            attainable=None,
        )
        for radius in REJECTION_RADII
    )


def _matched_kernel_genetic(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    """A rejection GA whose mutation kernel is scaled to the radius it searches.

    The arm that makes `rejection_curve` separable. The published kernel is
    ``p_m = 1/L`` at every radius, so the substitution count is Poisson(1) on
    every rung and the wide rungs of the axis are never actually visited -- an
    acceptance rate that stops falling there says the offspring stopped
    changing, not that the filter stopped refusing. This one sets
    ``p_m = max_mutations / L``, which puts the *mean* draw at the ceiling the
    environment is enforcing, so a design carrying that many substitutions is an
    ordinary draw rather than a tail event.

    Everything else is `BASELINES`'s ``genetic-feasible`` configuration, its 200
    attempts included, so a difference between the two columns is the kernel and
    nothing else.

    Args:
        env: Supplies the parent, the alphabet, the radius the kernel is scaled
            to, and the feasibility rule the draws are filtered against.
        seed: Seeds the population and the operators.
        _protocol: Unused; the radius comes from the environment, which is the
            object that will actually refuse an over-budget design.

    Returns:
        The sampler. The rate is capped at 1.0 for a radius at or above the
        sequence length, where every position is expected to mutate and the
        ceiling has stopped constraining anything anyway.
    """
    return GeneticAlgorithm(
        env,
        seed=seed,
        mutation_prob=min(1.0, env.max_mutations / env.sequence_length),
        feasible_only=True,
        max_attempts=200,
    )


def fixed_anchor_task() -> Task:
    """The shared diagnostic task with the anchor held still.

    The control for the first axis of the anchor study: identical to
    `objective_task` -- same instance, same protocol, same radius -- except
    that the search ball never moves. Everything a comparison against it
    measures is therefore the anchor rule.

    The axis is presently unmeasured rather than measured and flat. The
    diagnostics that vary protocol shape were run when no task could move its
    anchor, so every shape in them searched one identical Hamming ball and the
    curves could not have separated whatever they were varying from the fact
    that none of the campaigns could go anywhere. That is why the control has to
    be run rather than looked up.

    Returns:
        The task. It declares no attainable optimum, and that is the honest
        state rather than an omission: `DIAGNOSTIC_ATTAINABLE` is what four
        *re-anchored* rounds reach, and a fixed anchor reaches one round's
        radius from the wild type, so storing regret against the re-anchored
        value would report the difference between two reachable sets as this
        arm's shortfall.
    """
    return _task(
        "anchor-fixed",
        "Does moving the search ball to the best design so far help at all? "
        "The shared diagnostic task with the anchor held still, so a comparison "
        f"against it isolates the anchor rule from everything else. {CAPPED}",
        DIAGNOSTIC_LANDSCAPE,
        DIAGNOSTIC_PROTOCOL,
        reanchor=False,
        attainable=None,
    )


class Purpose(StrEnum):
    """What a tier's results are for, which decides how they may be used.

    A single "is this the headline" flag cannot express this, because the two
    kinds of non-headline tier differ in a way that matters to a reader rather
    than only to us. A diagnostic measures how methods behave; a selection tier
    measures nothing, it *chooses our own configuration*, and a choice made on a
    landscape a claim is later drawn from is tuning on the test set. Keeping the
    distinction in the type means a tier cannot quietly drift from one role to
    the other, and the results table can refuse a tier that was never eligible
    to appear in it.
    """

    #: Compares methods at equal budget. These rows carry the claims.
    BENCHMARK = "benchmark"

    #: Explains behaviour -- response curves, ablations, objective comparisons.
    #: Informs the discussion; never a row in the results table.
    DIAGNOSTIC = "diagnostic"

    #: Fixes our own hyperparameters before the benchmark runs. Must sit on a
    #: landscape no benchmark task uses, and must be reported as a method
    #: detail rather than as a finding.
    SELECTION = "selection"


@dataclass(frozen=True, slots=True)
class Tier:
    """A group of tasks run at one seed count, with a reason to exist.

    Attributes:
        name: Short identifier.
        tasks: What to run.
        seeds: Seeds per arm.
        purpose: What the results may be used for. See `Purpose`.
    """

    name: str
    tasks: tuple[Task, ...]
    seeds: tuple[int, ...]
    purpose: Purpose

    @property
    def headline(self) -> bool:
        """Whether these results carry claims.

        Kept because callers ask this question far more often than they ask
        which of the two non-headline roles a tier plays.
        """
        return self.purpose is Purpose.BENCHMARK

    def __repr__(self) -> str:
        """Name the tier, its size and its standing."""
        return f"{self.name} ({self.purpose}, {len(self.tasks)} tasks x {len(self.seeds)} seeds)"


def constraint_density_tier(seeds: Sequence[int]) -> Tier:
    """The constructibility sweep as a tier.

    Args:
        seeds: Seeds per arm.

    Returns:
        A `Purpose.DIAGNOSTIC` tier over `constraint_density`. Diagnostic and
        not benchmark: it explains what the feasibility mechanism is up against
        on a landscape whose density we chose, which informs the discussion and
        is not a row anyone may claim a method won.
    """
    return Tier("constraint-density", constraint_density(), tuple(seeds), Purpose.DIAGNOSTIC)


def rejection_arms() -> dict[str, Methodology]:
    """The two mutation kernels `rejection_curve` is read across.

    One axis of the diagnostic is the task's radius; this is the other, and it
    is an arm rather than a task because the mutation rate belongs to the
    sampler. Two columns is the minimum that separates anything: a single column
    at the published rate measures rejection and the kernel together and cannot
    say which produced the shape it draws.

    Returns:
        The shipped rejection arm under its own name, looked up rather than
        rebuilt so that the cells it shares with the rest of the suite stay the
        cells the store already keys, and `_matched_kernel_genetic` beside it.
        The two differ in the mutation rate and in nothing else -- same class,
        same attempt budget, same everything the campaign supplies -- which is
        what makes the difference between the columns attributable.
    """
    from evogfn.benchmark.methods import BASELINES, classical  # noqa: PLC0415 - arms import tasks

    return {
        "genetic-feasible": BASELINES["genetic-feasible"],
        "genetic-feasible-matched": classical(_matched_kernel_genetic),
    }


def rejection_tier(seeds: Sequence[int]) -> Tier:
    """The rejection sweep as a tier.

    Args:
        seeds: Seeds per arm.

    Returns:
        A `Purpose.DIAGNOSTIC` tier over `rejection_curve`. Diagnostic and not
        benchmark, on the same reasoning `constraint_density_tier` carries: it
        explains what a rejection-sampling method is up against on a landscape
        and at radii we chose, which informs the discussion and is not a row
        anyone may claim a method won.
    """
    return Tier("rejection-curve", rejection_curve(), tuple(seeds), Purpose.DIAGNOSTIC)


@dataclass(frozen=True, slots=True)
class AnchorCell:
    """One cell of the anchor study: a task, an arm, and what the pair varies.

    A cell rather than a tier, because this study is not a cross. A tier runs
    every arm on every task, and here two of the combinations that cross would
    produce are things that must not be run: one because it is already stored
    under this exact key, and one because it is the same campaign as its
    neighbour under a second name. Naming the cells is what keeps both out.

    Attributes:
        task: What to run. Existing tasks are returned by the functions that
            already define them rather than rebuilt, so a cell that reuses one
            reuses its **name**, which is half of the store's key.
        arm: The methodology's name, and the other half of that key. Arms that
            already exist keep their exact names for the same reason.
        moves_anchor: Whether this cell's task re-anchors. The first axis.
        carries_policy: Whether the arm's learned state survives a move, or
            ``None`` where the axis does not apply -- an arm with no learned
            state to carry, or a task whose anchor never moves, in which case
            nothing is ever rebuilt and the two settings are one campaign.
    """

    task: Task
    arm: str
    moves_anchor: bool
    carries_policy: bool | None

    def __repr__(self) -> str:
        """Name the cell by its two coordinates."""
        anchor = "moved" if self.moves_anchor else "fixed"
        carry = {True: "carried", False: "rebuilt", None: "no state"}[self.carries_policy]
        return f"{self.task.name}/{self.arm} ({anchor}, {carry})"


def anchor_study() -> tuple[AnchorCell, ...]:
    """The cells that separate "the anchor moved" from "the policy came with it".

    Two orthogonal mechanisms, and conflating them answers a different question
    than the one this project's claim rests on.

    **Does moving the ball help?** A property of the protocol, available to
    every method, and so not ours to take credit for. That is why a genetic
    algorithm is on this axis: if re-anchoring lifts it too, the protocol is
    doing the work.

    **Given that it moves, does bringing the trained policy along help?** This
    is amortisation, and it is the thing a learned constructive sampler has that
    a genetic algorithm structurally cannot: a GA's operator is the same
    before and after a move, so it has nothing to carry. The claim worth making
    is not that a trained policy is useful, which is trivial, but that it
    remains useful after the ball it was trained on has moved -- so the cell
    that matters is the interaction, and the fixed-anchor row is what makes it
    specific rather than a restatement of "training helps".

    No baselines beyond the one contrast, for the reason the ``+screen`` /
    ``+search`` ladder has none: this is an ablation, and an ablation answers
    which part of a method does the work. Which method wins is a different
    question with its own table.

    Returns:
        Five cells, of which the store already holds one under this exact key.

        The re-anchored GFlowNet cell is `objective_task` under its shipped arm
        name -- the same campaign the objectives diagnostic already ran, reused
        rather than re-declared, since a renamed twin would pay for it twice and
        file the two copies where nothing can compare them.

        The fixed-anchor row carries **one** GFlowNet cell rather than two.
        Carrying and rebuilding differ only in what happens when the anchor
        moves, and it never moves there, so the two arms describe the same
        campaign; running both would be the twin this study is arranged to
        avoid, and the interaction is read from the moved row against a shared
        control.

        A caveat on what "already stored" means for the genetic cell: the
        configuration of `objective_task` is also defined under two other names
        in this module, so a baseline result at this configuration may exist on
        disk under one of those keys instead. The study names the canonical task
        and lets the store decide what is missing.
    """
    moved = objective_task()
    fixed = fixed_anchor_task()
    return (
        AnchorCell(moved, "gfn-tb", moves_anchor=True, carries_policy=True),
        AnchorCell(moved, "gfn-tb-rebuilt", moves_anchor=True, carries_policy=False),
        AnchorCell(fixed, "gfn-tb", moves_anchor=False, carries_policy=None),
        AnchorCell(moved, "genetic", moves_anchor=True, carries_policy=None),
        AnchorCell(fixed, "genetic", moves_anchor=False, carries_policy=None),
    )


def _scores(
    task: Task, result: CampaignResult, attainable: AttainableOptimum | None
) -> dict[str, object]:
    """The two numbers a stored record is indexed by, whatever it measured.

    `RunRecord` has one larger-is-better field and one smaller-is-better field,
    and every table built off the store reads them positionally. So the only
    safe thing to put in them is a pair that means the same thing down the whole
    column, and what that pair *is* differs by objective count:

    * **One objective.** Best value measured, and the distance from it to the
      **attainable** optimum -- what the audit found this task's search space to
      contain, conservatively its searched lower bound. Not the landscape's own
      optimum: a target outside the reachable set contributes a floor no method
      could clear, and an arm sitting exactly on the reachable maximum would
      still be reported at a regret.
    * **More than one.** Hypervolume above the campaign's reference point, and
      IGD+ against its reference front. These are the multi-objective
      counterparts with the same orientation -- hypervolume rises as the set
      improves, IGD+ falls to zero when the front is covered -- so a column
      built from them is at least internally coherent.
      [CampaignResult.best_value][evogfn.loop.ledger.CampaignResult.best_value]
      raises on a multi-objective result rather than returning the maximum over
      designs *and* objectives, which is why this branch exists at all.

    A regret can come out **negative**, and that is deliberate. Where the audit
    could only bracket the attainable optimum, an arm beating the searched lower
    bound is evidence about the audit rather than about the arm, and clamping it
    at zero would erase the only signal that a bound needs re-deriving.

    A campaign that never finished does not reach this function at all: it has no
    `CampaignResult` to be scored from, and `run_task` writes ``nan``/``None``
    into the same two fields with ``exhausted=True`` beside them. So the pair
    this returns always means "measured", and the one thing that could break
    that -- an unfinished run smuggled in under a plausible number -- has no path
    to it.

    Args:
        task: The task being run, named in any error.
        result: The completed campaign.
        attainable: What this task's search space was audited to contain, or
            ``None`` where no audit covers it -- in which case no regret is
            stored at all. An absent number is recoverable; a number taken
            against an unreachable target is not distinguishable from a real one
            once it is in the column.

    Returns:
        ``best`` and ``regret``, ready to pass to
        [ResultStore.stamp][evogfn.benchmark.store.ResultStore.stamp].

    Raises:
        ValueError: If a multi-objective campaign supplied no reference point.
            Storing such a run under a ``best`` that means something different
            from every other row in the column is precisely the silent
            mixing the store exists to prevent, so it is refused rather than
            filled with ``nan``.
    """
    if not result.is_multi_objective:
        best = result.best_value
        return {
            "best": best,
            "regret": None if attainable is None else attainable.lower - best,
        }
    volume = _indicator(result)
    if volume is None:
        raise ValueError(
            f"{task.name} measured {result.n_objectives} objectives but supplied no reference "
            f"point, so its result has no indicator to be stored under; give the campaign a "
            f"reference_point, or score it single-objective"
        )
    return {"best": volume, "regret": result.igd_plus}


def _indicator(result: CampaignResult) -> float | None:
    """Hypervolume for a multi-objective run, or ``nan`` where it is not exact.

    The exact hypervolume in [evogfn.metrics.pareto][] is inclusion-exclusion
    over the front, which it caps at 16 points for three or more objectives. A
    384-design campaign can easily carry a larger front than that, and the
    honest answer is then that the indicator was not computed -- an approximation
    written into the same column as an exact value would be indistinguishable
    from one. ``nan`` propagates; the measurements survive on the result for
    anyone who wants to score them with a dedicated implementation.

    Args:
        result: The completed campaign.

    Returns:
        The dominated volume, ``nan`` where the exact method cannot run, or
        ``None`` when no reference point was supplied.
    """
    try:
        return result.hypervolume
    except NotImplementedError:
        return float("nan")


def _sampler_fields(sampler: Sampler) -> dict[str, object]:
    """What the sampler itself recorded about how it spent the round.

    Every one is read **by attribute**, so a sampler that does not carry a
    counter stores the neutral value rather than making the base interface
    declare a quantity only one method can measure. Zero is then the honest
    reading in each case: an arm that breeds nothing bred nothing, and an arm
    that rejects nothing rejected nothing.

    ``fitted`` is the one exception to that, and the exception is deliberate.
    Zero is not its neutral value: an arm with no model to fit and an arm whose
    model never fitted are opposite findings, and ``False`` would report the
    first as the second on every classical row in the table. So its neutral
    value is ``None`` -- read by attribute like the rest, defaulting to ``None``
    rather than to a measurement. See
    `RunRecord.fitted`.

    One helper rather than the same block written at two call sites, and the
    reason is which two: the completed run and the run that raised. The
    exhausted one is where these numbers matter most -- the counters explaining
    *why* a sampler ran out live on the sampler, never on the result it failed to
    produce -- and two copies would drift, with the copy that lost a field being
    the one nobody reads until a column has a hole in it.

    Args:
        sampler: The sampler the campaign finished with. Under re-anchoring that
            is not the object the campaign was built from, so it has to come
            from [Campaign.sampler][evogfn.loop.campaign.Campaign.sampler].

    Returns:
        Fields ready to pass to
        [ResultStore.stamp][evogfn.benchmark.store.ResultStore.stamp].
    """
    fitted = getattr(sampler, "is_fitted", None)
    return {
        # Read off the sampler the campaign *finished* with, which is what makes
        # this answerable at all: `is_fitted` is state on the object, it is not
        # derivable from anything the result carries, and a campaign that ended
        # in its random-screening stage spends and proposes exactly what a fitted
        # one does.
        "fitted": None if fitted is None else bool(fitted),
        "proxy_calls": int(getattr(sampler, "proxy_calls", 0)),
        "bred_designs": int(getattr(sampler, "bred_designs", 0)),
        # The three rejection counters travel together or not at all: two of them
        # are a rate and the third is what the rate means. A rejection rate that
        # looks survivable because most admitted draws were the unmutated anchor
        # is the exact misreading `draws_unmutated` exists to prevent.
        "draws_attempted": int(getattr(sampler, "draws_attempted", 0)),
        "draws_rejected": int(getattr(sampler, "draws_rejected", 0)),
        "draws_unmutated": int(getattr(sampler, "draws_unmutated", 0)),
        "unconstructible_fraction": float(getattr(sampler, "unconstructible_fraction", 0.0)),
        # Read off the *final* sampler, and safe under re-anchoring only because
        # the arm that reports this carries its repair counters across a move
        # rather than restarting them. An arm that reset them would report its
        # last round here while looking like it reported the campaign -- the same
        # silent undercount the proxy spend once had.
        "repaired_fraction": float(getattr(sampler, "repaired_fraction", 0.0)),
    }


def _arm_parameters(method: Methodology, task: Task) -> dict[str, str | float | bool]:
    """What an arm declares it resolved to, on the task the record is for.

    By attribute, like the sampler quantities beside it: an arm built by
    [methods][evogfn.benchmark.methods] declares what it closed over, and a
    methodology defined anywhere else is a plain callable that declares nothing.
    Empty is then the honest record -- the settings were not stated, which is not
    the same as an arm having none.

    The task is passed because one setting is not knowable without it. The
    capacity-matched control's trunk width is resolved from the task's own
    sequence length and alphabet, so the number that trained is a property of the
    pair; an arm asked only for its static settings would report the same width on
    every shape and be wrong on all but one. Arms that resolve nothing per task
    are unaffected and answer exactly as before.

    Args:
        method: The methodology, asked rather than inspected.
        task: The task being recorded.

    Returns:
        The settings, ready for the record.
    """
    resolve = getattr(method, "parameters_for", None)
    if resolve is not None:
        return dict(resolve(task))
    return dict(getattr(method, "parameters", {}))


def round_rows(records: Sequence[RoundRecord]) -> list[dict[str, object]]:
    """The per-round ledger, flattened for storage.

    Built key by key rather than from the dataclass, which is what makes this
    the second of two hops a new ledger field has to be walked through by hand:
    a field added to
    [RoundRecord][evogfn.loop.ledger.RoundRecord] and not added here is present
    in memory, absent from every stored record, and silent about the difference.

    **Public, and the only implementation.** The multi-objective suite kept a
    second copy of this dict inline, and the copy had already drifted -- it was
    missing both repetition counters, so every multi-objective record stored
    since those were added carries neither. One function is what stops the two
    suites disagreeing about what a round is.

    Args:
        records: The rounds that completed. On an exhausted campaign this is
            short of the protocol's round count, which is itself the record of
            how far the run got.

    Returns:
        One dict per round, in order. Values are scalars except ``fitness`` and
        ``anchor``, which are lists.
    """
    return [
        {
            "index": float(record.index),
            "proposed": float(record.proposed),
            "screened": float(record.screened),
            "evaluated": float(record.evaluated),
            "feasible": float(record.feasible),
            "best_in_round": record.best_in_round,
            "best_so_far": record.best_so_far,
            "mean_in_round": record.mean_in_round,
            "batch_diversity": record.batch_diversity,
            "surrogate_correlation": record.surrogate_correlation,
            "hypervolume": record.hypervolume,
            # How far this round's anchor sat from the wild type. Flat at zero
            # says the campaign searched one Hamming ball for its whole life,
            # which is the difference between a budget of `max_mutations` and a
            # budget of `max_mutations` per round.
            "anchor_distance": float(record.anchor_distance),
            # The two repetition counts, stored adjacent because they are only
            # readable as a pair. `duplicates` is repetition **within** one
            # plate and was charged a well; `redundant` is repetition against
            # the campaign's memory **across** rounds and cost a proposal and
            # nothing else. A converged sampler produces both at once, so either
            # one alone attributes the other's cost to itself -- which is the
            # confusion the pair exists to resolve, and it is only resolved
            # where both numbers are stored.
            "duplicates": float(record.duplicates),
            "duplicate_fraction": record.duplicate_fraction,
            # `nan` and never `0.0` where the campaign kept no cross-round
            # memory: zero is the measurement "the memory refused nothing", and
            # "there was no memory" is a different fact that would otherwise be
            # indistinguishable from it -- the distinction
            # `surrogate_correlation` above already makes with `nan`, applied
            # to the counter that needs it for the same reason.
            "redundant": float("nan") if record.redundant is None else float(record.redundant),
            "redundant_fraction": record.redundant_fraction,
            # The plate itself, and the design it was built from. Both are
            # *lists* where every other value here is a scalar, which is the
            # reason this function's return type widened: the questions they
            # answer -- where `E[max]` over a plate saturates, and what a
            # round's diversity is relative to the set its own anchor could
            # reach -- are not functions of any summary statistic, so they were
            # answerable only by re-running the campaign.
            #
            # Costed before being added: one float per well against `length`
            # integers per well, which is why the measurements are stored and
            # the batch *sequences* still are not.
            "fitness": [float(value) for value in record.fitness],
            "anchor": None if record.anchor is None else [int(t) for t in record.anchor],
        }
        for record in records
    ]


def _exhausted_record(  # noqa: PLR0913 - a record is defined by what it declares
    store: ResultStore,
    *,
    task: Task,
    method: Methodology,
    name: str,
    seed: int,
    campaign: Campaign,
    cpu_seconds: float,
    wall_seconds: float,
) -> RunRecord:
    """A record of a campaign that could not finish.

    Same key, same protocol, same fingerprint and same provenance as a completed
    run, because it is the same experiment -- what differs is that it produced no
    measurement, and ``exhausted`` is the field that says so. That is the whole
    design: the seed stays in the store, so the arm keeps its row and the row
    says what happened, rather than the arm silently having fewer seeds than the
    table's header claims.

    **Nothing that was not measured is stored as a number.** ``best``,
    ``diversity`` and ``feasible_fraction`` are ``nan`` and ``regret`` is
    ``None``. Zero would be a measurement -- "this arm found nothing good", "this
    arm's designs were never constructible" -- and it would average into a column
    beside real ones, which is the failure this record exists to prevent and not
    a lesser version of it. ``nan`` and ``None`` are what the store already uses
    for a quantity nobody obtained, in ``surrogate_correlation`` and in the
    regret of an unaudited task.

    What *is* stored as a number is what genuinely happened: the rounds that
    completed, the oracle calls they charged, the proposals they cost, and every
    counter the sampler carries. A run that exhausted in round four having
    measured 288 designs and one that exhausted in round one having measured
    none are different findings.

    Args:
        store: Where the record will go, and what stamps it.
        task: The task being run.
        method: The methodology, asked by attribute for the settings it closed
            over -- exactly as on the completed path, so an exhausted row is
            attributable to a configuration like any other.
        name: The arm's name, which is half the store's key.
        seed: The seed that exhausted.
        campaign: The campaign that raised, read for its final sampler and for
            the rounds it did complete.
        cpu_seconds: Processor time spent before the failure.
        wall_seconds: Elapsed time for the same.

    Returns:
        The record, ready to append.
    """
    completed = campaign.completed_rounds
    return store.stamp(
        depends_on=RESULT_DEPENDENCIES,
        task=task.name,
        method=name,
        seed=seed,
        protocol=repr(task),
        parameters=_arm_parameters(method, task),
        exhausted=True,
        best=float("nan"),
        regret=None,
        diversity=float("nan"),
        feasible_fraction=float("nan"),
        duplicate_fraction=float("nan"),
        oracle_calls=sum(record.evaluated for record in completed),
        proposals=sum(record.proposed for record in completed),
        cpu_seconds=cpu_seconds,
        wall_seconds=wall_seconds,
        deterministic=is_deterministic(),
        # No designs, rather than the best of a partial campaign: these are read
        # as "what this arm produced", and an arm that did not produce a plate
        # produced nothing to inspect.
        top_sequences=[],
        trace=[record.best_so_far for record in completed],
        rounds=round_rows(completed),
        **_sampler_fields(campaign.sampler),
    )


def run_task(
    task: Task,
    methods: Mapping[str, Methodology],
    store: ResultStore,
    seeds: Sequence[int],
    *,
    report: Callable[[str], None] = print,
) -> int:
    """Run whatever is missing for one task, storing each result as it lands.

    A campaign that cannot finish is stored too, as a record carrying
    ``exhausted`` and no measurement -- see `_exhausted_record`. That failure is
    a property of the method on this task and belongs in the store beside the
    successes, rather than being a seed that silently never appears.

    Args:
        task: What to run.
        methods: Methodologies by name.
        store: Where results go, and what says which are already held.
        seeds: Seeds wanted.
        report: Where progress lines go.

    Returns:
        How many campaigns were actually run, counting those that exhausted:
        they cost the same time and are now held, so the next sweep skips them.

    Raises:
        ValueError: If the task declares an attainable optimum its landscape
            contradicts -- an upper bound above the landscape's own maximum.
            Refused here rather than folded into every stored regret.
    """
    landscape = task.landscape()
    optimum = landscape.optimum
    # A multi-objective landscape's optimum is an ideal point, and the maximum
    # over its components is not a target anything could reach. The attainable
    # declaration is single-objective by construction, so it is simply absent
    # there and the multi-objective branch of `_scores` never consults it.
    attainable = (
        task.attainable_optimum(float(np.max(optimum)))
        if optimum is not None and landscape.n_objectives == 1
        else None
    )
    ran = 0

    for name, method in methods.items():
        outstanding = store.missing(task.name, name, seeds)
        if not outstanding:
            report(f"  {task.name}/{name}: {len(seeds)} seeds cached")
            continue
        started = time.perf_counter()
        for seed in outstanding:
            # Both clocks start before the methodology is built, not before
            # `run`. Fitting a surrogate over an exhaustive library is part of
            # what a method costs, and timing only the loop would credit MLDE
            # for the expensive half of its own work.
            cpu_started = time.process_time()
            wall_started = time.perf_counter()
            campaign = method(task, seed)
            try:
                result = campaign.run()
            except RuntimeError as exc:
                # An arm that cannot fill its plate raises rather than quietly
                # measuring fewer designs. On a sparse feasible set that is a
                # property of the method -- rejection sampling stalls where
                # masking is free -- so the failure is the finding, and it is
                # *stored*. Propagating it would discard every arm the sweep had
                # already finished; storing nothing, which is what this used to
                # do, deleted the finding instead. The arm then vanished from the
                # store, and an empty cell in a table is an absence a reader may
                # fill in as they like -- including as the sharpest result on it
                # -- while a re-run reproduces the absence and never the
                # evidence.
                report(f"  {task.name}/{name}: seed {seed} exhausted -- {exc}")
                store.append(
                    _exhausted_record(
                        store,
                        task=task,
                        method=method,
                        name=name,
                        seed=seed,
                        campaign=campaign,
                        cpu_seconds=time.process_time() - cpu_started,
                        wall_seconds=time.perf_counter() - wall_started,
                    )
                )
                # Counted as run, because it was: it cost the campaign's time and
                # it is now held, so the next sweep will not run it again. A
                # count that skipped it would report zero campaigns run on a pass
                # that wrote records, which is the one number this return value
                # is read for.
                ran += 1
                continue
            cpu_seconds = time.process_time() - cpu_started
            wall_seconds = time.perf_counter() - wall_started
            method_sampler = campaign.sampler
            feasible = (
                float(landscape.is_feasible(result.sequences).mean())
                if len(result.sequences)
                else 0.0
            )
            store.append(
                store.stamp(
                    # Declaring entry points is what makes the fingerprint pay: a
                    # record then goes stale only when something it can
                    # actually reach changed, instead of when any package did.
                    # Without this the mechanism is correct and useless:
                    # adding an unrelated file invalidates the whole store.
                    depends_on=RESULT_DEPENDENCIES,
                    task=task.name,
                    method=name,
                    seed=seed,
                    # The task's repr, not the protocol's: rounds and batch size
                    # alone do not say what a run could have reached, and two
                    # records at 4x96=384 that differ in search radius or in
                    # whether the anchor moved are not comparable.
                    protocol=repr(task),
                    # By attribute, like the sampler quantities below, and taken
                    # against *this* task: one arm's width is resolved from the
                    # task's own shape. See `_arm_parameters`.
                    parameters=_arm_parameters(method, task),
                    **_scores(task, result, attainable),
                    diversity=(diversity(result.sequences) if len(result.sequences) > 1 else 0.0),
                    feasible_fraction=feasible,
                    oracle_calls=result.oracle_calls,
                    proposals=result.proposals,
                    # The sampler's own accounting, shared with the exhausted
                    # path so the two cannot drift into carrying different
                    # columns. `campaign.sampler` is the sampler the run
                    # *finished* with, which under re-anchoring is not the one it
                    # started from.
                    **_sampler_fields(method_sampler),
                    # Processor time is what the arms are compared on; elapsed
                    # time is kept beside it only so a reader can see how hard
                    # the machine was contended when this ran. A suite is
                    # sharded across a dozen processes, so the two diverge by a
                    # factor that says nothing about any method.
                    cpu_seconds=cpu_seconds,
                    wall_seconds=wall_seconds,
                    # By attribute, like the sampler quantities above, and for
                    # the same reason: this lands from the campaign rather than
                    # from here, and a run against a campaign that does not yet
                    # report it stores zero instead of failing. Zero is also the
                    # honest reading for a sampler that cannot repeat itself.
                    duplicate_fraction=float(getattr(result, "duplicate_fraction", 0.0)),
                    deterministic=is_deterministic(),
                    top_sequences=_top_designs(result),
                    trace=result.trace(),
                    rounds=round_rows(result.rounds),
                )
            )
            ran += 1
        elapsed = time.perf_counter() - started
        report(
            f"  {task.name}/{name}: ran {len(outstanding)} "
            f"({elapsed / max(len(outstanding), 1):.1f}s each), "
            f"{len(seeds) - len(outstanding)} cached"
        )
    return ran


def run_tier(
    tier: Tier,
    methods: Mapping[str, Methodology],
    store: ResultStore,
    *,
    report: Callable[[str], None] = print,
    omit: Callable[[Task], Mapping[str, str]] | None = None,
) -> int:
    """Run every task in a tier.

    Args:
        tier: What to run.
        methods: Methodologies by name.
        store: Where results go.
        report: Where progress lines go.
        omit: Given a task, the arms on it whose campaign would repeat another
            arm's -- mapped to the arm it repeats -- or ``None`` to run the full
            cross. A tier is normally that cross, and this is the one departure
            from it that is a property of the *pair* rather than of the tier: an
            arm can be measurable on one of a tier's tasks and a duplicate of its
            neighbour on the next, so the decision cannot be taken by dropping the
            arm from the tier. What is omitted is said on the line where it would
            otherwise have been run, because a campaign that silently does not
            happen is indistinguishable from one nobody asked for.

    Returns:
        How many campaigns were actually run. Omitted pairs are not counted: they
        cost nothing and, unlike an exhausted campaign, nothing is now held for
        them.
    """
    report(f"{tier!r}")
    ran = 0
    for task in tier.tasks:
        repeats = {} if omit is None else {k: v for k, v in omit(task).items() if k in methods}
        for name, source in sorted(repeats.items()):
            report(
                f"  {task.name}/{name}: not run -- it is {source}'s own campaign on this task, "
                f"reproduced at report time rather than measured twice"
            )
        runnable = {name: m for name, m in methods.items() if name not in repeats}
        ran += run_task(task, runnable, store, tier.seeds, report=report)
    return ran


def run_anchor_study(
    store: ResultStore,
    seeds: Sequence[int],
    *,
    report: Callable[[str], None] = print,
) -> int:
    """Run whatever the anchor study is missing, one cell at a time.

    Cell by cell rather than by tier, and that is the whole content of this
    function: a tier crosses its tasks with its arms, and the cross of this
    study's two tasks and three arms contains a cell that is already stored and
    a cell that is a second name for its neighbour. Running the cross would pay
    for both and leave a reader unable to tell which of two identically-valued
    rows was the real one.

    Args:
        store: Where results go, and what says which cells are already held.
        seeds: Seeds per cell.
        report: Where progress lines go.

    Returns:
        How many campaigns were actually run.
    """
    from evogfn.benchmark.methods import anchor_arms  # noqa: PLC0415 - the arms import tasks

    arms = anchor_arms()
    ran = 0
    for cell in anchor_study():
        # One tier per cell so the purpose travels with it: these results
        # explain a mechanism and are never a row in the results table.
        tier = Tier(f"anchor:{cell!r}", (cell.task,), tuple(seeds), Purpose.DIAGNOSTIC)
        ran += run_tier(tier, {cell.arm: arms[cell.arm]}, store, report=report)
    return ran


def run_rejection_curve(
    store: ResultStore,
    seeds: Sequence[int],
    *,
    report: Callable[[str], None] = print,
) -> int:
    """Run whatever the rejection curve is missing, with the arms it needs.

    The tier and its arms are paired here rather than left to a caller, and that
    pairing is the whole content of this function. Only an arm that *rejects*
    has draws to report, so this family run against the suite's default
    methodologies would produce a full grid of records whose ``draws_attempted``
    is zero -- and a column of zeros reads as "nothing was refused" rather than
    as "nothing ran a rejection loop". `constraint_density` warns about the same
    failure in prose; this refuses it in code.

    Args:
        store: Where results go, and what says which cells are already held.
        seeds: Seeds per cell.
        report: Where progress lines go.

    Returns:
        How many campaigns were actually run, counting those that exhausted --
        which on the widest rungs of the matched-kernel column is expected to be
        most of them, and which is a measurement rather than a loss: the
        counters the curve is read on are stored either way.
    """
    return run_tier(rejection_tier(seeds), rejection_arms(), store, report=report)


def records_to_metric(
    records: Mapping[int, RunRecord], seeds: Sequence[int], metric: str
) -> np.ndarray:
    """Pull one metric out of stored records, in seed order.

    **Exhausted records are excluded, and the exclusion is the point.** Every
    caller of this averages what comes back, and a record from a campaign that
    never finished has no measurement to contribute: its ``best`` and
    ``diversity`` are ``nan`` by construction, so leaving it in would turn the
    whole arm's mean into ``nan`` while the seed count went on claiming a full
    row. Both readings are wrong, and the second is wrong silently.

    What the exclusion costs is that the failure is now *invisible here*, which
    is exactly what it was before this function had anything to exclude. So a
    caller that reports a mean must also report how many seeds it left out --
    `RunRecord.exhausted` is stored
    per seed for that purpose, and ``experiments/run_suite.py`` prints the count
    beside the row. A shorter array with nothing said about why is the absence
    this whole mechanism exists to end.

    **Not the reader for**
    `RunRecord.fitted`. Every field this
    returns is a quantity to be averaged, and ``None`` is mapped to ``nan``
    because for ``regret`` it means "no audit covers this task" -- an absence
    that should propagate. ``fitted`` is tri-valued in a different way: ``None``
    there means the arm fits nothing, so a column mixing classical and supervised
    arms would come back all-``nan`` and the mean would say nothing rather than
    saying "no arm fitted". `fit_status` is the reader for it.

    Args:
        records: Records by seed.
        seeds: The order to return them in.
        metric: Field name.

    Returns:
        An array with one entry per seed present that carries a completed
        campaign.
    """
    values = []
    for seed in seeds:
        record = records.get(seed)
        if record is None or record.exhausted:
            continue
        value = getattr(record, metric)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def fit_status(records: Mapping[int, RunRecord], seeds: Sequence[int]) -> tuple[int, int]:
    """How many of an arm's stored campaigns fit a model, and how many never did.

    The counterpart to the ``exhausted`` count beside a mean: both are facts
    about seeds that a column of averages cannot carry, and both change what the
    row on that column *means* rather than what it says.

    Exhausted records are **counted here**, unlike in `records_to_metric`, and
    the difference is what each number is for. That drops them because it is
    building a mean and they contribute no measurement. This is not building a
    mean -- it is asking whether the arm ever got as far as its own model -- and
    a campaign that exhausted in round one having never fitted is the sharpest
    instance of exactly that, not a case to leave out.

    Args:
        records: Records by seed.
        seeds: The seeds to look at.

    Returns:
        ``(reported, never)``: how many stored campaigns said anything about a
        model fit at all, and how many of those said the model was never fitted.
        ``reported`` at zero means this arm fits nothing -- or predates the field
        -- and ``never`` is then meaningless rather than reassuring, which is why
        the two are returned together and not as a bare count.
    """
    held = [record for seed in seeds if (record := records.get(seed)) is not None]
    reported = [record for record in held if record.fitted is not None]
    return len(reported), sum(1 for record in reported if not record.fitted)


def _top_designs(result: CampaignResult, k: int = 10) -> list[list[int]]:
    """The best designs a campaign found, for inspection.

    Storing every measured sequence would be hundreds of megabytes across a
    suite; storing the best ten is what anyone actually looks at when a number
    surprises them.

    Args:
        result: A completed campaign.
        k: How many to keep.

    Returns:
        Token lists, best first.
    """
    if not len(result.sequences):
        return []
    values = np.asarray(result.values, dtype=np.float64).reshape(len(result.sequences), -1)
    order = np.argsort(-values.max(axis=1))[:k]
    return [[int(t) for t in result.sequences[i]] for i in order]
