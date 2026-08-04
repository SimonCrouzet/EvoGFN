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

Two tasks are not like the others, and both for stated reasons. ``gb1-anchor``
has four sites and a budget of four, so its ball is the whole space and there is
nothing for an anchor to move towards. ``feasibility`` keeps a fixed anchor
because what binds there is the transition matrix, not the radius. Leaving the
anchor still is what keeps its attainable optimum an *enumerated* answer rather
than a bracket, which is the difference between reporting a fact and reporting a
search.

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

from evogfn.benchmark.determinism import is_deterministic
from evogfn.benchmark.protocol import PLATE, Protocol, round_sweep
from evogfn.benchmark.tasks import Attainable, Task
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.gb1 import GB1Landscape
from evogfn.metrics.diversity import diversity

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from evogfn.benchmark.attainable import AttainableOptimum
    from evogfn.benchmark.methods import Methodology
    from evogfn.benchmark.store import ResultStore, RunRecord
    from evogfn.landscapes.base import FitnessLandscape
    from evogfn.loop.ledger import CampaignResult

#: GB1's four measured sites. Equal to the sequence length, so every variant in
#: the published table is reachable and the anchor exercises no search radius.
GB1_MUTATIONS = 4

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


def run_task(
    task: Task,
    methods: Mapping[str, Methodology],
    store: ResultStore,
    seeds: Sequence[int],
    *,
    report: Callable[[str], None] = print,
) -> int:
    """Run whatever is missing for one task, storing each result as it lands.

    Args:
        task: What to run.
        methods: Methodologies by name.
        store: Where results go, and what says which are already held.
        seeds: Seeds wanted.
        report: Where progress lines go.

    Returns:
        How many campaigns were actually run.

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
                # masking is free -- so it is reported and the seed is left
                # unstored, which shows up as a short seed count rather than as
                # a number pretending to be comparable. Propagating it would
                # discard every arm the sweep had already finished.
                report(f"  {task.name}/{name}: seed {seed} exhausted -- {exc}")
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
                    # By attribute, like the sampler quantities below: an arm
                    # built by this module declares what it closed over, and a
                    # methodology defined anywhere else is a plain callable that
                    # declares nothing. Empty is then the honest record -- the
                    # settings were not stated, which is not the same as an arm
                    # having none, and the alternative would be inventing a
                    # configuration for a closure nobody can see into.
                    parameters=dict(getattr(method, "parameters", {})),
                    **_scores(task, result, attainable),
                    diversity=(diversity(result.sequences) if len(result.sequences) > 1 else 0.0),
                    feasible_fraction=feasible,
                    oracle_calls=result.oracle_calls,
                    proposals=result.proposals,
                    proxy_calls=int(getattr(method_sampler, "proxy_calls", 0)),
                    # By attribute, like the proxy spend above: a sampler that
                    # breeds nothing simply does not carry these, and asking the
                    # base interface for them would make every baseline declare
                    # a quantity only one method can measure.
                    bred_designs=int(getattr(method_sampler, "bred_designs", 0)),
                    unconstructible_fraction=float(
                        getattr(method_sampler, "unconstructible_fraction", 0.0)
                    ),
                    # Read off the *final* sampler, which is what `campaign.sampler`
                    # returns, and safe under re-anchoring only because the arm that
                    # reports this carries its repair counters across a move rather
                    # than restarting them. An arm that reset them would report its
                    # last round here while looking like it reported the campaign --
                    # the same silent undercount the proxy spend once had.
                    repaired_fraction=float(getattr(method_sampler, "repaired_fraction", 0.0)),
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
                    rounds=[
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
                            # How far this round's anchor sat from the wild
                            # type. Flat at zero says the campaign searched one
                            # Hamming ball for its whole life, which is the
                            # difference between a budget of `max_mutations` and
                            # a budget of `max_mutations` per round.
                            "anchor_distance": float(record.anchor_distance),
                        }
                        for record in result.rounds
                    ],
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
) -> int:
    """Run every task in a tier.

    Args:
        tier: What to run.
        methods: Methodologies by name.
        store: Where results go.
        report: Where progress lines go.

    Returns:
        How many campaigns were actually run.
    """
    report(f"{tier!r}")
    return sum(run_task(task, methods, store, tier.seeds, report=report) for task in tier.tasks)


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


def records_to_metric(
    records: Mapping[int, RunRecord], seeds: Sequence[int], metric: str
) -> np.ndarray:
    """Pull one metric out of stored records, in seed order.

    Args:
        records: Records by seed.
        seeds: The order to return them in.
        metric: Field name.

    Returns:
        An array with one entry per seed present.
    """
    values = []
    for seed in seeds:
        record = records.get(seed)
        if record is None:
            continue
        value = getattr(record, metric)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


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
