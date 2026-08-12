r"""What a benchmark task is: a landscape, a protocol, and a reason to run it.

A benchmark is not a landscape and a number. It is a set of *tests*, each chosen
because it can settle a question the others cannot, run under a protocol a wet
lab would recognise. This module holds only the shape of one such test; the
tests themselves live in [evogfn.benchmark.suite][].

That separation is the point, and it is enforced by there being nothing else
here. A second list of tasks alongside the one the results come from is not
redundant, it is wrong: two definitions of "the feasibility task" that differ in
sequence length are two different experiments wearing one name, and nothing in
either definition would say which had produced a given number.

The one field a task cannot omit is `Task.purpose`. A suite is only as
good as its ability to distinguish methods, so a row that cannot say what it
decides that the others do not should be deleted rather than kept for
completeness.

The radius is per *round*, and what a task can reach is a measurement
-------------------------------------------------------------------

`Task.max_mutations` bounds one round, not one campaign. Under
`Task.reanchor` the campaign moves its anchor to the best design measured so
far, so $R$ rounds of $b$ substitutions reach $R \cdot b$ from the wild type --
which is what directed evolution does, and which is the difference between a
planted optimum being reachable in principle and not.

That makes "what is the best value this task's search space contains" a
question with a measured answer rather than an assumed one, and
`Task.attainable` carries it. It is not the landscape's optimum: where a
feasibility constraint holds the constructible set below the nominal maximum,
the gap between the two is a constant added to every regret on that task and
attributable to no method. A task that has not been audited says so by leaving
`attainable` unset, and then no regret is stored for it at all -- an absent
number being far safer than one taken against an unreachable target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from evogfn.benchmark.attainable import AttainableOptimum
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.landscapes.gb1 import GB1Landscape
from evogfn.landscapes.trpb import TrpBLandscape

if TYPE_CHECKING:
    from collections.abc import Callable

    from evogfn.benchmark.protocol import Protocol
    from evogfn.core.types import Tokens
    from evogfn.env.feasibility import FeasibilityPredicate
    from evogfn.landscapes.base import FitnessLandscape

#: Slack for comparing a declared bound against a landscape's own optimum.
_DECLARATION_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class Attainable:
    """What an audit measured a task's search space to actually contain.

    A *declaration*, not a computation:
    [attainable_optimum][evogfn.benchmark.attainable.attainable_optimum] and
    [reanchored_attainable][evogfn.benchmark.attainable.reanchored_attainable]
    take minutes per task on the larger landscapes, and a suite that recomputed
    them on import would be unusable. So the audit is run by
    ``experiments/audit_optima.py`` and its answers are written down here, the
    same way the mutation budgets are: a number a reader can check without
    instantiating a landscape, and a test that re-derives it rather than
    trusting it.

    The two-value form is not decoration. Where the reachable set can be
    enumerated the attainable optimum is *known*; where it cannot, the audit
    returns a certified upper bound and a searched lower bound, and collapsing
    that bracket to its midpoint -- or to either end -- would turn an honest
    interval into a fact. `lower` is the conservative end and the one regret is
    stored against, because it is the only one witnessed by a design that was
    actually constructed.

    Attributes:
        lower: Best value a construction was found for, or ``None`` to mean the
            landscape's own optimum -- the audited claim that nothing is out of
            reach, which is true only where it has been checked.
        upper: Certified upper bound, or ``None`` for the landscape's optimum.
        source: How the numbers were obtained, in words a report can quote. A
            bound whose provenance is not written down cannot be re-checked, and
            an unre-checkable bound is what this whole mechanism exists to
            replace.

    Raises:
        ValueError: If only one end is given, if `lower` exceeds `upper`, or if
            `source` is empty.
    """

    lower: float | None
    upper: float | None
    source: str

    def __post_init__(self) -> None:
        """Reject a declaration that cannot describe a measurement."""
        if (self.lower is None) != (self.upper is None):
            raise ValueError(
                "an attainable optimum is declared as both ends of an interval or as neither; "
                f"got lower={self.lower}, upper={self.upper}"
            )
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(
                f"declared attainable lower {self.lower} exceeds upper {self.upper}; "
                f"the audit that produced them disagrees with itself"
            )
        if not self.source.strip():
            raise ValueError("an attainable optimum must say how it was measured")

    @classmethod
    def exactly(cls, value: float, source: str) -> Attainable:
        """Declare an attainable optimum that was measured rather than bracketed.

        Args:
            value: The attainable optimum.
            source: How it was measured.

        Returns:
            The declaration, with both ends of the interval closed.
        """
        return cls(lower=value, upper=value, source=source)

    @classmethod
    def between(cls, lower: float, upper: float, source: str) -> Attainable:
        """Declare a bracket, where the audit could not close it.

        Args:
            lower: Best value a construction was found for.
            upper: Certified upper bound.
            source: How both were obtained.

        Returns:
            The declaration.
        """
        return cls(lower=lower, upper=upper, source=source)

    @classmethod
    def whole_optimum(cls, source: str) -> Attainable:
        """Declare that the landscape's own optimum is reachable.

        The right declaration for a task whose mutation budget imposes no
        restriction at all -- GB1's four sites against a budget of four -- where
        the attainable optimum and the nominal one coincide and the regret floor
        is genuinely zero.

        Args:
            source: Why nothing is out of reach.

        Returns:
            The declaration, deferring both ends to the landscape.
        """
        return cls(lower=None, upper=None, source=source)

    def resolve(self, *, task: str, budget: int, nominal: float) -> AttainableOptimum:
        """Turn the declaration into the audited quantity the report reads.

        Args:
            task: Task name, carried so a result cannot drift from what it
                describes.
            budget: Substitutions the campaign can reach cumulatively.
            nominal: The landscape's own optimum, which is what makes the regret
                floor a floor.

        Returns:
            The attainable optimum, exact where the declared interval has
            closed and bracketed where it has not.

        Raises:
            ValueError: If the declared upper bound exceeds the landscape's own
                optimum, which would claim a design scoring above the maximum.
        """
        lower = nominal if self.lower is None else self.lower
        upper = nominal if self.upper is None else self.upper
        if upper > nominal + _DECLARATION_TOLERANCE:
            raise ValueError(
                f"{task}: declared attainable upper bound {upper} is above the landscape's own "
                f"optimum {nominal}; no design can score above the maximum"
            )
        return AttainableOptimum(
            task=task,
            budget=budget,
            nominal=nominal,
            lower=lower,
            upper=upper,
            exact=lower if lower == upper else None,
            method=self.source,
        )


@dataclass(frozen=True, slots=True)
class Task:
    """One benchmark test: a landscape, a protocol, and a reason to run it.

    Attributes:
        name: Short identifier.
        purpose: What this task can decide that the others cannot. Present so a
            suite cannot silently accumulate rows that measure the same thing.
        build: Makes the landscape. A factory rather than an instance, so each
            seed can draw its own instance where that is meaningful.
        protocol: Rounds and batch size.
        max_mutations: How far a design may stray from the parent **in one
            round**. Under `reanchor` the campaign's cumulative reach is this
            times the number of rounds; see `search_budget`.
        reanchor: Whether the campaign moves its anchor to the best design
            measured so far at the end of every round. Off means the task
            searches one Hamming ball around the wild type for its whole life,
            however many rounds it runs, which on every Ehrlich landscape here
            puts a reward of 1.0 outside the search space by construction.
        attainable: What an audit found this task's search space to contain,
            or ``None`` where none has been run. ``None`` is not a neutral
            default: it makes `run_task` store no regret at all, on the grounds
            that a regret against an unaudited optimum is the failure this field
            exists to prevent.
        feasibility: Builds the rule saying which sequences are legal, from the
            landscape. ``None`` means the landscape's own transition matrix,
            which is what every task declared before predicates existed and
            what all of them still declare.

            A factory rather than a predicate, because a predicate is sized to
            the alphabet and the sequence length and neither is known until the
            landscape is built -- and a task is a *declaration*, built once and
            reused across every seed, while a landscape may be drawn per seed.
            Declaring the instance here would pin one draw's shape onto every
            other.
        parent_seed: Seeds the starting sequence.
    """

    name: str
    purpose: str
    build: Callable[[], FitnessLandscape]
    protocol: Protocol
    max_mutations: int
    reanchor: bool = False
    attainable: Attainable | None = None
    feasibility: Callable[[FitnessLandscape], FeasibilityPredicate] | None = None
    parent_seed: int = 0

    def landscape(self) -> FitnessLandscape:
        """Build this task's landscape."""
        return self.build()

    @property
    def search_budget(self) -> int:
        """Substitutions from the wild type this campaign can reach in total.

        Returns:
            ``max_mutations * rounds`` under re-anchoring, since each round
            starts from the last one's best design and the distances add, and
            ``max_mutations`` without it -- where the anchor never moves and the
            extra rounds re-search the ball the first one already covered.
        """
        return self.max_mutations * self.protocol.rounds if self.reanchor else self.max_mutations

    def attainable_optimum(self, nominal: float) -> AttainableOptimum | None:
        """What this task can reach, as the audited quantity a report reads.

        Args:
            nominal: The landscape's own optimum, which is the target regret
                would be taken against if nobody checked it was available.

        Returns:
            The attainable optimum, or ``None`` for a task no audit has covered.

        Raises:
            ValueError: If the declaration is inconsistent with `nominal`.
        """
        if self.attainable is None:
            return None
        return self.attainable.resolve(task=self.name, budget=self.search_budget, nominal=nominal)

    def parent(self, landscape: FitnessLandscape) -> Tokens:
        """The wild type a campaign starts from.

        On a landscape with a feasibility constraint this must be a
        constructible sequence: an infeasible parent scores minus infinity and
        leaves a mutation-based sampler with nothing to climb from.

        The empirical four-site landscapes -- GB1 and TrpB -- each publish their
        own wild type, and it is the only defensible anchor for them: both
        datasets normalise fitness so that the wild type is 1.0, so a campaign
        anchored anywhere else would report improvements against a different
        baseline than the one the assay was calibrated on.

        Args:
            landscape: The landscape being searched.

        Returns:
            A starting sequence of the landscape's length.

        Raises:
            TypeError: If no wild type is defined for this landscape, rather
                than silently anchoring at an arbitrary sequence.
        """
        if isinstance(landscape, EhrlichLandscape):
            return landscape.feasible_sequence(self.parent_seed)
        if isinstance(landscape, GB1Landscape | TrpBLandscape):
            return landscape.wild_type
        raise TypeError(f"no wild type defined for {type(landscape).__name__}")

    @property
    def constrains_search(self) -> bool:
        """Whether the mutation budget actually restricts the reachable set."""
        return self.protocol.constrains_search(self.build().sequence_length)

    def __repr__(self) -> str:
        """Name the task, its budget, its search radius and whether the anchor moves.

        All four, because all four decide what the run could have found. This is
        what `run_task` stores as a record's provenance: two campaigns at
        ``4x96=384`` that differ in radius or in re-anchoring are different
        experiments, and a stored record naming only the protocol could not be
        told apart from one that had searched a space thirty times larger.
        """
        anchor = "re-anchored" if self.reanchor else "fixed anchor"
        return f"{self.name} ({self.protocol!r}, {self.max_mutations}/round, {anchor})"
