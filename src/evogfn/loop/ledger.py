"""What a campaign spent, and on what.

The oracle budget is the constraint every result is indexed by. Six counts are
kept separate because they diverge, and the gaps are what they are read for:

* **proposals** -- candidates the sampler generated. Free. A rejection-sampling
  baseline under a feasibility constraint can generate ten times what it keeps,
  and reporting only oracle calls would hide that entirely.
* **screened** -- proposals that survived the campaign's memory of earlier
  rounds and reached the selector. A sampler that has collapsed onto one mode
  re-proposes what it has already measured, and the gap from proposals to
  screened is where that shows.
* **redundant** -- of the proposals that did *not* survive, how many were
  refused because an earlier round had already measured that design. This is
  the named half of the proposals-to-screened gap; the rest of that gap is
  ``distinct_batch`` dropping a repeat off the same plate, and until the two
  were counted apart neither could be read off the other.
* **evaluated** -- oracle calls charged. The constrained resource.
* **duplicates** -- of those evaluated, how many repeated a design already on
  the same plate. Charged like any other well, because they are.
* **feasible** -- of those evaluated, how many the landscape could actually
  build.

``duplicates`` and ``redundant`` are twins and are never summed.
**``duplicates`` is repetition within one plate**, charged in wells: 96
offspring of which 10 are identical consume 96 wells for 86 distinct data
points. **``redundant`` is repetition against the campaign's memory across
rounds**, charged in proposals and in nothing else, because the candidate never
reached a well. A converged sampler produces both at once, so reading either
alone misattributes the other's cost.

Infeasible designs are charged: they cost the same to synthesise as feasible
ones, so a method that proposes them has spent the budget.

What is reported with more than one objective
---------------------------------------------

There is no scalar "best value" over objectives measured on different scales,
so [CampaignResult.best_value][evogfn.loop.ledger.CampaignResult.best_value]
raises and
[simple_regret][evogfn.loop.ledger.CampaignResult.simple_regret] is ``None``.
What replaces them are set indicators, computed on the objective vectors that
were actually measured:

* [hypervolume][evogfn.loop.ledger.CampaignResult.hypervolume] -- the volume the
  measured designs dominate above a **reference point**, which must be supplied.
  Monotone in Pareto dominance, so a run that dominates another always scores
  higher.
* [igd_plus][evogfn.loop.ledger.CampaignResult.igd_plus] -- how far a supplied
  reference front is from being covered, which is where hypervolume is weak: a
  single excellent design can enclose a large volume while covering almost none
  of the front.

Both are ``None`` rather than zero when the campaign was not given what they
need, because zero is a *result* -- "found nothing above the reference" -- and
"was never told where the reference is" is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from evogfn.metrics import pareto

if TYPE_CHECKING:
    from evogfn.core.types import Fitness, Tokens

# An objective matrix is two-dimensional by definition.
_MATRIX_NDIM = 2


@dataclass(frozen=True)
class RoundRecord:
    """What one design-build-test-learn round did.

    Attributes:
        index: Zero-based round number. Round 0 is the initial design, which is
            charged like any other -- seed data is not free.
        proposed: Candidates the sampler generated, across every proposal call
            the round needed to fill its plate.
        screened: Proposals that reached the selector -- those the campaign had
            not already measured in an earlier round.
        evaluated: Oracle calls charged this round.
        feasible: How many of the evaluated candidates were constructible.
        duplicates: How many of the evaluated candidates repeated a design
            already on the same plate -- the wells cost of convergence. The
            **within a plate** twin of ``redundant``; the module docstring says
            why the two are never summed.
        redundant: How many proposals the campaign refused because it had
            measured that design in an *earlier round*. The **across rounds**
            twin of ``duplicates``: this repetition costs proposals and no
            wells, because the candidate never reached one.

            ``None`` when the campaign has no cross-round memory to consult
            (``skip_measured`` off), which is not the same as having consulted
            it and refused nothing. Zero would conflate the two.
        best_in_round: Best objective value measured this round. With more than
            one objective this is the best *scalarised* value, under the
            trade-off the campaign's acquisition rule states -- the same one the
            surrogate was fitted to, so the three numbers cannot drift apart.
        best_so_far: Best measured since the campaign began, on the same scale.
        mean_in_round: Mean objective value over the feasible measurements.
        hypervolume: Volume dominated by everything measured up to and including
            this round, against the campaign's reference point. ``nan`` when no
            reference point was supplied, when the campaign is single-objective,
            or when the front outgrew the exact method in
            [evogfn.metrics.pareto][] -- a ``nan`` here is always "not computed"
            and never "no volume", which is a real result and would be ``0.0``.
        batch_diversity: Mean pairwise Hamming distance within the batch.
        surrogate_correlation: Pearson correlation between what the surrogate
            predicted for this batch and what the oracle measured. ``nan``
            before a surrogate exists. Separates a method failing because its
            sampler proposes badly from one failing because its model cannot
            tell good designs from bad.
        fitness: Every measured value this round, in the order the plate was
            evaluated, as a tuple. The aggregates beside it -- ``best_in_round``,
            ``mean_in_round`` -- are functions of this, and storing only them is
            what made a whole class of batch-level question unanswerable without
            re-running the campaign: what the marginal contribution of the k-th
            well to ``E[max]`` is, where a plate saturates, what the measured
            distribution over a plate looks like. Infeasible measurements appear
            as ``-inf``, the same value they carry in the aggregates, so the
            feasible count is recoverable from this alone.

            Cheap in a way the batch *sequences* are not: one float per well
            against ``length`` integers per well, which is why this is recorded
            and they are not.
        anchor: The design this round's proposals were built from, as a tuple of
            token indices, or ``None`` when the campaign was not tracking an
            anchor. A tuple rather than an array so the record stays comparable,
            hashable and serialisable like every other field.
        anchor_distance: Hamming distance from the campaign's original wild type
            to ``anchor``. Zero in every round under a fixed anchor, where the
            campaign can never reach a design further than one round's mutation
            budget from the wild type however many rounds it runs. Read down the
            rounds it is the cumulative distance travelled.
    """

    index: int
    proposed: int
    screened: int
    evaluated: int
    feasible: int
    best_in_round: float
    best_so_far: float
    mean_in_round: float
    batch_diversity: float
    fitness: tuple[float, ...] = ()
    surrogate_correlation: float = float("nan")
    hypervolume: float = float("nan")
    anchor: tuple[int, ...] | None = None
    anchor_distance: int = 0
    duplicates: int = 0
    redundant: int | None = None

    @property
    def feasible_fraction(self) -> float:
        """Share of the round's oracle calls spent on constructible designs."""
        return self.feasible / self.evaluated if self.evaluated else 0.0

    @property
    def duplicate_fraction(self) -> float:
        """Share of the round's oracle calls spent re-measuring its own plate.

        Zero means every well held a design the round had not already put on
        that plate. It rises as a method converges.
        """
        return self.duplicates / self.evaluated if self.evaluated else 0.0

    @property
    def redundant_fraction(self) -> float:
        """Share of the round's *proposals* the campaign's memory refused.

        Denominated in proposals where
        [duplicate_fraction][evogfn.loop.ledger.RoundRecord.duplicate_fraction]
        is denominated in wells: a duplicate consumed a well, a redundant
        candidate was dropped before selection. The two shares are over
        different denominators and must not be added.

        Returns:
            The share in ``[0, 1]``, or ``nan`` when the campaign had no memory
            to consult, which is not the same as having consulted it and
            refused nothing.
        """
        if self.redundant is None or not self.proposed:
            return float("nan")
        return self.redundant / self.proposed

    @property
    def rejection_ratio(self) -> float:
        """Proposals generated per oracle call charged.

        One means the sampler proposed exactly what was measured. Large values
        mean it is discarding most of its own output, which is free in compute
        terms and worth seeing.
        """
        return self.proposed / self.evaluated if self.evaluated else float("inf")


@dataclass(frozen=True)
class CampaignResult:
    """Everything a campaign measured, and the ledger of how it was spent.

    Attributes:
        sampler: Name of the sampler under test.
        rounds: One record per completed round, in order.
        sequences: Every sequence evaluated, in evaluation order.
        values: Their objective values, aligned with ``sequences``, shaped
            ``(n, n_objectives)``.
        optimum: The landscape's best attainable value, when it knows it and
            when there is one objective for it to be the best of. ``None`` for a
            multi-objective campaign, where the landscape's optimum is an
            *ideal point* no design need attain -- see ``ideal_point``.
        ideal_point: The best attainable value on each objective separately,
            when the landscape knows it. Not a design: on CH65 its three
            components belong to three different variants, which is why the gap
            to it is not a regret.
        reference_point: The ``(n_objectives,)`` point hypervolume is measured
            from. ``None`` leaves hypervolume unreported rather than inventing
            one, because two hypervolumes taken from different reference points
            are not comparable and nothing in either number says so.
        reference_front: A ``(m, n_objectives)`` front to measure IGD+ against
            -- the landscape's true front where it is known, or the
            non-dominated union of everything being compared where it is not.
    """

    sampler: str
    rounds: tuple[RoundRecord, ...]
    sequences: Tokens
    values: Fitness
    optimum: float | None = None
    ideal_point: Fitness | None = None
    reference_point: Fitness | None = None
    reference_front: Fitness | None = None

    @property
    def n_objectives(self) -> int:
        """How many objectives every measurement carries."""
        array = np.asarray(self.values)
        return int(array.shape[1]) if array.ndim == _MATRIX_NDIM else 1

    @property
    def is_multi_objective(self) -> bool:
        """Whether the campaign measured more than one objective per design."""
        return self.n_objectives > 1

    @property
    def oracle_calls(self) -> int:
        """Total oracle calls charged -- the number every claim is indexed by."""
        return sum(record.evaluated for record in self.rounds)

    @property
    def proposals(self) -> int:
        """Total candidates generated, including those never evaluated."""
        return sum(record.proposed for record in self.rounds)

    @property
    def best_value(self) -> float:
        """Best objective value measured, or ``-inf`` if nothing was feasible.

        Returns:
            The largest finite measurement.

        Raises:
            ValueError: If the campaign measured more than one objective, where
                the maximum over designs *and* objectives mixes scales and rises
                whenever any objective rises. Use
                [hypervolume][evogfn.loop.ledger.CampaignResult.hypervolume],
                [igd_plus][evogfn.loop.ledger.CampaignResult.igd_plus], or
                ``trace()`` for the scalarised best-so-far the acquisition rule
                actually optimised.
        """
        if self.is_multi_objective:
            raise ValueError(
                f"this campaign measured {self.n_objectives} objectives, and there is no "
                f"single best value among them; report hypervolume against a reference "
                f"point, or IGD+ against a reference front, instead"
            )
        finite = self.values[np.isfinite(self.values)]
        return float(finite.max()) if finite.size else float("-inf")

    @property
    def feasible_fraction(self) -> float:
        """Share of the whole budget spent on constructible designs."""
        calls = self.oracle_calls
        return sum(r.feasible for r in self.rounds) / calls if calls else 0.0

    @property
    def duplicate_fraction(self) -> float:
        """Mean over rounds of the wells each spent repeating its own plate.

        Averaged over rounds rather than pooled over wells, so the number reads
        as a typical plate for this method rather than as a total that grows
        with the budget.

        Returns:
            A share in ``[0, 1]``, and ``0.0`` for a campaign with no rounds.
        """
        if not self.rounds:
            return 0.0
        return float(np.mean([record.duplicate_fraction for record in self.rounds]))

    @property
    def pareto_front(self) -> Fitness:
        """The measured objective vectors no other measurement dominates.

        Returns:
            A ``(k, n_objectives)`` array, in measurement order. Empty when
            nothing was measured.

        Raises:
            ValueError: If any measurement is ``nan``, where dominance between a
                measured design and an unmeasured one is undefined.
        """
        values = np.asarray(self.values, dtype=np.float64)
        if values.size == 0:
            return values.reshape(0, self.n_objectives)
        return pareto.pareto_front(values)

    @property
    def hypervolume(self) -> float | None:
        r"""Volume the measured designs dominate above ``reference_point``.

        The reference point is part of the measurement: it is the worst value
        considered acceptable on each objective, designs failing to beat it on
        every objective contribute nothing, and moving it rescales every number
        computed against it. Hypervolumes taken from different reference points
        are not comparable, so this is ``None`` unless one was supplied and the
        point used is kept on the result.

        Setting it *too low* is not the safe choice: push it far below the data
        and every method earns a large constant box, so differences between
        methods shrink into that constant's noise. For CH65 the defensible
        choice is the Tite-Seq detection floor, ``(6, 6, 6)`` -- affinities
        there are $-\log_{10} K_D$ and 6.0 is where the assay stops resolving.

        Returns:
            The dominated volume, or ``None`` when no reference point was
            supplied.

        Raises:
            ValueError: If the reference point does not match the objectives.
            NotImplementedError: If three or more objectives are combined with a
                front larger than the exact method in [evogfn.metrics.pareto][]
                accepts. An approximation reported in the same field as an exact
                value would be indistinguishable from one.
        """
        if self.reference_point is None or np.asarray(self.values).size == 0:
            return None
        return pareto.hypervolume(np.asarray(self.values, dtype=np.float64), self.reference_point)

    @property
    def igd_plus(self) -> float | None:
        """How far ``reference_front`` is from being covered by what was measured.

        **Lower is better.** Reported alongside hypervolume, which rewards a
        single design enclosing a large box; this notices that the rest of the
        front was never approached.

        Returns:
            The IGD+ indicator, or ``None`` when no reference front was supplied
            or nothing was measured.

        Raises:
            ValueError: If the reference front does not describe the same
                objectives, or is not finite.
        """
        if self.reference_front is None or np.asarray(self.values).size == 0:
            return None
        return pareto.igd_plus(np.asarray(self.values, dtype=np.float64), self.reference_front)

    @property
    def simple_regret(self) -> float | None:
        """Distance from the best measurement to the true optimum.

        Returns:
            ``optimum - best_value``; ``None`` when the landscape does not know
            its optimum -- the honest answer for a real assay -- and ``None``
            with more than one objective, where neither term exists. The gap to
            a multi-objective landscape's ideal point is not a regret: no design
            attains the ideal point, so the gap never reaches zero and a run
            cannot be scored against it.
        """
        if self.optimum is None or self.is_multi_objective:
            return None
        return self.optimum - self.best_value

    def trace(self) -> list[float]:
        """Best-so-far after each round, for plotting a budget curve.

        With more than one objective this is the best *scalarised* value, on the
        trade-off the acquisition rule states.
        """
        return [record.best_so_far for record in self.rounds]

    def anchor_trace(self) -> list[int]:
        """How far each round's starting point sat from the original wild type.

        Flat at zero means the campaign searched one fixed Hamming ball for its
        whole life, so nothing further than a single round's mutation budget was
        ever reachable. Growing past that budget is the property re-anchoring
        exists to produce.

        Returns:
            One Hamming distance per completed round, in order.
        """
        return [record.anchor_distance for record in self.rounds]

    def hypervolume_trace(self) -> list[float]:
        """Hypervolume after each round, for plotting a budget curve of the set.

        Non-decreasing by construction -- measuring more can only add dominated
        volume -- so a flat stretch is a run that spent a round learning nothing
        about the front, which is exactly what the curve is read for.

        Returns:
            One value per completed round, ``nan`` where it could not be
            computed.
        """
        return [record.hypervolume for record in self.rounds]

    def summary(self) -> dict[str, float]:
        """Flat metrics for logging, keyed for a tracker.

        Returns:
            The budget counters always; ``best_value`` and ``simple_regret``
            only where they are defined; ``hypervolume`` and ``igd_plus`` only
            where a reference was supplied. A key is absent rather than ``nan``
            so that a missing indicator cannot be averaged into a table.
        """
        metrics = {
            "oracle_calls": float(self.oracle_calls),
            "proposals": float(self.proposals),
        }
        if not self.is_multi_objective:
            metrics["best_value"] = self.best_value
        metrics["feasible_fraction"] = self.feasible_fraction
        metrics["duplicate_fraction"] = self.duplicate_fraction
        metrics["rounds"] = float(len(self.rounds))
        if (regret := self.simple_regret) is not None:
            metrics["simple_regret"] = regret
        if (volume := self.hypervolume) is not None:
            metrics["hypervolume"] = volume
        if (coverage := self.igd_plus) is not None:
            metrics["igd_plus"] = coverage
        return metrics
