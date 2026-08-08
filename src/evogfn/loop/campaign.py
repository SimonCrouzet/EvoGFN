"""The design-build-test-learn round engine.

A campaign is what directed evolution actually is: a few rounds, each measuring
a batch of variants, each informed by everything measured before. This module
runs that loop identically for every sampler, so a difference in results is a
difference between methods rather than between harnesses.

What a round does
-----------------

#. The sampler proposes a **pool** of candidates. How large that pool is, is a
   property of the method -- a genetic algorithm's is its population, MLDE's is
   its library -- and generating them is free either way.
#. Designs measured in an *earlier* round are dropped, and the sampler is asked
   again until the plate can be filled without them.
#. The surrogate scores the pool; the acquisition rule turns predictions and
   uncertainty into one number per candidate.
#. The batch selector picks the ``batch_size`` that will actually be measured.
#. The oracle evaluates exactly those, and only those are charged.
#. The surrogate is refitted on everything measured so far, and the sampler is
   told what its proposals scored.

The plate is always full
------------------------

Every round charges exactly ``batch_size`` oracle calls, so a campaign spends
exactly ``rounds * batch_size``. That is an invariant rather than a tendency,
and it is stated here because the alternative is silent: with a pool the size of
the plate, deduplicating it and assaying whatever survives leaves part of the
oracle budget unspent without any round reporting an error, and the budget every
claim is indexed by is then wrong in the direction that flatters whichever
method happens to repeat itself least.

Where the line between "already measured" and "measured twice" is drawn
-----------------------------------------------------------------------

**Across rounds, the campaign remembers.** A design measured in an earlier round
is not re-ordered, and the sampler is asked again until the plate fills without
it. This is protocol, on the same footing as re-anchoring: the whole reason to
run rounds rather than one large batch is that each round is informed by what
the last one measured, and a lab does not re-assay a variant whose number is
already in the notebook. It is given to every arm on that reasoning, not to one
method as a favour.

**Within a round, duplicates are charged.** A genetic algorithm that breeds 96
offspring of which 10 are identical has consumed 96 wells and bought 86 distinct
data points, and that is the real price of its own convergence. Silently
collapsing them would hand a converging method free measurements and hide the
one cost convergence has. The topping-up above serves the cross-round memory and
nothing else: it never launders a repeat the sampler produced inside one plate.
[RoundRecord.duplicate_fraction][evogfn.loop.ledger.RoundRecord.duplicate_fraction]
reports what that cost, so it is measurable rather than assumed.

**Both halves are counted, and they are twins.** ``duplicate_fraction`` is
repetition *within* one plate; [RoundRecord.redundant][evogfn.loop.ledger.RoundRecord.redundant]
is repetition against the campaign's memory *across* rounds -- the skip below,
which used to happen silently. A converged sampler produces both at once, so
reading either alone attributes the other's cost to itself, and resolving that
misattribution is the only reason the second number exists.

``distinct_batch`` is the ablation that moves the line, filling the plate with
``batch_size`` distinct designs instead of ``batch_size`` proposals. It is a
separate arm rather than a post-hoc correction because deduplicating changes
*which* designs get measured, so the two campaigns diverge from round one and
neither can be derived from the other's ledger.

Why the sampler does not touch the oracle
-----------------------------------------

Training a GFlowNet takes thousands of reward evaluations. Charging those
against the oracle budget would exhaust a realistic 384-call campaign before the
first round finished, and no published method does it -- GFN-AL trains the
sampler against a learned proxy and spends the real budget only on the selected
batch. Getting this wrong does not produce an error; it produces a benchmark in
which the GFlowNet appears catastrophically sample-inefficient for a reason that
has nothing to do with GFlowNets.

The seam is deliberately implicit: a sampler that wants to train against the
surrogate is constructed with *the same surrogate instance* the campaign holds.
Refitting mutates it in place, so the sampler sees each round's model without
the campaign needing to know which samplers care.

Re-anchoring between rounds
---------------------------

A [MutationEnvironment][evogfn.env.mutation.MutationEnvironment] searches within
``max_mutations`` of the design it is anchored to. Hold that anchor at the wild
type for the whole campaign and every round re-searches the same Hamming ball:
four rounds of a four-mutation budget still reach only four mutations from the
wild type, not sixteen. That is not what directed evolution does. Each real round
starts from the best variant the last one produced, and cumulative distance grows
while the per-round budget does not.

The difference is not a matter of degree. A planted optimum can sit further from
the wild type than a per-round mutation budget reaches, so under a fixed anchor
no method can reach it *in principle* -- and a regret reported against it is a
regret nothing could have closed, which every method reports identically and
which says nothing about any of them.

``reanchor`` turns the mechanism on, and it is off by default so that no number
already reported moves without someone asking for it. What the sampler needs when
the anchor moves is not uniform: a policy defined over the same action space
survives it, a CMA-ES distribution decoded relative to the old parent does not.
So the campaign never reaches into a sampler. It asks -- through
[ReanchorableSampler][evogfn.loop.campaign.ReanchorableSampler] -- or it rebuilds
through a factory the caller supplied, and if it can do neither it refuses at
construction rather than mid-campaign.

Defaults
--------

Four rounds of 96, so 384 oracle calls. That is not a round number picked for
convenience -- it is the size of real ML-guided campaigns. ALDE (Arnold lab,
2025) screened 396 variants as six 96-well plates over three rounds; LaMBO-2's
wet-lab campaign measured 374 over three rounds. The iterative-benchmark
convention of 1,000-10,000 evaluations sits above even *classical* directed
evolution, and well above the regime where MLDE's advantage is claimed.

Running against more than one objective
--------------------------------------

[CH65Landscape][evogfn.landscapes.ch65.CH65Landscape] returns three affinities
per variant, and every step of the round above assumes one number: the surrogate
is fitted to a scalar, the acquisition rule ranks a scalar, the ledger records a
best-so-far. Something has to state the trade-off, and the campaign refuses to
be that something -- an invented weighting would be applied to the surrogate, to
the ranking and to the report without ever appearing in the output.

Instead the *acquisition rule* carries it. A multi-objective campaign is
constructed with
[ScalarizedAcquisition][evogfn.acquisition.rules.ScalarizedAcquisition], and the
loop asks it, through
[reduce_objectives][evogfn.acquisition.base.Acquisition.reduce_objectives], for
the one value it ranks. The surrogate's training target, the incumbent an
improvement rule improves on, and ``best_so_far`` are then the same trade-off by
construction rather than by three separate call sites agreeing. A rule that
cannot answer makes the campaign raise *before* the first oracle call, not
after 384 of them.

What survives as a vector is the *record*: every measurement is stored as the
objective vector it was, and
[CampaignResult][evogfn.loop.ledger.CampaignResult] reports hypervolume and
IGD+ over those vectors. The scalarisation directs the search; it does not
decide what the search is scored by.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from evogfn.acquisition.rules import Greedy, TopK
from evogfn.loop.ledger import CampaignResult, RoundRecord
from evogfn.loop.provenance import write_manifest, write_round
from evogfn.metrics.diversity import diversity
from evogfn.metrics.pareto import hypervolume
from evogfn.tracking.base import NoOpTracker

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt

    from evogfn.acquisition.base import Acquisition, BatchSelector
    from evogfn.algorithms.base import Sampler
    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment
    from evogfn.landscapes.base import FitnessLandscape
    from evogfn.surrogate.base import Surrogate
    from evogfn.tracking.base import Tracker

#: Variants per round. One 96-well plate, which is the unit a wet lab works in.
DEFAULT_BATCH_SIZE = 96

#: Rounds per campaign. Three to four is what published wet-lab campaigns run.
DEFAULT_ROUNDS = 4

#: Candidates generated per round before selection. Free, so generous -- but a
#: default rather than a right answer: a method's pool is a published property
#: of that method, and [evogfn.benchmark.methods][] sets it per arm.
DEFAULT_POOL_SIZE = 2048

#: Proposal calls one round may make before it gives up on filling its plate.
#: Generous, because a call is free against the oracle budget and the only thing
#: this bound is protecting against is a sampler that can produce *nothing* new
#: -- which is a terminal condition, not a slow one. Reached only when the
#: sampler's unmeasured reachable set is effectively empty.
MAX_PROPOSAL_ATTEMPTS = 32

#: Below two sequences, pairwise diversity is undefined rather than zero.
_MIN_FOR_DIVERSITY = 2

# An objective matrix is two-dimensional by definition.
_MATRIX_NDIM = 2


@runtime_checkable
class StatesReferencePoint(Protocol):
    """A landscape that can say where its own hypervolume should be measured from.

    A reference point is a claim about the assay -- "below this, a design has
    contributed nothing on this objective" -- and the landscape is the only
    object that holds the assay. CH65's is the Tite-Seq detection floor of 6.0
    on all three antigens, which is where the titration stops resolving; nothing
    outside the landscape knows that. Where a landscape states one, the campaign
    prefers it over guessing, and an explicit argument still overrides it.
    """

    @property
    def reference_point(self) -> Fitness:
        """The ``(n_objectives,)`` worst value worth counting on each objective."""


@runtime_checkable
class StatesReferenceFront(Protocol):
    """A landscape that knows its own Pareto front, for IGD+."""

    @property
    def reference_front(self) -> Fitness:
        """A ``(m, n_objectives)`` array of the truly non-dominated values."""


@runtime_checkable
class ReanchorableSampler(Protocol):
    """A sampler that can be handed a re-anchored environment and carry on.

    What has to happen when the anchor moves is a property of the sampler, and
    the campaign is not entitled to guess. A GFlowNet policy is defined over an
    action space of ``length * |alphabet| + 1`` indices that does not change with
    the anchor, and its input is the state sequence, so the trained weights stay
    meaningful and only the masks move; a hill climber's current point is a
    sequence and survives untouched; a CMA-ES mean lives in a one-hot space that
    is *decoded* relative to the parent, so the same vector means a different
    design after the move and carrying it across would be silently wrong; a
    genetic algorithm's population may fall wholly outside the new mutation
    budget.

    Implementing this is a sampler saying it knows which of those it is. The
    campaign's alternative is to rebuild through a factory, which is correct but
    discards whatever the sampler had learned, so a sampler with state worth
    keeping -- a fitted model, a trained policy, a replay buffer -- should
    implement this instead.
    """

    def reanchored(self, env: MutationEnvironment) -> Sampler:
        """Return the sampler to use from now on, anchored at ``env``.

        Args:
            env: The environment the campaign has moved to.

        Returns:
            The sampler for subsequent rounds. Returning ``self`` after
            updating in place is allowed; the campaign uses whatever comes back.
        """


class Campaign:
    """Runs a sampler against a landscape under a fixed oracle budget.

    Args:
        landscape: The oracle. Every call to it is charged.
        sampler: The method under test.
        surrogate: Model fitted to the measurements and used to score the pool.
            ``None`` runs the sampler unassisted -- the ablation that says how
            much of the result is the surrogate rather than the sampler.
        acquisition: Turns predictions and uncertainty into one score. Defaults
            to [Greedy][evogfn.acquisition.rules.Greedy], the baseline the
            published nulls favour.
        selector: Picks the batch to measure. Defaults to
            [TopK][evogfn.acquisition.rules.TopK].
        rounds: How many design-build-test-learn cycles.
        batch_size: Variants measured per round. Every round measures exactly
            this many.
        pool_size: Candidates the sampler is asked for per proposal call. This
            is a property of the method rather than a harness setting -- a
            genetic algorithm's population is its plate, MLDE's library is
            thousands -- so it belongs to whoever builds the sampler. Where it
            equals ``batch_size`` the campaign still fills the plate, by asking
            again.
        initial_design: Sequences to measure in round 0. ``None`` takes them
            from the sampler, unassisted. Fewer than ``batch_size`` of them is
            allowed and the rest of the plate is topped up from the sampler:
            round 0 is charged like every other round, and a short opening plate
            would be budget quietly not spent.
        skip_measured: Skip candidates measured in an *earlier round*, and keep
            asking until the plate fills without them. A lab does not re-order a
            variant whose number is already in the notebook, and a sampler that
            has collapsed onto one mode would otherwise spend its whole budget
            re-measuring it. Repeats *within* one round are charged regardless:
            they are the cost of the method's own convergence, and
            ``distinct_batch`` is the ablation that removes them. Setting this
            false gives the campaign no memory at all, which is the ablation
            that says how much of a method's apparent efficiency is the
            screening rather than the method.
        distinct_batch: Fill the plate with ``batch_size`` *distinct* designs
            rather than ``batch_size`` proposals, asking the sampler again for
            whatever the duplicates cost. Off by default because charging them
            is what makes convergence cost something. On, this is the same
            algorithm run under a different plate rule, and it is a separate
            campaign rather than a re-reading of the bare one: removing a
            duplicate changes which design takes that well, so the two diverge
            from round one.
        tracker: Where per-round metrics go.
        artifact_dir: Where to write each round's batch, as a chained artifact.
            ``None`` writes nothing, which is right for a benchmark sweep where
            the aggregate is the product. A campaign wants the opposite -- the
            designs that went to the lab in round three and what came back --
            and that is what this records.
        reference_point: The ``(n_objectives,)`` point hypervolume is measured
            from -- the worst value worth counting on each objective. ``None``
            takes the landscape's own, through
            [StatesReferencePoint][evogfn.loop.campaign.StatesReferencePoint],
            and otherwise leaves hypervolume unreported. Nothing here invents
            one: hypervolumes taken from different reference points are not
            comparable, and neither number carries the point it was taken from,
            so a default would silently make two runs incomparable *and* look
            like it had not.
        reference_front: A ``(m, n_objectives)`` front to score IGD+ against,
            or ``None`` to take the landscape's own where it has one.
        environment: The mutation environment the sampler searches. Supplying it
            makes the ledger record which design each round was anchored to and
            how far that sat from the wild type, and it is what ``reanchor``
            moves. ``None`` leaves the anchor untracked and unmoved.
        reanchor: Move the environment's anchor to the best feasible design
            measured so far, at the end of every round. Off by default: turning
            it on changes what a campaign can reach, so no number already
            reported moves unless someone asks for it. With it off the campaign
            searches one fixed Hamming ball for its whole life and can never
            reach a design further than ``max_mutations`` from the wild type,
            however many rounds it runs.
        sampler_factory: Builds a sampler for a re-anchored environment. Used
            when the sampler does not implement
            [ReanchorableSampler][evogfn.loop.campaign.ReanchorableSampler].
            Rebuilding is correct but forgetful -- a factory that closes over
            the policy, surrogate or population it means to keep is what carries
            state across the move.

    Raises:
        ValueError: If any size is not positive, if the pool is smaller than
            the batch it must be selected from, if the landscape returns more
            than one objective and the acquisition rule cannot rank one, if a
            supplied reference point or front does not match the objectives, or
            if re-anchoring is asked for without the means to do it -- no
            environment to move, or a sampler that can neither be informed nor
            rebuilt. Refusing at construction rather than at the end of round
            one is deliberate: the alternative is discovering it after a quarter
            of the oracle budget has been spent.
    """

    def __init__(  # noqa: PLR0913 - a campaign is defined by its protocol
        self,
        *,
        landscape: FitnessLandscape,
        sampler: Sampler,
        surrogate: Surrogate | None = None,
        acquisition: Acquisition | None = None,
        selector: BatchSelector | None = None,
        rounds: int = DEFAULT_ROUNDS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        pool_size: int = DEFAULT_POOL_SIZE,
        initial_design: Tokens | None = None,
        skip_measured: bool = True,
        distinct_batch: bool = False,
        tracker: Tracker | None = None,
        artifact_dir: Path | None = None,
        reference_point: npt.ArrayLike | None = None,
        reference_front: Fitness | None = None,
        environment: MutationEnvironment | None = None,
        reanchor: bool = False,
        sampler_factory: Callable[[MutationEnvironment], Sampler] | None = None,
    ) -> None:
        """Configure the campaign without running it."""
        for name, value in [
            ("rounds", rounds),
            ("batch_size", batch_size),
            ("pool_size", pool_size),
        ]:
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if pool_size < batch_size:
            raise ValueError(
                f"pool_size {pool_size} is smaller than batch_size {batch_size}; "
                "there would be nothing to select from"
            )

        self._landscape = landscape
        self._sampler = sampler
        self._surrogate = surrogate
        self._acquisition = acquisition or Greedy()
        self._selector = selector or TopK()
        # Refused here rather than at the first fit: the alternative is a
        # traceback after a round of oracle calls has already been spent, and
        # the alternative to *that* is a campaign that quietly ranks on one
        # antigen out of three.
        if landscape.n_objectives > 1 and not self._acquisition.supports_multi_objective:
            raise ValueError(
                f"{type(landscape).__name__} returns {landscape.n_objectives} objectives per "
                f"design and {type(self._acquisition).__name__} ranks one value; state the "
                f"trade-off explicitly, e.g. acquisition=ScalarizedAcquisition(Greedy(), "
                f"WeightedSum(), preference)"
            )
        # A zero-row probe of the right width. It touches no data and costs
        # nothing, and it turns "the preference covers two objectives, the
        # landscape returns three" from a traceback in round two -- after a
        # plate has been measured -- into a refusal at construction.
        self._acquisition.reduce_objectives(np.zeros((0, landscape.n_objectives)))
        self._reference_point = _resolve_reference_point(
            landscape, reference_point, n_objectives=landscape.n_objectives
        )
        self._reference_front = _resolve_reference_front(landscape, reference_front)
        self._rounds = rounds
        self._batch_size = batch_size
        self._pool_size = pool_size
        self._initial_design = initial_design
        self._skip_measured = skip_measured
        self._distinct_batch = distinct_batch
        self._tracker = tracker or NoOpTracker()
        self._artifact_dir = artifact_dir
        self._environment = environment
        self._reanchor = reanchor
        self._sampler_factory = sampler_factory
        self._completed: tuple[RoundRecord, ...] = ()
        if reanchor:
            self._check_can_reanchor()

    def _check_can_reanchor(self) -> None:
        """Refuse a re-anchoring campaign that has no way to re-anchor.

        Raises:
            ValueError: If there is no environment whose anchor could move, or
                if the sampler can neither be informed of the move nor rebuilt
                for it. Both are configuration errors, and both would otherwise
                surface at the end of round one with a quarter of the oracle
                budget already spent.
        """
        if self._environment is None:
            raise ValueError(
                "reanchor=True needs the environment whose anchor should move; "
                "pass environment=<the MutationEnvironment the sampler searches>"
            )
        if not isinstance(self._sampler, ReanchorableSampler) and self._sampler_factory is None:
            raise ValueError(
                f"{type(self._sampler).__name__} cannot follow a moved anchor: it does not "
                f"implement reanchored(env), and no sampler_factory was given to rebuild it. "
                f"Re-anchoring anyway would leave the sampler proposing around the old parent "
                f"while the ledger recorded the new one"
            )

    @property
    def budget(self) -> int:
        """Total oracle calls the campaign may spend."""
        return self._rounds * self._batch_size

    @property
    def sampler(self) -> Sampler:
        """The method under test, for reading its own accounting after a run.

        Under ``reanchor`` this is whatever the last move produced -- a rebuilt
        sampler is a different object from the one passed in, and its proposal
        count starts from zero, so read it here rather than from the reference
        you constructed the campaign with.
        """
        return self._sampler

    @property
    def completed_rounds(self) -> tuple[RoundRecord, ...]:
        """The rounds that finished, readable after a run that did not.

        [run][evogfn.loop.campaign.Campaign.run] returns a ledger only when
        every round filled its plate, so a campaign that raises takes its whole
        history with it -- and the caller is left unable to say whether it
        failed in round one having measured nothing or in round four having
        measured 288 designs. Those are different findings, and the difference
        is exactly what a stored record of the failure has to carry.

        Returns:
            One record per completed round, in order, and empty before the first
            round finishes. On a run that completed this is the same tuple the
            result carries.
        """
        return self._completed

    @property
    def environment(self) -> MutationEnvironment | None:
        """The environment as currently anchored, or ``None`` if none was given.

        After a re-anchoring run this is anchored at the best design measured,
        not at the wild type. It is a different object from the one passed in:
        anchors do not move in place.
        """
        return self._environment

    def run(self) -> CampaignResult:
        """Execute every round and return the ledger.

        Returns:
            The measurements and the accounting behind them. Exactly
            [budget][evogfn.loop.campaign.Campaign.budget] oracle calls were
            charged, and the ledger says which.

        Raises:
            RuntimeError: If a round cannot fill its plate -- the sampler was
                asked ``MAX_PROPOSAL_ATTEMPTS`` times and could not between them
                produce ``batch_size`` designs the campaign had not already
                measured. Stopping quietly with a short plate is what this
                replaces, and it is the worse answer: the run then reports a
                full-looking ledger against a budget it never spent, and nothing
                in the numbers says so.
        """
        measured: list[Tokens] = []
        values: list[Fitness] = []
        seen: set[bytes] = set()
        records: list[RoundRecord] = []
        self._completed = ()
        best_so_far = float("-inf")
        spent = 0
        # The anchor moves; the wild type is what distance is measured from, so
        # it is read once and never again.
        wild_type = None if self._environment is None else self._environment.parent
        best_anchor_value = float("-inf")

        for index in range(self._rounds):
            # No `remaining` arithmetic: every round fills its plate or raises,
            # so the batch is always `batch_size` and a running check against
            # the budget would be a branch that can never be taken. `spent` is
            # kept because the tracker reports it, not because it gates anything.
            proposed, redundant, screened, batch, predicted = self._design(
                index, measured, values, seen
            )

            scores = self._landscape.evaluate(batch)
            spent += batch.shape[0]
            measured.append(batch)
            values.append(scores)
            seen.update(row.tobytes() for row in np.ascontiguousarray(batch))
            self._sampler.observe(batch, scores)

            record = self._record(
                index=index,
                proposed=proposed,
                redundant=redundant,
                screened=screened,
                batch=batch,
                scores=scores,
                previous_best=best_so_far,
                predicted=predicted,
                history=values,
            )
            # Stamped after the fact so that the round is recorded against the
            # anchor it actually proposed from, never the one it moves to below.
            record = self._stamp_anchor(record, wild_type)
            if self._artifact_dir is not None:
                write_round(
                    self._artifact_dir,
                    record=record,
                    sequences=batch,
                    values=scores,
                    predicted=predicted,
                    tracker=self._tracker,
                )
            best_so_far = record.best_so_far
            records.append(record)
            # Published on the campaign as each round lands, not at the end: the
            # whole point of this field is to survive the raise, and a run that
            # raises never reaches the end.
            self._completed = tuple(records)
            metrics = {
                "best_so_far": record.best_so_far,
                "best_in_round": record.best_in_round,
                "batch_diversity": record.batch_diversity,
                "feasible_fraction": record.feasible_fraction,
                "duplicate_fraction": record.duplicate_fraction,
                "oracle_calls": float(spent),
            }
            # Logged only when it exists. A key that is always present and
            # usually ``nan`` averages into a table as a missing value nobody
            # asked for; an absent key is read as "this run did not report it".
            if np.isfinite(record.hypervolume):
                metrics["hypervolume"] = record.hypervolume
            if wild_type is not None:
                metrics["anchor_distance"] = float(record.anchor_distance)
            # Same rule as the two above, for the same reason: a campaign with
            # no cross-round memory never consulted one, and logging a zero for
            # it would average into a table as "this arm repeated nothing"
            # rather than as "this arm was not screened".
            if record.redundant is not None:
                metrics["redundant_fraction"] = record.redundant_fraction
            self._tracker.log_metrics(metrics, step=index)

            if self._reanchor:
                best_anchor_value = self._advance_anchor(batch, scores, best_anchor_value)

        if self._artifact_dir is not None and records:
            write_manifest(self._artifact_dir, tuple(records))

        optimum = self._landscape.optimum
        # With one objective the landscape's optimum is a target and the gap to
        # it is a regret. With several it is an *ideal point* -- the best value
        # on each objective separately, attained by different designs and by no
        # single one -- so collapsing it to a scalar would produce a target
        # nothing can reach and a regret that never reaches zero.
        single = self._landscape.n_objectives == 1
        return CampaignResult(
            sampler=self._sampler.name,
            rounds=tuple(records),
            sequences=(
                np.concatenate(measured)
                if measured
                else np.zeros((0, self._landscape.sequence_length), dtype=np.int32)
            ),
            values=(
                np.concatenate(values) if values else np.zeros((0, self._landscape.n_objectives))
            ),
            optimum=float(np.max(optimum)) if optimum is not None and single else None,
            ideal_point=(
                np.asarray(optimum, dtype=np.float64)
                if optimum is not None and not single
                else None
            ),
            reference_point=self._reference_point,
            reference_front=self._reference_front,
        )

    def _design(
        self,
        index: int,
        measured: list[Tokens],
        values: list[Fitness],
        seen: set[bytes],
    ) -> tuple[int, int | None, int, Tokens, npt.NDArray[np.floating] | None]:
        """Choose the batch, returning proposals generated, pool screened, and batch.

        Round 0 has nothing to fit a surrogate on, so it is the sampler's own
        proposals unassisted. Every later round refits on the accumulated
        measurements first.

        Args:
            index: Zero-based round number.
            measured: Every earlier round's designs, for refitting.
            values: Their objective values, aligned with ``measured``.
            seen: Every design measured so far, as row bytes. Read to decide
                what the plate may still hold; updated by the caller once the
                batch has actually been charged.

        Returns:
            Proposals generated, proposals refused as already measured in an
            earlier round, proposals that reached the selector, the batch of
            exactly ``batch_size`` designs to measure, and what the surrogate
            predicted for them where there was one.

        Raises:
            RuntimeError: If the plate cannot be filled; see
                [_fill][evogfn.loop.campaign.Campaign._fill].
        """
        size = self._batch_size
        fitted = False
        if index > 0 and self._surrogate is not None:
            # A method can fail to produce a single buildable design in a whole
            # round -- on a sparse feasible set an unmasked sampler routinely
            # does. There is then nothing to fit, and that is a result about the
            # method rather than an error: it proceeds unassisted and the ledger
            # records a feasible fraction of zero. Raising here would turn the
            # finding into a traceback and lose the rest of the campaign.
            # Reduced by the acquisition rule rather than by the campaign, so
            # the target the surrogate learns is the same trade-off the rule
            # will rank predictions on. With one objective this is that
            # objective, unchanged and still a column.
            history = self._acquisition.reduce_objectives(np.concatenate(values))[:, None]
            if np.isfinite(history).any():
                # In place, so any sampler holding this instance sees the update.
                self._surrogate.fit(np.concatenate(measured), history)
                fitted = True

        proposed, redundant, pool = self._fill(index, seen, size)
        screened = pool.shape[0]
        # Round 0 has no model to score with, so the pool order stands. Taking a
        # prefix is not arbitrary: a sampler that ranks its own output -- MLDE
        # returns its library best-first -- has already said which designs it
        # would send to the lab, and reordering them would measure the harness.
        if index == 0 or self._surrogate is None or not fitted:
            return proposed, redundant, screened, pool[:size], None

        mean, spread = self._surrogate.predict(pool)
        best_observed = self._best_observed(values)
        scored = self._acquisition.score(mean, spread, best_observed=best_observed)
        chosen = self._selector.select(pool, scored, size)
        return proposed, redundant, screened, pool[chosen], mean[chosen]

    def _stamp_anchor(self, record: RoundRecord, wild_type: Tokens | None) -> RoundRecord:
        """Record which design this round searched from, and how far out it was.

        Written onto the record rather than passed into
        [_record][evogfn.loop.campaign.Campaign._record] because it is not a
        summary of the batch: it is where the batch came from, and it is the
        only field in the ledger whose value is fixed *before* the round rather
        than measured after it.

        Args:
            record: The round's summary, as measured.
            wild_type: The campaign's original parent, or ``None`` when no
                environment was supplied and there is no anchor to report.

        Returns:
            The record, carrying the anchor and its distance from the wild type
            where those are known, and unchanged where they are not.
        """
        if wild_type is None or self._environment is None:
            return record
        anchor = self._environment.parent
        return replace(
            record,
            anchor=tuple(int(token) for token in anchor),
            anchor_distance=int((np.asarray(anchor) != np.asarray(wild_type)).sum()),
        )

    def _advance_anchor(self, batch: Tokens, scores: Fitness, incumbent: float) -> float:
        """Move the environment to the best feasible design measured so far.

        The anchor follows the ledger's own ``best_so_far``, on the acquisition
        rule's scale: the design that set it is the design the next round starts
        from. Asking the rule rather than reducing the objectives here is what
        keeps the search, the report and the anchor on one trade-off -- a
        campaign that ranked by one weighting and re-anchored by another would
        walk somewhere its own ledger never claimed was good. Ties and
        non-improving rounds leave the anchor where it is, so a round that
        learned nothing does not move the search off a peak already found.

        Infeasible designs are never candidates -- they score ``-inf`` and are
        filtered here -- and
        [reanchored][evogfn.env.mutation.MutationEnvironment.reanchored] refuses
        one anyway, which is the second line of defence rather than the first.

        Args:
            batch: The designs measured this round.
            scores: Their objective values, ``(n, n_objectives)``.
            incumbent: The best value that has moved the anchor so far.

        Returns:
            The value the anchor now sits at, unchanged when nothing improved.

        Raises:
            ValueError: If the new anchor is not a design this environment
                could be anchored at -- most usefully, an infeasible one -- or
                if the acquisition rule cannot reduce the measurements to the
                one value it ranks.
        """
        if self._environment is None:  # pragma: no cover - refused at construction
            raise RuntimeError("re-anchoring without an environment")
        flat = self._acquisition.reduce_objectives(scores)
        finite = np.isfinite(flat)
        if not finite.any():
            return incumbent

        best = float(flat[finite].max())
        if best < incumbent:
            return incumbent
        tied = np.flatnonzero(finite & (flat >= best))
        if best > incumbent:
            position = int(tied[0])
        else:
            # A tie, and moving is still right. Rewards here are products of
            # quantised terms, so they are flat across most of a neighbourhood,
            # and requiring a strict improvement would leave a campaign pinned
            # at the wild type for its whole life with the mechanism never
            # firing.
            #
            # Among equally good designs, take the one furthest from the current
            # anchor. That crosses the plateau rather than sitting on it, and it
            # is what a lab does with a flat round: carry forward the variant
            # that opens the most new territory. Distance is the tie-break
            # rather than chance, so the walk stays reproducible.
            designs = np.asarray(batch)
            parent = self._environment.parent
            distance = (designs[tied] != parent).sum(axis=1)
            position = int(tied[int(np.argmax(distance))])
            if int((designs[position] != parent).sum()) == 0:
                return incumbent

        env = self._environment.reanchored(np.asarray(batch)[position])
        if isinstance(self._sampler, ReanchorableSampler):
            self._sampler = self._sampler.reanchored(env)
        elif self._sampler_factory is not None:
            self._sampler = self._sampler_factory(env)
        else:  # pragma: no cover - refused at construction
            raise RuntimeError("re-anchoring without a way to move the sampler")
        self._environment = env
        return float(flat[position])

    def _fill(self, index: int, seen: set[bytes], size: int) -> tuple[int, int | None, Tokens]:
        """Ask the sampler until the plate can be filled, and say what that cost.

        One call to a sampler whose pool is its published population -- 96 for a
        genetic algorithm -- returns exactly one plate's worth, and any of it the
        campaign has already measured leaves the plate short. Asking again is
        what fills it, and it is the whole reason this loop exists: without it,
        the round assays whatever survives the filter and leaves the rest of the
        budget unspent without saying so.

        What the loop does *not* do is remove a repeat the sampler produced
        inside a single call. Those stay in the pool and go on to consume wells,
        because they are the method's own convergence rather than the harness's
        bookkeeping -- unless ``distinct_batch`` is set, which is the arm that
        asks what the alternative plate rule would have measured.

        Args:
            index: Zero-based round number. Only round 0 may draw its first
                candidates from a supplied initial design.
            seen: Designs measured in earlier rounds, as row bytes.
            size: Wells to fill.

        Returns:
            Candidates generated across every call made; how many of them the
            campaign's memory refused, or ``None`` when there is no memory to
            refuse with; and the pool the selector will choose from -- at least
            ``size`` rows, in the order the sampler produced them.

        Raises:
            RuntimeError: If ``MAX_PROPOSAL_ATTEMPTS`` calls could not between
                them produce ``size`` admissible designs. Loud because the
                alternative is a campaign reporting a full ledger against a
                budget it never spent, and because the condition is terminal:
                the sampler's unmeasured reachable set is empty, and a further
                round would find it empty too.
        """
        proposed = 0
        held = 0
        remembered = 0
        collected: list[Tokens] = []
        plated: set[bytes] = set()
        for attempt in range(MAX_PROPOSAL_ATTEMPTS):
            chunk = np.ascontiguousarray(self._propose(index, attempt))
            proposed += chunk.shape[0]
            keep, repeats = self._admissible(chunk, seen, plated)
            remembered += repeats
            if keep.size:
                collected.append(chunk[keep])
                held += int(keep.size)
            if held >= size:
                # ``None`` rather than the zero the accumulator holds when the
                # memory is off. The count is a measurement only where something
                # did the measuring, and a campaign run without screening never
                # looked -- see
                # [RoundRecord.redundant][evogfn.loop.ledger.RoundRecord.redundant].
                return (
                    proposed,
                    (remembered if self._skip_measured else None),
                    np.concatenate(collected),
                )
        raise RuntimeError(
            f"{self._sampler.name} filled {held} of {size} wells in round {index} across "
            f"{MAX_PROPOSAL_ATTEMPTS} proposal calls ({proposed} candidates); it cannot produce "
            f"designs this campaign has not already measured, so the remaining budget could only "
            f"be spent re-measuring them"
        )

    def _propose(self, index: int, attempt: int) -> Tokens:
        """One call's worth of candidates, from the design or from the sampler.

        Args:
            index: Zero-based round number.
            attempt: Which call this is within the round, from zero.

        Returns:
            The candidates. A supplied initial design is round 0's *first* call
            only: it is a stated opening plate, not a well the sampler may keep
            re-proposing, and topping it up has to come from somewhere that can
            produce something new.
        """
        if index == 0 and attempt == 0 and self._initial_design is not None:
            return np.asarray(self._initial_design)
        return self._sampler.propose(self._pool_size)

    def _admissible(
        self, pool: Tokens, seen: set[bytes], plated: set[bytes]
    ) -> tuple[npt.NDArray[np.intp], int]:
        """Positions in ``pool`` this round may still spend a well on.

        Args:
            pool: One proposal call's candidates, C-contiguous.
            seen: Designs measured in earlier rounds.
            plated: Designs already admitted to *this* round's pool. Mutated
                here, and carried across the round's proposal calls so that
                ``distinct_batch`` deduplicates the plate rather than each call
                separately.

        Returns:
            The positions to keep, in order, and how many were dropped because
            an *earlier round* had already measured them. That second number is
            a raw count and always an integer here; turning "the memory was
            never consulted" into ``None`` is
            [_fill][evogfn.loop.campaign.Campaign._fill]'s job, because it is
            the only place that knows whether any call was made at all.
        """
        if not self._skip_measured and not self._distinct_batch:
            return np.arange(pool.shape[0], dtype=np.intp), 0
        keep: list[int] = []
        redundant = 0
        for position, row in enumerate(pool):
            key = row.tobytes()
            if self._skip_measured and key in seen:
                # Counted rather than merely skipped. Without this the drop was
                # invisible, and the proposals-to-screened gap conflated it with
                # a within-plate repeat ``distinct_batch`` had removed -- two
                # different costs, one number, and no way to tell which had
                # produced it.
                redundant += 1
                continue
            if self._distinct_batch:
                if key in plated:
                    continue
                plated.add(key)
            keep.append(position)
        return np.asarray(keep, dtype=np.intp), redundant

    def _record(  # noqa: PLR0913 - a round record is its fields
        self,
        *,
        index: int,
        proposed: int,
        redundant: int | None,
        screened: int,
        batch: Tokens,
        scores: Fitness,
        previous_best: float,
        predicted: npt.NDArray[np.floating] | None = None,
        history: list[Fitness] | None = None,
    ) -> RoundRecord:
        """Summarise a completed round.

        Args:
            index: Zero-based round number.
            proposed: Candidates the sampler generated, across every proposal
                call the round needed.
            redundant: Proposals refused because an earlier round had already
                measured them, or ``None`` where the campaign keeps no memory.
            screened: Proposals that reached the selector.
            batch: The sequences measured.
            scores: Their ``(n, n_objectives)`` measured values.
            previous_best: Best value before this round.
            predicted: What the surrogate said the batch would score, or
                ``None`` when there was no surrogate.
            history: Every round's measurements including this one, for the
                cumulative hypervolume. ``None`` leaves it unreported.

        Returns:
            The round's ledger entry.
        """
        flat = self._acquisition.reduce_objectives(scores)
        finite = flat[np.isfinite(flat)]
        best_in_round = float(finite.max()) if finite.size else float("-inf")
        return RoundRecord(
            index=index,
            proposed=proposed,
            screened=screened,
            evaluated=batch.shape[0],
            feasible=int(np.isfinite(flat).sum()),
            best_in_round=best_in_round,
            best_so_far=max(previous_best, best_in_round),
            mean_in_round=float(finite.mean()) if finite.size else float("-inf"),
            batch_diversity=(diversity(batch) if batch.shape[0] >= _MIN_FOR_DIVERSITY else 0.0),
            surrogate_correlation=_correlation(predicted, flat),
            hypervolume=self._hypervolume(history),
            # The twins, written together so a reader of this call site sees
            # that they are two counts of two different things: wells spent on
            # the same design inside this plate, and proposals refused for
            # having been measured in an earlier one.
            duplicates=_duplicates(batch),
            redundant=redundant,
        )

    def _hypervolume(self, history: list[Fitness] | None) -> float:
        """Volume dominated by everything measured so far, or ``nan``.

        Args:
            history: Every round's measurements, including the one just made.

        Returns:
            The dominated volume, or ``nan`` when there is no reference point to
            measure it from or when the exact method cannot run.

        Note:
            A front larger than the exact hypervolume in
            [evogfn.metrics.pareto][] accepts is caught and reported as ``nan``
            rather than raised. Raising would abort a campaign *after* its oracle
            calls had been spent, over a summary statistic -- the measurements
            are the product and they survive either way. It is not swallowed
            into a plausible number: ``nan`` propagates, and the front is still
            on the result for anyone who wants to score it with a dedicated
            implementation.
        """
        if self._reference_point is None or not history:
            return float("nan")
        try:
            return hypervolume(np.concatenate(history), self._reference_point)
        except NotImplementedError:
            return float("nan")

    def _best_observed(self, values: list[Fitness]) -> float:
        """Best finite measurement so far, for improvement-based acquisition.

        Args:
            values: One ``(n, n_objectives)`` array of objective values per
                completed round.

        Returns:
            The largest finite value measured, on the scale the acquisition
            rule's
            [reduce_objectives][evogfn.acquisition.base.Acquisition.reduce_objectives]
            puts measurements on, or ``0.0`` if nothing finite has been measured
            yet -- the incumbent an improvement rule falls back to before there
            is one.

        Raises:
            ValueError: If the rule cannot reduce the measurements it is given,
                which is where a scalar rule on a multi-objective landscape
                stops rather than improving on an incumbent taken across
                objectives of different scales.
        """
        flat = self._acquisition.reduce_objectives(np.concatenate(values))
        finite = flat[np.isfinite(flat)]
        return float(finite.max()) if finite.size else 0.0

    def __repr__(self) -> str:
        """Name the sampler and the budget it is held to."""
        return (
            f"Campaign(sampler={self._sampler.name}, rounds={self._rounds}, "
            f"batch_size={self._batch_size}, budget={self.budget})"
        )


def _duplicates(batch: Tokens) -> int:
    """Wells in ``batch`` holding a design already elsewhere on the same plate.

    Counted on the batch that was charged rather than on the pool it came from,
    because a well is what a duplicate costs: a pool may hold a design twice and
    have neither copy selected, and that cost nothing.

    Args:
        batch: The ``(n, sequence_length)`` designs measured this round.

    Returns:
        ``n`` minus the number of distinct designs among them, so zero when
        every well holds something different.
    """
    rows = np.ascontiguousarray(batch)
    return int(rows.shape[0] - len({row.tobytes() for row in rows}))


def _correlation(
    predicted: npt.NDArray[np.floating] | None, measured: npt.NDArray[np.floating]
) -> float:
    """Pearson correlation between prediction and measurement, or ``nan``.

    Returns ``nan`` rather than zero when it cannot be computed -- no surrogate,
    fewer than two finite measurements, or a constant on either side. Zero would
    read as "the model is useless", which is a different claim from "there was
    nothing to correlate".
    """
    if predicted is None:
        return float("nan")
    usable = np.isfinite(predicted) & np.isfinite(measured)
    if usable.sum() < _MIN_FOR_DIVERSITY:
        return float("nan")
    left, right = predicted[usable], measured[usable]
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _resolve_reference_point(
    landscape: FitnessLandscape,
    supplied: npt.ArrayLike | None,
    *,
    n_objectives: int,
) -> npt.NDArray[np.float64] | None:
    """Settle where hypervolume is measured from, or leave it unmeasured.

    Precedence is caller, then landscape, then nothing. Nothing is a real
    answer: a default reference point would be applied silently, would not
    appear next to the number it produced, and would make two runs with
    different landscapes -- or the same landscape a version apart --
    incomparable in a way no assertion could catch.

    Args:
        landscape: The oracle, asked for its own reference point when the caller
            supplied none.
        supplied: An explicit ``(n_objectives,)`` point, or ``None``.
        n_objectives: Objectives the landscape returns.

    Returns:
        The reference point, or ``None`` when neither the caller nor the
        landscape stated one.

    Raises:
        ValueError: If the point is not a finite vector of the right width.
    """
    point = supplied
    if point is None:
        if not isinstance(landscape, StatesReferencePoint):
            return None
        point = landscape.reference_point
    array = np.asarray(point, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != n_objectives:
        raise ValueError(
            f"expected a reference point of shape ({n_objectives},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(
            "the reference point must be finite; an infinite one makes every hypervolume "
            "computed against it infinite or zero"
        )
    return array


def _resolve_reference_front(
    landscape: FitnessLandscape, supplied: Fitness | None
) -> npt.NDArray[np.float64] | None:
    """Settle which front IGD+ is scored against, or leave it unscored.

    Args:
        landscape: The oracle, asked for its own front when the caller supplied
            none.
        supplied: An explicit ``(m, n_objectives)`` front, or ``None``.

    Returns:
        The reference front, or ``None`` when neither the caller nor the
        landscape stated one.

    Raises:
        ValueError: If the front is not a two-dimensional matrix of the right
            width, or is not finite.
    """
    front = supplied
    if front is None:
        if not isinstance(landscape, StatesReferenceFront):
            return None
        front = landscape.reference_front
    array = np.asarray(front, dtype=np.float64)
    if array.ndim != _MATRIX_NDIM or array.shape[1] != landscape.n_objectives:
        raise ValueError(
            f"expected a reference front of shape (m, {landscape.n_objectives}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(
            "the reference front must be finite; an infinite reference point makes every "
            "distance to it infinite or undefined"
        )
    return array
