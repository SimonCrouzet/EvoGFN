"""Traditional directed evolution: the two site-saturation arms of the MLDE line.

Site-saturation mutagenesis is one move: pick a residue, make every substitution
at it, measure them all. What a campaign does with those measurements separates
the arms. Two are here, under the names Li et al. use:

* [SingleStepWalk][evogfn.algorithms.baselines.directed_evolution.SingleStepWalk]
  -- ``single-step``. Saturate one site, **fix** the best residue, move to the
  next site and saturate it *on that background*, never returning to a fixed
  site. Deterministic given the site order, and path-dependent.
* [Recombination][evogfn.algorithms.baselines.directed_evolution.Recombination]
  -- ``recomb``. Saturate every site *independently* on one fixed background,
  then build a single design carrying the winner from each. No site is ever
  measured against another site's winner, so the arm assumes the sites combine
  additively and spends one assay finding out.

Neither is [HillClimbing][evogfn.algorithms.baselines.mutagenesis.HillClimbing],
which draws a *random* single substitution anywhere and may return to a position
it has already changed. These arms exhaust one position before committing to it
and then never touch it again. The two are the different objects the field calls
"directed evolution".

Sample counts, and what they fix
---------------------------------

Li et al. state each arm's cost and itemise it, which settles two questions the
implementation would otherwise guess at. Verbatim:

    "The DE strategies are summarized as "recomb", a recombination of the best
    SSM variant at each site (19 x n_site + 2 samples, including the initial and
    final variant); "single-step", an iterative process starting from any site
    with subsequent variants built on the best variant found (19 x n_site + 1
    samples, including the initial variant); and "top96 recomb" ..."

**The background is measured once.** Both arms carry an ``initial variant``, and
``single-step``'s extra is exactly that one sample -- no final variant, because
the design it ends on is one of the per-site scans already paid for.

**The residue already there is a competitor.** The count is ``19`` per site
rather than ``20``, and the reference implementation scans all twenty amino acids
at each position -- the incumbent's own among them, as a re-lookup costing no new
sample. A site keeps the residue it has unless one of the nineteen substitutions
beats the background's measured value, which is what makes the walk monotone and
is the only thing that initial measurement is for.

**A consequence to expect:** on a landscape where no single substitution improves
on the background, ``recomb`` nominates the background itself -- a design already
measured -- and that replicate has nothing left to propose. That is a true fact
about recombination-based directed evolution on a flat or sign-epistatic
landscape; forcing a harmful substitution in to avoid it would be a better arm
than the one published.

Which sites, and in which order
-------------------------------

The published protocols take the site set from the library. Nothing here has a
library, so the site set is the sequence itself and the order is a uniformly
random permutation drawn from the campaign's seed.

The walk is path-dependent -- Li et al. note a four-site library has *"a total of
24 (4!) possible orders of sampling"* -- and both their simulation and ALDE's
enumerate every order and average over them. A seeded permutation makes each
campaign one order and the seed average their average.

What bounds the walk, and what does not
---------------------------------------

**The mutation budget bounds it.** Every substitution is checked against the
environment's Hamming ball before it is proposed, so a site whose substitutions
would carry the design outside it contributes nothing and the walk moves on. No
separate site count is maintained, and none could be right: a site whose winner
is the incumbent's own residue costs no budget, so how many sites a fixed anchor
admits is not known until the walk has run. Under a moved anchor the ball follows
the incumbent and the budget is restored every round.

**The feasibility constraint does not bound it**, deliberately. These are
unmasked classical arms, on the footing of the genetic algorithm, CMA-ES and
random mutagenesis: an adjacency-violating design reaches the assay, scores minus
infinity, and the well it cost is the finding. Site saturation makes *all*
substitutions at a residue, so filtering the infeasible ones would quietly give
these arms the feasibility-by-construction the masked policy claims as an
advantage, on the one task built to measure it.

The surplus budget is spent on replicates, and that choice is ours
------------------------------------------------------------------

A four-site protocol costs about one plate, while a campaign here runs four. The
papers do not say what to do with the difference, because in their setting there
is none -- they report each arm at its own fixed cost. Here every round charges a
full plate, so something has to fill the surplus.

`ReplicatedProtocol` is that choice: run several copies of the protocol
concurrently, each with **its own site order**, pool what they ask for into each
plate, and start a further copy if they all finish while budget remains. A
replicate is the same walk at another order, which is what both reference
implementations do; the alternative -- re-proposing designs the protocol had
already named -- charges several wells per measurement and handicaps the arm by
most of its budget. Nothing selects a winning replicate: every design measured
goes into the ledger and the reported best-so-far is over all of them, which is
how Li et al. score these arms too.

Two things to keep in view. Such a row is a *best over site orders at the
campaign's budget* where the source reports an *average over site orders at one
walk's cost*, and a comparison against a published DE number has to say which it
is. And where a replicate's site set cannot differ from another's, the replicates
name identical designs, the pooled request deduplicates them, and the arm spends
exactly the protocol's cost with the surplus showing up as duplicate wells.

`requested` reports the distinct designs an arm has named, which is the number to
quote beside an oracle-call column that counts wells.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines._reanchor import carried_design
from evogfn.algorithms.baselines._values import single_objective

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment

#: Assays a recombination arm spends on top of the per-site scans. Li et al.
#: state the arm's cost as ``19 x n_site + 2 samples, including the initial and
#: final variant``: the background every site's substitutions are judged against,
#: and the recombinant built from the winners.
RECOMBINATION_OVERHEAD = 2

#: Draws allowed per replicate when looking for a site order nobody else drew.
#: Bounded because on a short sequence there are few orders to find and an
#: unbounded search for a distinct one would not return.
_ORDER_ATTEMPTS = 8

#: Replicates a pooled request may start in one call before concluding that a
#: fresh one would have nothing to ask for either. Bounded for the same reason:
#: on a library the campaign has measured in full, every replicate is born
#: finished, and starting them one after another would not terminate.
_SPAWN_ATTEMPTS = 4


def site_order(env: MutationEnvironment, rng: np.random.Generator) -> npt.NDArray[np.intp]:
    """A permutation of the sequence's positions, to be walked in that order.

    Args:
        env: Supplies the sequence length.
        rng: Draws the permutation. The generator is advanced, so a caller that
            also samples from it gets a stream that depends on this call.

    Returns:
        A ``(sequence_length,)`` array of positions in walk order.
    """
    return np.asarray(rng.permutation(env.sequence_length), dtype=np.intp)


def distinct_site_orders(
    env: MutationEnvironment, rng: np.random.Generator, count: int
) -> list[npt.NDArray[np.intp]]:
    """Up to ``count`` different site orders, for replicates of one protocol.

    Distinct rather than merely independent, because two replicates at the same
    order are the same experiment: they would name identical designs, pool down
    to one, and leave the arm silently running fewer replicates than its budget
    paid for.

    Args:
        env: Supplies the sequence length.
        rng: Draws the permutations.
        count: How many are wanted.

    Returns:
        Between one and ``count`` orders. Fewer than asked for where the
        sequence is short enough that there are not that many permutations to
        find -- which is a real limit on how many replicates a library supports,
        not an error.
    """
    orders: list[npt.NDArray[np.intp]] = []
    seen: set[bytes] = set()
    for _ in range(max(count, 1) * _ORDER_ATTEMPTS):
        if len(orders) >= count:
            break
        order = site_order(env, rng)
        key = order.tobytes()
        if key in seen:
            continue
        seen.add(key)
        orders.append(order)
    return orders or [site_order(env, rng)]


def within_budget(env: MutationEnvironment, design: Tokens) -> bool:
    """Whether a design lies inside the environment's Hamming ball.

    The mutation budget alone, and not
    [is_reachable][evogfn.env.mutation.MutationEnvironment.is_reachable], which
    also requires feasibility. The distinction is the whole of these arms'
    relationship to the constraint experiment: a design outside the budget is one
    no sampler in the suite could have produced and scoring it would make the
    comparison meaningless, whereas an infeasible design is one an unmasked
    method proposes and pays for. See the module docstring.

    Args:
        env: Supplies the anchor and the budget.
        design: A single sequence of the environment's length.

    Returns:
        Whether it differs from the anchor in at most ``max_mutations``
        positions.
    """
    return bool((np.asarray(design) != env.parent).sum() <= env.max_mutations)


def substitutions_at(
    env: MutationEnvironment, design: Tokens, position: int
) -> list[npt.NDArray[np.integer]]:
    """Every single substitution of ``design`` at one position, within budget.

    Saturation mutagenesis makes all substitutions at a residue, which is
    ``|alphabet| - 1`` of them. Those that would carry the design outside the
    environment's Hamming ball are omitted, since nothing in the suite can
    produce them; those that merely violate the adjacency constraint are kept,
    since an unmasked method proposing them is what the constraint task measures.

    A site at the edge of the budget therefore contributes nothing, and the
    caller is expected to move on rather than to treat the empty list as an
    error.

    Args:
        env: Supplies the alphabet, the anchor and the budget.
        design: The background to substitute into.
        position: Which residue to saturate.

    Returns:
        The variants, in token order, each a fresh array. Empty where the budget
        admits none of them.
    """
    background = np.ascontiguousarray(design)
    variants = []
    for token in range(env.alphabet.size):
        if token == background[position]:
            continue
        variant = background.copy()
        variant[position] = token
        if within_budget(env, variant):
            variants.append(variant)
    return variants


class SaturationProtocol(Sampler, ABC):
    """A directed-evolution protocol that names the designs it wants measured.

    The shared shape of both arms, and the seam
    `ReplicatedProtocol` needs. A protocol here does not
    *search* for a plate's worth of candidates: at any moment it wants a specific,
    usually small, set of designs measured, and it wants nothing else. Everything
    else -- how a larger plate is filled, how many copies of the protocol run at
    once -- is budget policy sitting above this class rather than part of any
    published method.

    Subclasses supply `requests`; the cycling, the accounting and the
    refusal-when-finished are shared so the two arms cannot drift apart on any of
    them.
    """

    def __init__(self, env: MutationEnvironment) -> None:
        """Store the environment the protocol is defined against."""
        super().__init__()
        self._env = env
        # Every measurement this protocol has been shown, not only the ones it
        # asked for. Under `ReplicatedProtocol` a sibling's plate is offered
        # here too, and a design this protocol will want later may already have
        # been measured through one of them; without the memory it would ask for
        # a design the campaign has already paid for, have it dropped as a
        # repeat, and wait for a value that can never arrive.
        self._measured: dict[bytes, float] = {}

    def remember(self, measurements: Mapping[bytes, float]) -> None:
        """Adopt measurements taken before this protocol existed.

        What a replicate started part-way through a campaign needs, and the
        reason it is worth having rather than letting the fresh replicate
        rediscover them: everything already measured is in the campaign's memory
        and would be dropped as a repeat, so a replicate that did not know would
        spend its whole request on designs that can never come back with a value.

        Args:
            measurements: Design identities, as `_key` produces them, mapped to
                what they scored.
        """
        self._measured.update(measurements)
        self._on_remembered()

    def _on_remembered(self) -> None:
        """React to measurements that arrived without being asked for.

        A no-op for a protocol whose outstanding request is derived fresh each
        time it is asked. A protocol holding an *open* unit of work -- a site
        part-way through saturation -- has to check whether that unit is now
        answered, since nothing else will prompt it to.
        """

    @abstractmethod
    def requests(self) -> list[npt.NDArray[np.integer]]:
        """The designs the protocol currently wants measured, in its own order.

        Returns:
            The outstanding designs, or an empty list once the protocol has
            nothing further to ask for.
        """

    @property
    @abstractmethod
    def requested(self) -> int:
        """Distinct designs the protocol has named so far."""

    @abstractmethod
    def reanchored(self, env: MutationEnvironment) -> SaturationProtocol:
        """Return the protocol to use from now on, anchored at ``env``."""

    @abstractmethod
    def _exhausted_message(self) -> str:
        """What to say when the protocol has nothing left to ask for."""

    @property
    def finished(self) -> bool:
        """Whether the protocol has nothing further to ask for."""
        return not self.requests()

    def propose(self, n: int) -> Tokens:
        """Return the outstanding designs, cycled to fill the plate.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array. Where the protocol wants fewer
            than ``n`` designs the remainder repeats them in protocol order --
            which is why an arm run at a plate much larger than its protocol
            should be wrapped in `ReplicatedProtocol`
            rather than left to repeat itself.

        Raises:
            RuntimeError: If the protocol is complete. Raised rather than
                answered with fresh moves, because any such move -- a restart, a
                random draw, a revision of a fixed site -- is a mechanism no
                published protocol describes.
        """
        pending = self.requests()
        if not pending:
            raise RuntimeError(self._exhausted_message())
        self._count(n)
        return np.stack([pending[index % len(pending)] for index in range(n)])

    def _key(self, design: Tokens) -> bytes:
        """A dtype-stable identity for a design, so lookups cannot silently miss."""
        return np.ascontiguousarray(design, dtype=self._env.parent.dtype).tobytes()


class SingleStepWalk(SaturationProtocol):
    """Saturate a site, fix its best residue, move on, and never look back.

    ALDE's specification, which is the one every "MLDE beats DE" claim is
    measured against: one residue is mutated to all possible amino acids, the
    best is fixed, and the process repeats at each of the residues under study.
    Li et al. describe the same object at greater length -- *"The substitution
    yielding the highest fitness is fixed, and the position is restricted from
    further exploration"*, and *"each site is optimized once per simulation"* --
    and cost it at ``19 x n_site + 1 samples, including the initial variant``.
    The residue already carried by the incumbent therefore competes on equal
    terms with the substitutions without costing a sample of its own: the
    background is measured once at the start and its value carried.

    Two properties this arm exists to have, both of which a near-duplicate of
    hill climbing would lose:

    * **A site is never revisited.** The cursor into the site order only ever
      advances, so a residue fixed in an early round cannot be revised by a later
      one however badly it turns out. That is the cost of a greedy walk and the
      reason the field reports it as path-dependent.
    * **A site is saturated before it is fixed.** The winner is chosen only once
      every substitution at that site has been measured, so a plate too small to
      hold a whole site spreads it over consecutive rounds rather than committing
      on a partial view.

    Ties keep the incumbent. A substitution has to *beat* the standing value to
    be fixed, so a flat site leaves the walk where it stood and the result stays
    reproducible rather than depending on which equal design happened to be
    measured first.

    Args:
        env: Supplies the anchor, alphabet, mutation budget and feasibility.
        order: The positions to walk, in order. ``None`` draws a permutation from
            ``seed``. Passed explicitly by
            `replicated_walk`, which needs its replicates to
            differ in exactly this and in nothing else.
        seed: Draws the order when one is not supplied, and nothing else -- the
            walk is deterministic once the order is fixed.
    """

    def __init__(
        self,
        env: MutationEnvironment,
        *,
        order: npt.NDArray[np.intp] | None = None,
        seed: int = 0,
    ) -> None:
        """Take the site order and open the first site around the anchor."""
        super().__init__(env)
        self._rng = np.random.default_rng(seed)
        self._order = site_order(env, self._rng) if order is None else np.asarray(order)
        self._cursor = -1
        self._incumbent = np.ascontiguousarray(env.parent)
        self._incumbent_value = -np.inf
        self._site: int | None = None
        self._designs: list[npt.NDArray[np.integer]] = []
        self._visited: list[int] = []
        self._named = 0
        self._advance()
        # The background's own measurement, at the head of the first plate. It is
        # the value every substitution at the first site is judged against, and
        # without it the walk would fix a residue at that site even where all of
        # them are deleterious.
        self._designs.insert(0, self._incumbent.copy())
        self._named += 1

    @property
    def name(self) -> str:
        """Short label, as the literature's arm name."""
        return "SingleStepWalk"

    @property
    def order(self) -> npt.NDArray[np.intp]:
        """The site order this walk is following."""
        return self._order.copy()

    @property
    def incumbent(self) -> Tokens:
        """The design every remaining site is saturated around."""
        return self._incumbent.copy()

    @property
    def best_value(self) -> float:
        """Value of the incumbent, or ``-inf`` before anything has been measured."""
        return self._incumbent_value

    @property
    def sites_fixed(self) -> tuple[int, ...]:
        """Positions the walk has opened, in the order it opened them.

        The invariant this reports is the arm's defining one: no position appears
        twice, so a residue committed to in an early round was never revised.
        """
        return tuple(self._visited)

    @property
    def requested(self) -> int:
        """Distinct designs the protocol has named so far.

        Returns:
            The background plus every substitution the walk has asked to have
            measured. This is the walk's own budget, and it is the number to read
            beside an oracle-call column: the campaign charges a full plate per
            round whatever the walk asked for.
        """
        return self._named

    def requests(self) -> list[npt.NDArray[np.integer]]:
        """The open site's designs that have not yet come back with a value."""
        return [design for design in self._designs if self._key(design) not in self._measured]

    def reanchored(self, env: MutationEnvironment) -> SingleStepWalk:
        """Carry the walk across the move, which restores its mutation budget.

        Almost everything here is anchor-free. The site order is a permutation of
        positions, the cursor is an index into it, the incumbent is a sequence and
        its value is a measurement; none of them is expressed relative to a
        parent. What *is* anchor-relative is which substitutions the environment
        can still build, and that is the half of the arm re-anchoring helps: the
        campaign anchors at the best design measured, so a walk standing there is
        spending its budget from zero again and can carry on down its permutation
        instead of stopping when the ball runs out.

        Rebuilding through the campaign's factory would instead restart the walk
        at site zero of a freshly drawn order, re-saturating residues it had
        already fixed and re-measuring designs the campaign has already paid for.

        Two cases are handled differently, and the split is what keeps "never
        revisit a site" true across a move:

        * **The incumbent survives.** The open site keeps whichever of its
          designs the new ball still admits, plus any already measured, and
          closes immediately if nothing is left to measure.
        * **The incumbent does not survive** -- the campaign anchored on a design
          this walk was not standing on, far enough away that the new ball
          excludes the incumbent. This is the ordinary case for a *replicate*,
          since at most one of several concurrent walks can be standing on the
          design the campaign chose. The walk then re-bases at the anchor,
          discarding the open site's partial measurements and moving to the
          *next* position in its own order. It keeps its order and its record of
          fixed sites, so it stays the replicate it was; re-opening a site it had
          already fixed is the one thing this arm must not do.

        Args:
            env: The re-anchored environment.

        Returns:
            A walk over ``env``, standing where this one stood.
        """
        moved = SingleStepWalk(env, order=self._order)
        moved._cursor = self._cursor
        moved._site = self._site
        moved._visited = list(self._visited)
        moved._named = self._named
        moved._rng = self._rng
        moved._proposals_made = self._proposals_made

        carried = carried_design(env, self._incumbent)
        moved._incumbent = np.ascontiguousarray(carried)
        moved._measured = dict(self._measured)
        if np.array_equal(carried, self._incumbent):
            moved._incumbent_value = self._incumbent_value
            moved._designs = [
                design
                for design in self._designs
                if moved._key(design) in moved._measured or within_budget(env, design)
            ]
            if not moved.requests():
                moved._fix_winner()
                moved._advance()
            return moved

        moved._incumbent_value = -np.inf
        moved._designs = []
        moved._advance()
        return moved

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Record the site's measurements, and fix its winner once it is saturated.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which gives
                the site no single order to pick a winner by.
        """
        flat = single_objective(values)
        for row, value in zip(np.asarray(sequences), flat, strict=False):
            key = self._key(row)
            if key not in self._measured:
                self._measured[key] = float(value)
        if self._designs and not self.requests():
            self._fix_winner()
            self._advance()

    def _on_remembered(self) -> None:
        """Close the open site if the measurements just adopted answered it."""
        if self._designs and not self.requests():
            self._fix_winner()
            self._advance()

    def _exhausted_message(self) -> str:
        """What to say when every site the budget admits has been fixed."""
        return (
            f"{self.name} has fixed every site its mutation budget admits and has nothing "
            f"left to measure; the walk named {self._named} designs and the campaign offered "
            f"more rounds than the protocol spends"
        )

    def _advance(self) -> None:
        """Open sites until one has something left for the campaign to measure.

        Ordinarily this opens exactly one. It loops because a site can arrive
        already answered: under `ReplicatedProtocol` a
        sibling walk may have had every substitution at this position measured
        already, and the right response is to *close* that site on the values in
        hand rather than to skip it -- skipping would leave the residue unfixed
        and the walk would carry a background it never chose.
        """
        while self._open_next_site():
            if self.requests():
                return
            self._fix_winner()

    def _open_next_site(self) -> bool:
        """Move to the next position with substitutions the budget admits.

        Positions whose substitutions would all leave the environment's Hamming
        ball are skipped rather than stalling the walk, which is what makes the
        mutation budget the only thing bounding it.

        Returns:
            Whether a site was opened. ``False`` means the order is spent.
        """
        while self._cursor + 1 < self._order.size:
            self._cursor += 1
            position = int(self._order[self._cursor])
            designs = substitutions_at(self._env, self._incumbent, position)
            if designs:
                self._site = position
                self._visited.append(position)
                self._designs = designs
                self._named += len(designs)
                return True
        self._site = None
        self._designs = []
        return False

    def _fix_winner(self) -> None:
        """Commit the saturated site to its best residue.

        The background's carried value competes with the substitutions through
        `_incumbent_value`, so a site at which nothing helps leaves the walk
        standing where it was. A strict improvement is required, which is what
        makes a flat site leave the walk in a reproducible place rather than on
        whichever equal design was measured first.
        """
        best: npt.NDArray[np.integer] | None = None
        best_value = -np.inf
        for design in self._designs:
            value = self._measured.get(self._key(design), -np.inf)
            if value > best_value:
                best_value, best = value, design
        if best is not None and best_value > self._incumbent_value:
            self._incumbent = np.ascontiguousarray(best).copy()
            self._incumbent_value = best_value

    def __repr__(self) -> str:
        """Name the walk and how far down its site order it has got."""
        return f"SingleStepWalk(sites_fixed={len(self._visited)}, requested={self._named})"


class Recombination(SaturationProtocol):
    """Saturate every site on one background, then combine the winners.

    Li et al.'s ``recomb``, which their methods call a naive recombination:
    *"This approach randomly samples the combinatorial space, independently
    optimizing each site within the context of the initial sequence and then
    combining the best substitutions from each site into a new variant."* Its
    cost is ``19 x n_site + 2 samples, including the initial and final
    variant`` -- the background every site is judged against, and the
    recombinant.

    The difference from
    [SingleStepWalk][evogfn.algorithms.baselines.directed_evolution.SingleStepWalk]
    is the background. Every substitution here is a substitution of the *same*
    sequence, so no site is ever measured in the presence of another site's
    winner, and the arm has no information at all about how the winners interact.
    Combining them is a bet that they are additive, and the single recombinant
    assay is the whole of the evidence it collects about that bet. That is the
    property that makes it the weaker of the two published walks on an epistatic
    landscape, and the reason both are worth running.

    How many sites
    --------------

    The site set is the largest the campaign could saturate and still have a
    plate left to measure the recombinant on, bounded by the sequence length and
    by the mutation budget. The mutation bound is not optional: the recombinant
    carries one substitution per improved site, so a site set wider than the
    budget would produce a design the environment cannot build.

    Bounding by the campaign rather than fixing a library size is the direct
    reading of the source's own budget formula, which states the arm's cost *as a
    function of* the number of sites. A lab chooses how many residues to saturate
    by what it can afford to screen, and this is that choice made explicitly
    rather than by truncating a protocol half way.

    Why the saturations are spread across rounds
    --------------------------------------------

    They are independent -- one background, one measurement each -- so which
    plate a saturation lands on changes nothing about the method, and laying them
    out across every round but the last is what leaves a round for the
    recombinant. Packing them into the earliest plates instead would finish the
    protocol with rounds to spare and nothing legal to spend them on.

    Args:
        env: Supplies the background, alphabet, mutation budget and feasibility.
        rounds: Rounds the campaign will run, which decides the layout above.
        batch_size: Designs the campaign measures per round, which with
            ``rounds`` decides how many sites are affordable.
        sites: The positions to saturate. ``None`` draws them from ``seed``.
            Passed explicitly by
            `replicated_recombination`, whose replicates
            differ in exactly this.
        seed: Draws the site set when one is not supplied.
    """

    def __init__(
        self,
        env: MutationEnvironment,
        *,
        rounds: int,
        batch_size: int,
        sites: Sequence[int] | None = None,
        seed: int = 0,
    ) -> None:
        """Take the site set and lay out every saturation the budget affords."""
        super().__init__(env)
        self._rng = np.random.default_rng(seed)
        self._rounds = max(int(rounds), 1)
        self._batch_size = max(int(batch_size), 1)
        self._background = np.ascontiguousarray(env.parent)
        # Sorted, always. A site set is a set -- the saturations are independent,
        # so the order they are listed in changes nothing about the method -- but
        # it does decide which of them each round's tranche contains. Two
        # replicates that drew the *same* sites in different orders would
        # otherwise release different tranches, and the pooled request would
        # cover the whole library in a fraction of the rounds the pacing was
        # laid out over, finishing the protocol early and leaving the campaign
        # with plates it could not fill.
        drawn = site_order(env, self._rng)[: self.affordable_sites()] if sites is None else sites
        self._sites = tuple(sorted(int(site) for site in drawn))
        self._by_site = {
            site: substitutions_at(env, self._background, site) for site in self._sites
        }
        self._library = [self._background.copy()] + [
            design for site in self._sites for design in self._by_site[site]
        ]
        self._released: list[npt.NDArray[np.integer]] = []
        self._round = 0
        self._recombinant: npt.NDArray[np.integer] | None = None
        self._combined = False

    @property
    def name(self) -> str:
        """Short label, as the literature's arm name."""
        return "Recombination"

    @property
    def sites(self) -> tuple[int, ...]:
        """Positions being saturated, in the order they were drawn."""
        return self._sites

    @property
    def recombinant(self) -> Tokens | None:
        """The combined design, once every site has been saturated.

        Returns:
            The design carrying each site's winning residue, or ``None`` before
            the saturation phase has finished -- and also where no site improved
            on the background, in which case the protocol nominates the
            background itself, which the campaign has already measured.
        """
        return None if self._recombinant is None else self._recombinant.copy()

    @property
    def requested(self) -> int:
        """Distinct designs the protocol has named so far.

        Returns:
            Saturations released, plus the recombinant once it exists. The
            protocol's full cost is
            ``(|alphabet| - 1) * len(sites) + RECOMBINATION_OVERHEAD`` before
            anything is excluded for leaving the mutation budget.
        """
        return len(self._released)

    def affordable_sites(self) -> int:
        """How many sites the campaign can saturate and still measure a recombinant.

        Returns:
            At least one site, and never more than the sequence length or the
            mutation budget allow. The mutation bound is what keeps the
            recombinant inside the environment's graph.
        """
        plates = max(self._rounds - 1, 1)
        per_site = max(self._env.alphabet.size - 1, 1)
        saturable = (plates * self._batch_size - 1) // per_site
        return max(1, min(self._env.sequence_length, self._env.max_mutations, saturable))

    def requests(self) -> list[npt.NDArray[np.integer]]:
        """This round's saturations, or the recombinant once they are all done."""
        pending = self._pending()
        if pending:
            return pending
        self._release()
        return self._pending()

    def reanchored(self, env: MutationEnvironment) -> Recombination:
        """Keep the background, whatever the campaign anchors on.

        This is the one protocol here that declines to follow a moved anchor, and
        the reason is the method rather than convenience: every site's winner is
        chosen by comparing measurements taken on **one** background, so a
        background that moved half way through would have the early sites judged
        against a different sequence from the late ones, and the recombination
        would combine winners that were never comparable.

        What the move does change is which designs are still inside the mutation
        budget. Anything the new ball excludes and that has not already been
        measured is dropped, on the same rule that excludes an out-of-budget
        substitution in the first place.

        In practice the anchor stays close: during saturation every design
        measured is one substitution from the background, so the campaign can
        only anchor at one of those, and the recombinant differs from such an
        anchor at one fewer site than it differs from the background.

        Args:
            env: The re-anchored environment.

        Returns:
            A recombination protocol over ``env``, saturating the same sites
            around the same background.
        """
        moved = Recombination(
            env,
            rounds=self._rounds,
            batch_size=self._batch_size,
            sites=self._sites,
        )
        moved._background = self._background
        moved._measured = dict(self._measured)
        moved._round = self._round
        moved._recombinant = self._recombinant
        moved._combined = self._combined
        moved._rng = self._rng
        moved._proposals_made = self._proposals_made

        moved._by_site = {
            site: [design for design in designs if moved._admits(design)]
            for site, designs in self._by_site.items()
        }
        moved._library = [self._background.copy()] + [
            design for site in moved._sites for design in moved._by_site[site]
        ]
        moved._released = [design for design in self._released if moved._admits(design)]
        return moved

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Record measurements against the designs this protocol released.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which gives
                a site no single order to pick a winner by.
        """
        flat = single_objective(values)
        for row, value in zip(np.asarray(sequences), flat, strict=False):
            key = self._key(row)
            if key not in self._measured:
                self._measured[key] = float(value)
        self._round += 1

    def _exhausted_message(self) -> str:
        """What to say when the saturations and the recombinant are both spent."""
        return (
            f"{self.name} has saturated every site it can afford and nominated its "
            f"recombinant; the protocol named {self.requested} designs and the campaign "
            f"offered more rounds than it spends"
        )

    def _admits(self, design: Tokens) -> bool:
        """Whether a design is still inside the budget, or has already been measured."""
        if self._key(design) in self._measured:
            return True
        return within_budget(self._env, design)

    def _pending(self) -> list[npt.NDArray[np.integer]]:
        """Released designs that have not yet come back with a value."""
        return [design for design in self._released if self._key(design) not in self._measured]

    def _release(self) -> None:
        """Hand out this round's tranche of saturations, or the recombinant.

        The tranche is recomputed each round against how many saturations are
        left and how many rounds remain before the last one, so a plate too small
        to hold a tranche pushes the remainder forward rather than dropping it.
        """
        released = {self._key(design) for design in self._released}
        outstanding = [
            design
            for design in self._library
            if self._key(design) not in released and self._key(design) not in self._measured
        ]
        if outstanding:
            plates_left = max(max(self._rounds - 1, 1) - self._round, 1)
            tranche = -(-len(outstanding) // plates_left)
            self._released.extend(outstanding[:tranche])
            return
        if self._combined:
            return
        self._combined = True
        self._recombinant = self._combine()
        if self._recombinant is not None:
            self._released.append(self._recombinant)

    def _combine(self) -> npt.NDArray[np.integer] | None:
        """Build the design carrying each site's winning residue.

        A site keeps the background's own residue unless one of its substitutions
        *beats the background's measured value*. The reference implementation
        reaches the same place by scanning all twenty amino acids at each
        position and taking the argmax, the incumbent's residue among them; the
        published count of nineteen per site is what tells you that scan costs no
        extra sample, and the separately itemised ``initial variant`` is what
        the comparison is against.

        Ties within a site keep the background, so the recombinant is a function
        of the measurements and not of plate order.

        The result is not checked for feasibility. A recombinant whose sites are
        individually feasible and jointly not is the additive assumption failing,
        which is this arm's own risk and belongs in the assay's answer rather
        than in a filter here.

        Returns:
            The recombinant, or ``None`` where no site improved and the design
            that results is the background the campaign has already measured, or
            where it falls outside the mutation budget. In both cases there is
            nothing new to measure and the protocol has nothing further to say.
        """
        design = self._background.copy()
        background = self._measured.get(self._key(self._background), -np.inf)
        for site, designs in self._by_site.items():
            best: npt.NDArray[np.integer] | None = None
            best_value = background
            for variant in designs:
                value = self._measured.get(self._key(variant), -np.inf)
                if value > best_value:
                    best, best_value = variant, value
            if best is not None:
                design[site] = best[site]
        if self._key(design) in self._measured:
            return None
        if not within_budget(self._env, design):
            return None
        return design

    def __repr__(self) -> str:
        """Name the protocol, its site count and whether it has combined yet."""
        return f"Recombination(sites={len(self._sites)}, combined={self._combined})"


class ReplicatedProtocol(Sampler):
    """Several copies of one protocol, at different site orders, sharing a plate.

    The budget policy described in the module docstring, kept in its own class so
    that it is visibly **ours** and the protocols underneath it stay exactly what
    their papers describe. Every replicate is the published method; what this
    adds is that more than one of them runs, which is what a lab with a plate and
    a one-plate protocol does.

    Three things it does and nothing else:

    * **Pools what the replicates ask for, round-robin.** Taking one design from
      each in turn rather than concatenating is what stops a plate too small to
      hold the pooled request from serving the first replicates forever and
      starving the last.
    * **Deduplicates.** Replicates share a background and can name the same
      substitution, so an undeduplicated pool would spend several wells on one
      design. Deduplication is also what makes the degenerate case correct
      without a special case: where the site sets cannot differ, the replicates
      name identical designs and the arm collapses back to exactly the
      protocol's own cost.
    * **Broadcasts measurements.** Every replicate is offered the whole plate and
      keeps what it recognises, so a design one replicate asked for and another
      happens to want is measured once and informs both.

    Args:
        protocols: The replicates to start with. At least one.
        env: The environment the replicates search, so that one started later can
            be built against the anchor the campaign has moved to rather than the
            one this object was created at.
        spawn: Builds a further replicate from an environment and an index.
            ``None`` fixes the replicate count at what was passed in. Supplied,
            it is called only when every existing replicate has finished and the
            campaign is still asking -- which is the same budget policy applied
            when it is needed rather than predicted, and is what a lab does with
            a protocol that has returned its answer and a plate still to fill.

    Raises:
        ValueError: If there are no replicates, since the arm would then have no
            method at all.
    """

    def __init__(
        self,
        protocols: Sequence[SaturationProtocol],
        *,
        env: MutationEnvironment,
        spawn: Callable[[MutationEnvironment, int], SaturationProtocol] | None = None,
    ) -> None:
        """Hold the replicates without running them."""
        super().__init__()
        if not protocols:
            raise ValueError("a replicated protocol needs at least one replicate")
        self._protocols = list(protocols)
        self._env = env
        self._spawn = spawn
        self._named: set[bytes] = set()
        self._measured: dict[bytes, float] = {}
        # Set once a fresh replicate turns out to have nothing to ask for, so the
        # arm stops trying within this anchor. Cleared by a move, because a new
        # ball puts designs in reach that the old one excluded.
        self._spawn_exhausted = False

    @property
    def name(self) -> str:
        """The protocol's own label, with how many copies of it are running."""
        return f"{self._protocols[0].name} x{len(self._protocols)}"

    @property
    def replicates(self) -> int:
        """How many copies of the protocol are running."""
        return len(self._protocols)

    @property
    def protocols(self) -> tuple[SaturationProtocol, ...]:
        """The replicates, for inspection."""
        return tuple(self._protocols)

    @property
    def requested(self) -> int:
        """Distinct designs the replicates have between them named.

        Counted as a set rather than summed over replicates, because the pooled
        request is deduplicated: two replicates naming one design have asked for
        one measurement, not two, and a sum would report a budget the arm never
        spent.
        """
        return len(self._named)

    @property
    def finished(self) -> bool:
        """Whether every replicate has nothing further to ask for."""
        return not self.requests()

    def requests(self) -> list[npt.NDArray[np.integer]]:
        """The pooled request, starting a further replicate if all are finished.

        Returns:
            The deduplicated union of what the replicates want measured. Empty
            only when no replicate has anything left *and* a fresh one would have
            nothing either -- which happens when the library the protocol can
            draw from has been measured in full, and is the honest end of the
            arm rather than a state to be filled with invented designs.
        """
        merged = self._pooled()
        if merged or self._spawn is None or self._spawn_exhausted:
            return merged
        for _ in range(_SPAWN_ATTEMPTS):
            fresh = self._spawn(self._env, len(self._protocols))
            fresh.remember(self._measured)
            # Kept only if it has something to ask for. A replicate born finished
            # -- every design in its library already measured -- would otherwise
            # accumulate one per proposal call and rename the arm after a count
            # that means nothing.
            if not fresh.requests():
                continue
            self._protocols.append(fresh)
            return self._pooled()
        self._spawn_exhausted = True
        return []

    def _pooled(self) -> list[npt.NDArray[np.integer]]:
        """The deduplicated union of the replicates' outstanding requests."""
        queues = [protocol.requests() for protocol in self._protocols]
        merged: list[npt.NDArray[np.integer]] = []
        seen: set[bytes] = set()
        for column in range(max((len(queue) for queue in queues), default=0)):
            for queue in queues:
                if column >= len(queue):
                    continue
                design = np.ascontiguousarray(queue[column])
                key = design.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(design)
        return merged

    def reanchored(self, env: MutationEnvironment) -> ReplicatedProtocol:
        """Move every replicate, keeping each one's own identity.

        A replicate keeps its site order across the move even when the campaign
        anchored somewhere it was not standing, so the replicates stay different
        experiments rather than converging on one. What each does about an
        incumbent the new ball excludes is its own business; see
        [SingleStepWalk.reanchored][evogfn.algorithms.baselines.directed_evolution.SingleStepWalk.reanchored].

        Args:
            env: The re-anchored environment.

        Returns:
            The replicates over ``env``, with this one's accounting carried so a
            campaign's proposal and request totals are not restarted by a move.
        """
        moved = ReplicatedProtocol(
            [protocol.reanchored(env) for protocol in self._protocols],
            env=env,
            spawn=self._spawn,
        )
        moved._named = set(self._named)
        moved._measured = dict(self._measured)
        moved._proposals_made = self._proposals_made
        return moved

    def propose(self, n: int) -> Tokens:
        """Return the pooled request, cycled to fill the plate.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array. Where the replicates between them
            want fewer than ``n`` designs the remainder repeats them, which the
            campaign charges as duplicate wells --
            [duplicate_fraction][evogfn.loop.ledger.RoundRecord.duplicate_fraction]
            is then reporting the share of the plate the protocol had no use for,
            and on a library whose site sets cannot differ that share is a fact
            about the method rather than about this class.

        Raises:
            RuntimeError: If every replicate is finished.
        """
        pending = self.requests()
        if not pending:
            raise RuntimeError(
                f"{self.name} has no replicate with anything left to measure; between them "
                f"they named {self.requested} designs and the campaign offered more rounds "
                f"than the protocol spends"
            )
        self._named.update(design.tobytes() for design in pending)
        self._count(n)
        return np.stack([pending[index % len(pending)] for index in range(n)])

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Offer the whole plate to every replicate.

        Each keeps only the designs it asked for, so a measurement one replicate
        wanted and another happens to want informs both without being charged
        twice.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective.
        """
        flat = single_objective(values)
        for row, value in zip(np.asarray(sequences), flat, strict=False):
            key = np.ascontiguousarray(row, dtype=self._env.parent.dtype).tobytes()
            if key not in self._measured:
                self._measured[key] = float(value)
        for protocol in self._protocols:
            protocol.observe(sequences, values)

    def __repr__(self) -> str:
        """Name the protocol and how many copies are running."""
        return f"ReplicatedProtocol({self._protocols[0]!r}, replicates={self.replicates})"


def walk_replicates(env: MutationEnvironment, batch_size: int) -> int:
    """How many concurrent walks a plate of ``batch_size`` supports.

    One walk asks for ``|alphabet| - 1`` designs per round, so this is how many
    of them a plate holds. Sizing to the plate rather than to the campaign is
    what keeps every round's request close to full: fewer replicates leaves wells
    to be filled by repetition, and more leaves replicates lagging a round behind
    for no gain.

    Args:
        env: Supplies the alphabet.
        batch_size: Designs the campaign measures per round.

    Returns:
        At least one.
    """
    return max(1, batch_size // max(env.alphabet.size - 1, 1))


def replicated_walk(
    env: MutationEnvironment, *, batch_size: int, seed: int = 0
) -> ReplicatedProtocol:
    """Traditional directed evolution, run at as many site orders as a plate holds.

    Args:
        env: Supplies the anchor, alphabet, mutation budget and feasibility.
        batch_size: Designs the campaign measures per round.
        seed: Draws the site orders. Distinct orders are drawn where the sequence
            is long enough to have them, so the replicates are different
            experiments rather than copies.

    Returns:
        The replicates behind one sampler.
    """
    rng = np.random.default_rng(seed)
    orders = distinct_site_orders(env, rng, walk_replicates(env, batch_size))

    def spawn(anchored: MutationEnvironment, _index: int) -> SaturationProtocol:
        """A further walk at a fresh order, for a campaign with budget to spare."""
        return SingleStepWalk(anchored, order=site_order(anchored, rng))

    return ReplicatedProtocol(
        [SingleStepWalk(env, order=order) for order in orders], env=env, spawn=spawn
    )


def recombination_replicates(
    env: MutationEnvironment, *, rounds: int, batch_size: int, n_site: int
) -> int:
    """How many concurrent recombination protocols the campaign can afford.

    Unlike the walk, this is sized against the *campaign* rather than the plate:
    a recombination protocol spends its saturations across every round but the
    last, so what bounds the replicate count is the total those rounds hold.

    Args:
        env: Supplies the alphabet.
        rounds: Rounds the campaign will run.
        batch_size: Designs it measures per round.
        n_site: Sites each replicate saturates.

    Returns:
        At least one.
    """
    plates = max(rounds - 1, 1)
    per_replicate = max(env.alphabet.size - 1, 1) * max(n_site, 1)
    return max(1, (plates * batch_size - 1) // per_replicate)


def replicated_recombination(
    env: MutationEnvironment, *, rounds: int, batch_size: int, seed: int = 0
) -> ReplicatedProtocol:
    """Li et al.'s recombination arm, at as many site sets as the campaign affords.

    Each replicate saturates a different set of residues, which on a sequence
    much longer than the mutation budget is a genuinely different experiment. On
    a library whose sites *are* the whole sequence there is only one set to draw,
    the replicates name identical designs, and the pooled request deduplicates
    back to the protocol's own cost -- which is the honest answer there.

    Args:
        env: Supplies the background, alphabet, mutation budget and feasibility.
        rounds: Rounds the campaign will run.
        batch_size: Designs it measures per round.
        seed: Draws the site sets.

    Returns:
        The replicates behind one sampler.
    """
    rng = np.random.default_rng(seed)
    n_site = Recombination(env, rounds=rounds, batch_size=batch_size).affordable_sites()
    count = recombination_replicates(env, rounds=rounds, batch_size=batch_size, n_site=n_site)
    orders = distinct_site_orders(env, rng, count)

    def spawn(anchored: MutationEnvironment, _index: int) -> SaturationProtocol:
        """A further protocol on fresh sites, saturating and combining at once.

        Built at ``rounds=2`` rather than the campaign's own count: a replicate
        started this late has no rounds to spread its saturations over, so it
        releases them in one plate and nominates its recombinant in the next,
        which is the most of the protocol that still fits.
        """
        return Recombination(
            anchored,
            rounds=2,
            batch_size=batch_size,
            sites=[int(site) for site in site_order(anchored, rng)[:n_site]],
        )

    return ReplicatedProtocol(
        [
            Recombination(
                env,
                rounds=rounds,
                batch_size=batch_size,
                sites=[int(site) for site in order[:n_site]],
            )
            for order in orders
        ],
        env=env,
        spawn=spawn,
    )
