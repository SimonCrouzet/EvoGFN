r"""What a task can actually reach, as opposed to what its optimum says.

Regret is reported against a landscape's `optimum`, and on an Ehrlich instance
that is $1.0$ by construction. But a campaign does not search the landscape: it
searches the [MutationEnvironment][evogfn.env.mutation.MutationEnvironment]
built around a parent, under a mutation budget and a feasibility mask. If the
planted optimum is not in *that* set, every method is measured against a target
none of them could have hit, and the regret column carries a constant nobody
computed.

Reaching reward $1.0$ on an Ehrlich instance means placing every residue of every
motif, which can sit far outside the mutation budget a task runs under.

What "attainable" means, and why it is often an interval
--------------------------------------------------------

The honest quantity is

$$
\mathrm{att}(\text{task}) \;=\; \max_{x \in \mathcal{R}} f(x),
\qquad
\mathcal{R} = \{\text{terminal states of the environment}\},
$$

and the regret floor is $\max f - \mathrm{att}$. When $\mathcal{R}$ is small
enough to enumerate this is a measurement, and
[MutationEnvironment.reachable_terminal_states][evogfn.env.mutation.MutationEnvironment.reachable_terminal_states]
makes it. It usually is not: a Hamming ball of any useful radius runs to many
orders of magnitude more designs than can be held, and reporting a searched
maximum as though it were the optimum would be this module's own error one level
down.

So [attainable_optimum][evogfn.benchmark.attainable.attainable_optimum] returns
an [AttainableOptimum][evogfn.benchmark.attainable.AttainableOptimum], which
carries `exact` (``None`` unless it was measured) beside a certified `upper` and
a searched `lower`. A caller cannot read a bound as a measurement without
choosing to.

The certified upper bound
-------------------------

For an Ehrlich instance the bound comes from the definition of the reward rather
than from a search. Write $c$ for the number of motifs, $k$ for the motif
length, $q$ for the quantisation and $B$ for the mutation budget. The reward is

$$
f(x) = \prod_{i=1}^{c} \frac{1}{q}\left\lfloor \frac{g_i(x)}{k/q} \right\rfloor,
\qquad
g_i(x) = \max_{\ell} \sum_{j=1}^{k} \mathbb{1}\{x_{\ell + s^{(i)}_j} = m^{(i)}_j\},
$$

so the question is how large the match counts $g_i$ can be made. Two facts bound
them, both properties of the construction graph rather than of any search:

**Per motif.** Every position of a placement is read once -- the offsets
$s^{(i)}$ are strictly increasing -- so one substitution raises $g_i$ by at most
one. Let $b_i = \max_\ell \sum_j \mathbb{1}\{p_{\ell+s_j} = m_j\}$ be the best
the *parent* already does for motif $i$. Then

$$
g_i \;\le\; \min(k,\; b_i + B).
$$

**Across motifs.** A substitution serves motif $i$ only if it sits inside that
motif's chosen placement carrying the token that placement requires. Let $D_i$
be the set of positions mutated for motif $i$ and $M \supseteq \bigcup_i D_i$
the mutations actually made, so $|M| \le B$. Bonferroni gives

$$
\sum_i |D_i| \;\le\; \Big|\bigcup_i D_i\Big| + \sum_{i<j} |D_i \cap D_j|
\;\le\; B + \sum_{i<j} S_{ij},
$$

where $S_{ij}$ is the largest number of positions two placements of motifs $i$
and $j$ can share *while requiring the same token there* -- computed exactly, by
enumerating placement pairs. Since $g_i \le b_i + |D_i|$, this caps the total
match gain. Without it the bound would be the per-motif one applied $c$ times
over, which spends the whole budget once per motif.

Maximising the product over the integer vectors $(g_1,\dots,g_c)$ that satisfy
both constraints is then a tiny enumeration, and its value is a certified upper
bound: it relaxes the feasibility mask entirely, and relaxes the requirement
that the shared positions be simultaneously realisable.

**Two motifs.** When $c = 2$ the placement pairs can be enumerated outright,
which prices sharing exactly instead of charging the Bonferroni worst case, and
usually beats the general bound. Both are computed and the smaller is kept.

The searched lower bound
------------------------

Anything the environment can construct is attainable, so a search gives a lower
bound for free -- provided it only ever walks legal edges. Both strategies here
do, by driving
[MutationEnvironment.forward_mask][evogfn.env.mutation.MutationEnvironment.forward_mask]
rather than re-deriving reachability:

* a **march** at the planted optimum, which is also a decision procedure --
  see [planted_optimum_reachable][evogfn.benchmark.attainable.planted_optimum_reachable];
* a **beam search** scored by motif completion, since the reward itself is flat
  across most of the neighbourhood and gives a greedy search nothing to follow.

A lower bound above the certified upper bound is a contradiction, and means the
derivation above is wrong rather than the search being lucky. It is checked, and
raises.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from evogfn.env.base import State
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.benchmark.tasks import Task
    from evogfn.core.types import Tokens
    from evogfn.landscapes.base import FitnessLandscape

#: Token cells the beam may hold, which is what sets its width. A fixed width
#: is the wrong knob: the cost of a step scales with the sequence length and so
#: does the number of substitutions that do nothing, so one number is either
#: unaffordable at $L = 256$ or a greedy walk in disguise at $L = 32$. Too narrow
#: a beam returns a lower bound below what an ordinary search has already reached
#: on the same task, which is a search failure reported as a property of the task.
DEFAULT_BEAM_CELLS = 20_000

#: Floor on the beam width, for the short sequences where the cell budget would
#: otherwise buy more states than the graph has.
MIN_BEAM_WIDTH = 48

#: Placements kept per motif when scoring the beam. The reward reads only the
#: best-matching placement, so ranking against a shortlist costs a fraction of
#: scanning all $L$ of them -- but the shortlist has to be long enough to hold
#: the placement the answer actually uses, which is not always one the parent
#: already matches well.
DEFAULT_PLACEMENTS = 16

#: Beam steps allowed per motif residue. Completing a motif takes at most $k$
#: substitutions; the rest of the allowance pays for the *enabling* mutations a
#: sparse transition matrix demands next to a residue before it can be placed.
DEPTH_PER_RESIDUE = 4

#: Beam steps without an improvement before the search stops. A large mutation
#: budget does not mean that many useful moves, and the unproductive tail is
#: where an unpatient search spends most of its time.
DEFAULT_PATIENCE = 6

#: Largest number of placement pairs the two-motif bound will enumerate. Above
#: this the general bound is used instead; it is looser but costs nothing.
MAX_PLACEMENT_PAIRS = 4_000_000

#: Token cells a bounded enumeration of the reachable set may hold. Expressed in
#: cells rather than sequences because what has to fit in memory is an
#: ``(n, length)`` array, and the suite's lengths differ by a factor of four.
MAX_REACHABLE_CELLS = 20_000_000

#: States whose outgoing edges are masked in one call, and edges turned into
#: child sequences in one call. The frontier of a mutation graph has orders of
#: magnitude more edges than it has states, so both have to be split or a single
#: layer allocates hundreds of gigabytes before the size guard is ever consulted.
_FRONTIER_CHUNK = 1024
_EDGE_CHUNK = 32_768


@dataclass(frozen=True, slots=True)
class AttainableOptimum:
    """The best objective value a task's search space actually contains.

    The three value fields are deliberately not interchangeable, because the
    failure this whole module exists to prevent is a bound being read as a
    measurement. `exact` is ``None`` unless the reachable set was enumerated or
    the interval closed; `lower` is witnessed by a sequence the environment
    built; `upper` is proved from the landscape's structure.

    Attributes:
        task: Task name, so a result cannot drift from what it describes.
        budget: Mutation budget the bound was computed under. Not read from the
            task, because auditing a stored result means asking what was
            attainable under the budget *that run* had.
        nominal: The optimum regret was reported against -- the landscape's own,
            which is what makes the floor a floor.
        lower: Best value a construction was actually found for.
        upper: Certified upper bound: no reachable design exceeds it.
        exact: The attainable optimum, when it is known rather than bracketed.
        method: How it was obtained, in words, for a report to quote.

    Raises:
        ValueError: If the bounds are inconsistent -- `lower` above `upper`, or
            an `exact` outside them. Either means a derivation is wrong, and a
            wrong bound is worse than no bound because it looks like a fact.
    """

    task: str
    budget: int
    nominal: float
    lower: float
    upper: float
    exact: float | None
    method: str

    def __post_init__(self) -> None:
        """Reject an interval that cannot describe a real quantity."""
        if self.lower > self.upper + 1e-12:
            raise ValueError(
                f"{self.task}: search reached {self.lower} but the certified upper bound is "
                f"{self.upper}; the bound derivation is wrong, not the search"
            )
        if self.exact is not None and not (self.lower - 1e-12 <= self.exact <= self.upper + 1e-12):
            raise ValueError(
                f"{self.task}: exact optimum {self.exact} lies outside [{self.lower}, {self.upper}]"
            )

    @property
    def is_exact(self) -> bool:
        """Whether the attainable optimum is known rather than bracketed."""
        return self.exact is not None

    @property
    def regret_floor(self) -> tuple[float, float]:
        """The regret no method on this task can get below.

        Returns:
            Lower and upper end of the floor. They coincide when the attainable
            optimum is exact. The *lower* end is the conservative claim: even in
            the best case for the benchmark, every arm's reported regret carries
            at least this much that no search could have removed.
        """
        return (self.nominal - self.upper, self.nominal - self.lower)

    @property
    def solvable_headroom(self) -> float:
        """How much of the attainable range an arm could still be losing.

        Returns:
            ``upper - lower``. Zero when the interval has closed, and the width
            of the honest uncertainty otherwise.
        """
        return self.upper - self.lower

    def regret_against_attainable(self, best: float) -> tuple[float, float]:
        """Rescore one measured result against what was reachable.

        Args:
            best: Best objective value an arm achieved.

        Returns:
            The regret interval ``(lower - best, upper - best)``. The first
            entry is the conservative one: if it is at or below zero the arm
            matched everything this module could construct, and the task has
            no demonstrated headroom left to separate methods with.
        """
        return (self.lower - best, self.upper - best)

    def solved_by(self, best: float, *, tolerance: float = 1e-9) -> bool:
        """Whether an arm has exhausted the task's demonstrated headroom.

        A task solved by an arm cannot rank the arms above it, so a comparison
        drawn on it is vacuous regardless of how many seeds it ran.

        Args:
            best: Best objective value an arm achieved.
            tolerance: Slack for floating-point equality.

        Returns:
            ``True`` when the arm reached at least the searched lower bound.
        """
        return best >= self.lower - tolerance

    def __repr__(self) -> str:
        """State the value or the interval, never one as the other."""
        value = (
            f"{self.exact:.4f} (exact)"
            if self.exact is not None
            else f"[{self.lower:.4f}, {self.upper:.4f}]"
        )
        floor = self.regret_floor
        return (
            f"{self.task} @ {self.budget} mutations: nominal {self.nominal:.4f}, "
            f"attainable {value}, regret floor [{floor[0]:.4f}, {floor[1]:.4f}]"
        )


def attainable_optimum(
    task: Task,
    *,
    budget: int | None = None,
    beam_width: int | None = None,
    placements: int = DEFAULT_PLACEMENTS,
    patience: int = DEFAULT_PATIENCE,
) -> AttainableOptimum:
    """The best objective value this task's search space actually contains.

    Args:
        task: The task to audit.
        budget: Mutation budget to evaluate under. Defaults to the task's own.
            Pass the historical value when rescoring stored results, since a
            record produced at a different budget was searching a different set.
        beam_width: Beam width for the searched lower bound. Defaults to
            `DEFAULT_BEAM_CELLS` divided by the sequence length, so a short
            landscape is searched properly and a long one stays affordable.
        placements: Placements kept per motif when scoring the beam.
        patience: Beam steps without improvement before the search stops.

    Returns:
        The attainable optimum, exact where it could be measured and bracketed
        where it could not.

    Raises:
        ValueError: If the landscape reports no optimum, so there is nothing to
            be regret against, or if the budget is not positive.
        NotImplementedError: If the reachable set is too large to enumerate and
            the landscape is not one whose structure yields a bound. Refusing is
            the point: a number invented here would be indistinguishable from a
            measured one downstream.
    """
    landscape = task.landscape()
    parent = task.parent(landscape)
    budget = task.max_mutations if budget is None else budget
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")

    optimum = landscape.optimum
    if optimum is None:
        raise ValueError(
            f"{task.name}: landscape reports no optimum, so it has no regret to have a floor"
        )
    nominal = float(np.max(optimum))

    transitions = _transitions(landscape)
    env = MutationEnvironment(
        parent,
        landscape.alphabet,
        max_mutations=min(budget, landscape.sequence_length),
        transitions=transitions,
    )

    measured = _exact_optimum(landscape, env, constrained=transitions is not None)
    if measured is not None:
        value, size = measured
        return AttainableOptimum(
            task=task.name,
            budget=budget,
            nominal=nominal,
            lower=value,
            upper=value,
            exact=value,
            method=f"exact: enumerated {size:,} reachable terminal states",
        )

    if not isinstance(landscape, EhrlichLandscape):
        raise NotImplementedError(
            f"{task.name}: the reachable set is too large to enumerate and "
            f"{type(landscape).__name__} offers no structure to bound it from"
        )

    upper, upper_how = _certified_upper_bound(landscape, parent, budget)
    lower, lower_how = _searched_lower_bound(
        landscape,
        env,
        parent,
        budget,
        beam_width=(
            max(MIN_BEAM_WIDTH, DEFAULT_BEAM_CELLS // landscape.sequence_length)
            if beam_width is None
            else beam_width
        ),
        placements=placements,
        patience=patience,
        ceiling=upper,
    )
    pinned = math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12)
    return AttainableOptimum(
        task=task.name,
        budget=budget,
        nominal=nominal,
        lower=lower,
        upper=min(upper, nominal),
        exact=lower if pinned else None,
        method=(
            f"pinned: {lower_how} met {upper_how}"
            if pinned
            else f"bounded: {lower_how} below {upper_how}"
        ),
    )


def per_round_budget(task: Task, *, rounds: int | None = None) -> int | None:
    r"""Smallest per-round mutation budget that could reach the planted optimum.

    A campaign that re-anchors on its best design each round -- see
    [MutationEnvironment.reanchored][evogfn.env.mutation.MutationEnvironment.reanchored]
    -- travels at most ``max_mutations`` per round but *cumulatively*, so $R$
    rounds reach $R \cdot b$ substitutions from the wild type rather than $b$.
    The budget needed to put the planted optimum inside the campaign's reach is
    then $\lceil D / R \rceil$ rather than $D$.

    This is a *counting* bound and only a necessary condition. It assumes every
    round spends its whole budget travelling toward the optimum, which requires
    the re-anchoring rule to pick anchors that make progress and the feasibility
    mask to permit them. What it is good for is settling a question a single
    fixed budget cannot: whether a search radius a wet lab would recognise can
    reach the answer at all.

    Args:
        task: The task to size.
        rounds: Rounds the campaign runs. Defaults to the task's protocol.

    Returns:
        The smallest per-round budget, or ``None`` for a landscape with no
        planted optimal sequence.

    Raises:
        ValueError: If ``rounds`` is not positive.
    """
    rounds = task.protocol.rounds if rounds is None else rounds
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    distance = planted_distance(task)
    return None if distance is None else -(-distance // rounds)


def reanchored_attainable(  # noqa: PLR0913 - the audit's knobs are its definition
    task: Task,
    *,
    per_round: int | None = None,
    rounds: int | None = None,
    beam_width: int | None = None,
    placements: int = DEFAULT_PLACEMENTS,
    patience: int = DEFAULT_PATIENCE,
) -> AttainableOptimum:
    """What the task could attain if the anchor moved between rounds.

    The fixed-anchor audit asks what one Hamming ball around the wild type
    contains. A re-anchoring campaign searches a *chain* of balls, each centred
    on the last round's best design, so the question changes and so does the
    answer: the reachable set is larger, and -- the part that budget alone
    cannot buy -- a position already mutated becomes mutable again, which is
    what lets a design reach through an adjacency that was blocked from the
    wild type.

    The bounds are the same two kinds as
    [attainable_optimum][evogfn.benchmark.attainable.attainable_optimum], with
    the budget replaced by the cumulative one:

    * the **upper** bound is the certified bound at ``rounds * per_round``,
      which holds because a round moves at most ``per_round`` positions, so no
      design the chain reaches is further than their product from the wild type;
    * the **lower** bound chains the beam search, re-anchoring on each round's
      best design exactly as
      [Campaign][evogfn.loop.campaign.Campaign] does under ``reanchor=True``.
      Every design in it is constructed through the environment's own masks, so
      it is attainable rather than merely close.

    Args:
        task: The task to audit.
        per_round: Mutation budget per round. Defaults to the task's own, which
            is the interesting question: what the current radius buys once the
            anchor is allowed to move.
        rounds: Rounds to chain. Defaults to the task's protocol.
        beam_width: Beam width. Defaults as for `attainable_optimum`.
        placements: Placements kept per motif when scoring the beam.
        patience: Beam steps without improvement before a round stops.

    Returns:
        The attainable optimum under re-anchoring, bracketed the same way.

    Raises:
        ValueError: If the landscape reports no optimum, or a budget is not
            positive.
        NotImplementedError: If the landscape is not one whose structure yields
            a bound.
    """
    rounds = task.protocol.rounds if rounds is None else rounds
    per_round = task.max_mutations if per_round is None else per_round
    if rounds < 1 or per_round < 1:
        raise ValueError(f"rounds and per_round must be positive, got {rounds} and {per_round}")

    landscape = task.landscape()
    if not isinstance(landscape, EhrlichLandscape):
        raise NotImplementedError(
            f"{task.name}: re-anchoring is audited only for Ehrlich instances, not "
            f"{type(landscape).__name__}"
        )
    optimum = landscape.optimum
    if optimum is None:
        raise ValueError(f"{task.name}: landscape reports no optimum")
    nominal = float(np.max(optimum))

    wild_type = task.parent(landscape)
    transitions = _transitions(landscape)
    cumulative = min(rounds * per_round, landscape.sequence_length)
    upper, upper_how = _certified_upper_bound(landscape, wild_type, cumulative)
    width = (
        max(MIN_BEAM_WIDTH, DEFAULT_BEAM_CELLS // landscape.sequence_length)
        if beam_width is None
        else beam_width
    )

    anchor = np.asarray(wild_type)
    best = float(np.asarray(landscape.evaluate(anchor[None, :]))[0, 0])
    for _ in range(rounds):
        if best >= upper - 1e-12:
            break
        env = MutationEnvironment(
            anchor,
            landscape.alphabet,
            max_mutations=min(per_round, landscape.sequence_length),
            transitions=transitions,
        )
        found, anchor = _beam_search(
            landscape,
            env,
            anchor,
            beam_width=width,
            placements=placements,
            patience=patience,
            ceiling=upper,
        )
        best = max(best, found)

    pinned = math.isclose(best, upper, rel_tol=0.0, abs_tol=1e-12)
    return AttainableOptimum(
        task=task.name,
        budget=cumulative,
        nominal=nominal,
        lower=best,
        upper=min(upper, nominal),
        exact=best if pinned else None,
        method=(
            f"{'pinned' if pinned else 'bounded'}: {rounds} re-anchored rounds of {per_round} "
            f"mutations, width-{width} beam, against {upper_how} at {cumulative}"
        ),
    )


def planted_distance(task: Task) -> int | None:
    """How many substitutions separate the parent from the planted optimum.

    This is the number a mutation budget has to clear for regret to mean
    anything on an Ehrlich task.

    Args:
        task: The task to measure.

    Returns:
        The Hamming distance, or ``None`` for a landscape with no planted
        optimal sequence.
    """
    landscape = task.landscape()
    if not isinstance(landscape, EhrlichLandscape):
        return None
    return int(np.count_nonzero(task.parent(landscape) != landscape.optimal_sequence))


def planted_optimum_reachable(task: Task, *, budget: int | None = None) -> bool | None:
    """Whether the environment can actually construct the planted optimum.

    Being inside the mutation budget is necessary and *not* sufficient, which is
    the part that surprises. Mutations are applied one position at a time and
    each position may be mutated only once, so every intermediate must satisfy
    the transition matrix on its own; a target can sit well within budget and
    still have no legal order.

    The greedy walk here decides the question rather than merely attempting it.
    A pending position $p$ may be set to its target token when both current
    neighbours permit that token, and a neighbour already at its target always
    does -- the target sequence is feasible, so the pair $(x^*_{p-1}, x^*_p)$ is
    allowed. Availability is therefore monotone in the set of placed positions,
    which only grows, so the walk reaches the unique maximal closure and getting
    stuck there proves no ordering succeeds.

    Args:
        task: The task to test.
        budget: Mutation budget to test under. Defaults to the task's own.

    Returns:
        ``True`` when a legal order exists, ``False`` when none does, and
        ``None`` for a landscape with no planted optimal sequence.
    """
    landscape = task.landscape()
    if not isinstance(landscape, EhrlichLandscape):
        return None
    parent = task.parent(landscape)
    budget = task.max_mutations if budget is None else budget
    env = MutationEnvironment(
        parent,
        landscape.alphabet,
        max_mutations=min(budget, landscape.sequence_length),
        transitions=_transitions(landscape),
    )
    return _march(env, parent, landscape.optimal_sequence, budget) is not None


def _transitions(landscape: FitnessLandscape) -> npt.NDArray[np.floating] | None:
    """The landscape's feasibility matrix, when it constrains anything.

    Args:
        landscape: The landscape being searched.

    Returns:
        The matrix whose zeros mark forbidden adjacencies, or ``None``.
    """
    matrix = getattr(landscape, "transition_matrix", None)
    return None if matrix is None else np.asarray(matrix)


def _exact_optimum(
    landscape: FitnessLandscape, env: MutationEnvironment, *, constrained: bool
) -> tuple[float, int] | None:
    """Enumerate the reachable set and take its maximum, where that is possible.

    Three attempts, cheapest first. Unconstrained, the reachable set *is* the
    Hamming ball and has a closed form. Constrained, the environment's own walk
    is preferred where its guard admits it, since it is the implementation the
    rest of the package is tested against. Only then the bounded walk below,
    which exists for the case that guard turns away.

    Args:
        landscape: The landscape being searched.
        env: The environment defining the reachable set.
        constrained: Whether a transition matrix masks edges out.

    Returns:
        The attainable optimum and how many terminal states were enumerated, or
        ``None`` when the set is too large to enumerate.
    """
    if not constrained:
        try:
            states = env.enumerate_terminal_states()
        except ValueError:
            return None
        return float(np.asarray(landscape.evaluate(states)).max()), int(states.shape[0])

    try:
        reachable = env.reachable_terminal_states()
    except ValueError:
        return _bounded_reachable_optimum(landscape, env)
    return float(np.asarray(landscape.evaluate(reachable)).max()), int(reachable.shape[0])


def _bounded_reachable_optimum(
    landscape: FitnessLandscape, env: MutationEnvironment
) -> tuple[float, int] | None:
    r"""Walk the reachable set, giving up on size rather than on the Hamming ball.

    [MutationEnvironment.reachable_terminal_states][evogfn.env.mutation.MutationEnvironment.reachable_terminal_states]
    refuses up front on the size of the Hamming ball, which is the honest bound
    on what its unbatched walk could allocate. But a sparse transition matrix
    makes the ball a wild over-estimate of the set -- the constructible part of
    it can be smaller by many orders of magnitude, and enumerable when the ball
    is not. Refusing there gives up an exact answer precisely where
    feasibility bites hardest -- which is the one place a benchmark most needs
    to know what it was really asking for.

    So this walks the same graph through the same masks, and gives up on the
    number of states actually discovered instead. Nothing about reachability is
    re-derived here; every edge comes from
    [MutationEnvironment.forward_mask][evogfn.env.mutation.MutationEnvironment.forward_mask].

    Args:
        landscape: The landscape being searched.
        env: The environment defining the reachable set.

    Returns:
        The attainable optimum and the size of the reachable set, or ``None``
        when it exceeds `MAX_REACHABLE_CELLS` cells.
    """
    cap = max(1, MAX_REACHABLE_CELLS // env.sequence_length)
    parent = env.parent
    seen = {parent.tobytes()}
    frontier = parent[None, :].copy()
    best = -np.inf
    total = 1

    while frontier.shape[0]:
        discovered: list[Tokens] = []
        for start in range(0, frontier.shape[0], _FRONTIER_CHUNK):
            block = frontier[start : start + _FRONTIER_CHUNK]
            mask = env.forward_mask(
                State(sequences=block, stopped=np.zeros(block.shape[0], dtype=np.bool_))
            )
            terminal = block[np.nonzero(mask[:, env.stop_action])[0]]
            if terminal.shape[0]:
                best = max(best, float(np.asarray(landscape.evaluate(terminal)).max()))

            rows, actions = np.nonzero(mask[:, : env.n_mutation_actions])
            for edge in range(0, rows.size, _EDGE_CHUNK):
                span = slice(edge, edge + _EDGE_CHUNK)
                children = block[rows[span]].copy()
                children[np.arange(rows[span].size), actions[span] // env.alphabet.size] = (
                    actions[span] % env.alphabet.size
                )
                # Every forward substitution raises the mutation count by one, so
                # a state first seen in this layer is never reached from a later
                # one and this dedup loses no edge worth walking.
                for child in np.unique(children, axis=0):
                    key = child.tobytes()
                    if key not in seen:
                        seen.add(key)
                        discovered.append(child)
                if len(discovered) + total > cap:
                    return None
        total += len(discovered)
        frontier = (
            np.stack(discovered)
            if discovered
            else np.empty((0, env.sequence_length), dtype=parent.dtype)
        )
    return best, total


def _motif_profile(landscape: EhrlichLandscape, parent: Tokens) -> npt.NDArray[np.int64]:
    """Best match count the parent already achieves, per motif.

    Args:
        landscape: The Ehrlich instance.
        parent: The sequence campaigns start from.

    Returns:
        A ``(n_motifs,)`` array of match counts, each in ``[0, motif_length]``.
    """
    return np.asarray(
        [int(counts.max()) for counts in _placement_matches(landscape, parent)],
        dtype=np.int64,
    )


def _placement_matches(
    landscape: EhrlichLandscape, sequence: Tokens
) -> list[npt.NDArray[np.int64]]:
    """Match count at every placement of every motif, for one sequence.

    Args:
        landscape: The Ehrlich instance.
        sequence: A single sequence of the landscape's length.

    Returns:
        One ``(n_placements,)`` array per motif.
    """
    return [
        np.asarray(
            (sequence[positions] == motif[None, :]).sum(axis=1),
            dtype=np.int64,
        )
        for motif, positions in zip(landscape.motifs, _placement_positions(landscape), strict=True)
    ]


def _placement_positions(landscape: EhrlichLandscape) -> list[npt.NDArray[np.int64]]:
    """Absolute positions each placement of each motif reads.

    Args:
        landscape: The Ehrlich instance.

    Returns:
        One ``(n_placements, motif_length)`` array per motif.
    """
    positions = []
    for spacing in landscape.spacings:
        n_placements = landscape.sequence_length - int(spacing[-1])
        positions.append(
            np.arange(n_placements, dtype=np.int64)[:, None] + np.asarray(spacing, dtype=np.int64)
        )
    return positions


def _sharing_capacity(landscape: EhrlichLandscape, parent: Tokens) -> int:
    """Most positions any two motifs can usefully mutate in common.

    A substitution serves two motifs at once only when both chosen placements
    read that position and require the same token there, and the parent does not
    already carry it. This counts the largest such overlap for every motif pair
    exactly, by enumerating placement pairs -- which is what lets the joint
    bound charge for sharing instead of assuming it away.

    Args:
        landscape: The Ehrlich instance.
        parent: The sequence campaigns start from.

    Returns:
        The sum over motif pairs of the largest shared-position count. Zero when
        no mutation can ever serve two motifs, in which case the budget splits
        cleanly and the joint bound becomes a plain knapsack.
    """
    v = landscape.alphabet.size
    keys = []
    for motif, positions in zip(landscape.motifs, _placement_positions(landscape), strict=True):
        # One integer per requirement, so "same position, same token" is a
        # single comparison. Requirements the parent already satisfies are
        # marked with a value no other key can take, since fixing them costs
        # nothing and so cannot be shared.
        key = positions * v + np.asarray(motif, dtype=np.int64)[None, :]
        keys.append(np.where(parent[positions] == motif[None, :], -1, key))

    total = 0
    for left, right in itertools.combinations(keys, 2):
        shared = (left[:, None, :, None] == right[None, :, None, :]) & (left[:, None, :, None] >= 0)
        total += int(shared.any(axis=3).sum(axis=2).max())
    return total


def _levels(landscape: EhrlichLandscape, matches: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Quantised satisfaction for a vector of match counts.

    Args:
        landscape: The Ehrlich instance.
        matches: Match counts, one per motif.

    Returns:
        The per-motif reward factors, each a multiple of ``1 / quantization``.
    """
    step = landscape.motif_length // landscape.quantization
    return np.asarray(matches // step, dtype=np.float64) / landscape.quantization


def _certified_upper_bound(
    landscape: EhrlichLandscape, parent: Tokens, budget: int
) -> tuple[float, str]:
    """No reachable design scores above this, proved from the reward's shape.

    Two bounds are computed and the smaller kept. Both relax the feasibility
    mask -- masking only removes designs -- so both remain valid under it.

    Args:
        landscape: The Ehrlich instance.
        parent: The sequence campaigns start from.
        budget: Mutation budget.

    Returns:
        The bound and a phrase naming which argument produced it.
    """
    best = _motif_profile(landscape, parent)
    general = _budget_split_bound(landscape, best, budget, _sharing_capacity(landscape, parent))
    pairwise = _placement_pair_bound(landscape, parent, budget)
    if pairwise is not None and pairwise < general:
        return pairwise, "the placement-pair bound"
    return general, "the budget-split bound"


def _budget_split_bound(
    landscape: EhrlichLandscape,
    best: npt.NDArray[np.int64],
    budget: int,
    sharing: int,
) -> float:
    """Maximise the reward over match vectors the budget could pay for.

    The two constraints are the ones derived in this module's docstring: one
    substitution buys at most one match in any single motif, and the mutations
    bought across motifs cannot overlap more than the placements physically
    allow.

    Args:
        landscape: The Ehrlich instance.
        best: Match count the parent already achieves, per motif.
        budget: Mutation budget.
        sharing: Total positions motif pairs can usefully mutate in common.

    Returns:
        The largest reward any admissible match vector attains.
    """
    k = landscape.motif_length
    caps = np.minimum(k, best + budget)
    allowance = budget + sharing

    value = 0.0
    for matches in itertools.product(*(range(int(cap) + 1) for cap in caps)):
        counts = np.asarray(matches, dtype=np.int64)
        if int(np.maximum(counts - best, 0).sum()) > allowance:
            continue
        value = max(value, float(np.prod(_levels(landscape, counts))))
    return value


def _placement_pair_bound(landscape: EhrlichLandscape, parent: Tokens, budget: int) -> float | None:
    """The same question priced exactly, when there are only two motifs.

    With $c = 2$ every pair of placements can be visited, so the cost of a
    target is the two per-motif costs less the positions the pair genuinely
    shares -- rather than the worst case any pair could offer, which is what the
    general bound has to assume. It stays an upper bound because it still
    ignores the feasibility mask, and because it does not charge for a mutation
    made for one motif destroying a match the other placement already had.

    Args:
        landscape: The Ehrlich instance.
        parent: The sequence campaigns start from.
        budget: Mutation budget.

    Returns:
        The bound, or ``None`` when the instance has other than two motifs or
        the pair count is too large to enumerate.
    """
    if landscape.n_motifs != 2:  # noqa: PLR2004 - this bound is the two-motif case by definition
        return None
    positions = _placement_positions(landscape)
    if int(positions[0].shape[0]) * int(positions[1].shape[0]) > MAX_PLACEMENT_PAIRS:
        return None

    k = landscape.motif_length
    v = landscape.alphabet.size
    keys = [
        np.where(parent[pos] == motif[None, :], -1, pos * v + np.asarray(motif, dtype=np.int64))
        for motif, pos in zip(landscape.motifs, positions, strict=True)
    ]
    matched = [np.asarray((key < 0).sum(axis=1), dtype=np.int64) for key in keys]
    shared = np.asarray(
        (
            (keys[0][:, None, :, None] == keys[1][None, :, None, :])
            & (keys[0][:, None, :, None] >= 0)
        )
        .any(axis=3)
        .sum(axis=2),
        dtype=np.int64,
    )

    value = 0.0
    for first, second in itertools.product(range(k + 1), repeat=2):
        need = (
            np.maximum(first - matched[0], 0)[:, None],
            np.maximum(second - matched[1], 0)[None, :],
        )
        cost = need[0] + need[1] - np.minimum(shared, np.minimum(need[0], need[1]))
        if bool((cost <= budget).any()):
            counts = np.asarray([first, second], dtype=np.int64)
            value = max(value, float(np.prod(_levels(landscape, counts))))
    return value


def _searched_lower_bound(  # noqa: PLR0913 - the search's knobs are its definition
    landscape: EhrlichLandscape,
    env: MutationEnvironment,
    parent: Tokens,
    budget: int,
    *,
    beam_width: int,
    placements: int,
    patience: int,
    ceiling: float,
) -> tuple[float, str]:
    """The best value a construction was actually found for.

    Args:
        landscape: The Ehrlich instance.
        env: The environment, which decides every move.
        parent: The sequence campaigns start from.
        budget: Mutation budget.
        beam_width: Beam width.
        placements: Placements kept per motif when scoring.
        patience: Steps without improvement before stopping.
        ceiling: Stop early on reaching this; nothing above it is reachable.

    Returns:
        The value and a phrase naming which strategy found it.
    """
    marched = _march(env, parent, landscape.optimal_sequence, budget)
    if marched is not None:
        return float(
            np.asarray(landscape.evaluate(marched[None, :]))[0, 0]
        ), "a march to the planted optimum"

    value, _ = _beam_search(
        landscape,
        env,
        parent,
        beam_width=beam_width,
        placements=placements,
        patience=patience,
        ceiling=ceiling,
    )
    return value, f"a width-{beam_width} beam search"


def _march(env: MutationEnvironment, parent: Tokens, target: Tokens, budget: int) -> Tokens | None:
    """Walk from the parent to a target sequence, one legal mutation at a time.

    Complete rather than heuristic: see
    [planted_optimum_reachable][evogfn.benchmark.attainable.planted_optimum_reachable]
    for why a stuck greedy walk proves no ordering exists.

    Args:
        env: The environment, which decides every move.
        parent: Where the walk starts.
        target: Where it is trying to get to.
        budget: Mutation budget.

    Returns:
        The target when it was constructed, ``None`` when no ordering reaches
        it -- whether because it is out of budget or out of reach.
    """
    pending = [int(p) for p in np.nonzero(np.asarray(parent) != np.asarray(target))[0]]
    if len(pending) > budget:
        return None

    v = env.alphabet.size
    sequence = np.asarray(parent).copy()
    while pending:
        mask = env.forward_mask(
            State(sequences=sequence[None, :], stopped=np.zeros(1, dtype=np.bool_))
        )
        legal = [p for p in pending if mask[0, p * v + int(target[p])]]
        if not legal:
            return None
        sequence[legal[0]] = target[legal[0]]
        pending.remove(legal[0])
    return sequence


def _beam_search(  # noqa: PLR0913 - the search's knobs are its definition
    landscape: EhrlichLandscape,
    env: MutationEnvironment,
    parent: Tokens,
    *,
    beam_width: int,
    placements: int,
    patience: int,
    ceiling: float,
) -> tuple[float, Tokens]:
    """Search for a high-scoring reachable design, scored by motif completion.

    The reward is flat over most of a design's neighbourhood -- it is a product
    of quantised terms, so nothing changes until a whole quantisation step is
    crossed -- and a search following it directly has no gradient to follow.
    The beam is ranked by [_beam_score][evogfn.benchmark.attainable._beam_score]
    instead, which counts residues placed and residues that could be placed
    next, and only the survivors are scored by the landscape itself.

    Args:
        landscape: The Ehrlich instance.
        env: The environment, which decides every move.
        parent: The sequence campaigns start from.
        beam_width: Beam width.
        placements: Placements kept per motif when scoring.
        patience: Steps without improvement before stopping.
        ceiling: Stop on reaching this; nothing above it is reachable.

    Returns:
        The best value found -- never below the parent's own -- and a design to
        carry forward. A value alone cannot start the next round of a
        re-anchoring audit, and on a landscape that is flat almost everywhere
        *which* of the designs tied at that value is carried forward is most of
        what decides whether the next round gets anywhere. So the design
        returned is the reward-best one, and where the round improved on nothing
        it is the head of the beam: the state the completion heuristic rates
        highest, which is the one with the most residues within reach.
    """
    targets = _scoring_placements(landscape, parent, placements)
    beam = np.asarray(parent)[None, :].copy()
    best = float(np.asarray(landscape.evaluate(beam))[0, 0])
    witness: Tokens | None = None
    stale = 0

    depth = min(env.max_mutations, DEPTH_PER_RESIDUE * landscape.n_motifs * landscape.motif_length)
    for _ in range(depth):
        if best >= ceiling - 1e-12 or stale >= patience:
            break
        children = _expand(env, beam)
        if children is None:
            break
        score = _beam_score(landscape, children, parent, targets, _transitions(landscape))
        beam = children[np.argsort(-score, kind="stable")[:beam_width]]
        values = np.asarray(landscape.evaluate(beam)).reshape(beam.shape[0])
        found = float(values.max())
        improved = found > best
        if improved:
            # `argmax` is stable and the beam is in heuristic order, so among
            # designs tied on reward this is the one the heuristic likes best.
            best, witness = found, beam[int(np.argmax(values))].copy()
        stale = 0 if improved else stale + 1
    head: Tokens = beam[0].copy()
    return best, head if witness is None else witness


def _expand(env: MutationEnvironment, beam: Tokens) -> Tokens | None:
    """Every distinct state one legal substitution away from the beam.

    Args:
        env: The environment, whose mask is the only definition of legal used.
        beam: The current states.

    Returns:
        The distinct successors, or ``None`` when the beam has none.
    """
    mask = env.forward_mask(State(sequences=beam, stopped=np.zeros(beam.shape[0], dtype=np.bool_)))
    rows, actions = np.nonzero(mask[:, : env.n_mutation_actions])
    if rows.size == 0:
        return None
    children = beam[rows].copy()
    children[np.arange(rows.size), actions // env.alphabet.size] = actions % env.alphabet.size
    unique: Tokens = np.unique(children, axis=0)
    return unique


def _scoring_placements(
    landscape: EhrlichLandscape, parent: Tokens, placements: int
) -> list[npt.NDArray[np.int64]]:
    """Placements worth steering the beam toward, per motif.

    The ones the parent already matches best, plus every placement the planted
    optimum satisfies in full -- those are jointly satisfiable by construction,
    which is exactly the property a search for a high product needs.

    Args:
        landscape: The Ehrlich instance.
        parent: The sequence campaigns start from.
        placements: How many parent-ranked placements to keep per motif.

    Returns:
        One ``(kept, motif_length)`` array of absolute positions per motif.
    """
    positions = _placement_positions(landscape)
    parent_matches = _placement_matches(landscape, parent)
    planted_matches = _placement_matches(landscape, landscape.optimal_sequence)

    kept = []
    for index, place in enumerate(positions):
        ranked = np.argsort(-parent_matches[index], kind="stable")[:placements]
        planted = np.nonzero(planted_matches[index] == landscape.motif_length)[0]
        chosen = np.unique(np.concatenate([ranked, planted]))
        kept.append(place[chosen])
    return kept


def _beam_score(
    landscape: EhrlichLandscape,
    sequences: Tokens,
    parent: Tokens,
    targets: list[npt.NDArray[np.int64]],
    transitions: npt.NDArray[np.floating] | None,
) -> npt.NDArray[np.float64]:
    """Rank candidate states by how close they are to completing every motif.

    Three terms, in decreasing order of authority: the reward restricted to the
    scored placements, the residues actually placed, and the residues that could
    be placed *next* without breaking an adjacency. The last is what keeps a
    sparse transition matrix searchable -- a mutation beside a motif residue
    pays nothing immediately and is the only way to unlock it.

    Args:
        landscape: The Ehrlich instance.
        sequences: The states to rank.
        parent: The sequence campaigns start from.
        targets: Positions to score, per motif.
        transitions: Feasibility matrix, or ``None``.

    Returns:
        One score per state. Larger is better.
    """
    length = landscape.sequence_length
    permitted = None if transitions is None else transitions > 0

    reward = np.ones(sequences.shape[0], dtype=np.float64)
    placed = np.zeros(sequences.shape[0], dtype=np.float64)
    unlockable = np.zeros(sequences.shape[0], dtype=np.float64)

    for motif, positions in zip(landscape.motifs, targets, strict=True):
        wanted = np.asarray(motif, dtype=np.int64)[None, None, :]
        current = sequences[:, positions]
        matches = current == wanted

        ready = (current == np.asarray(parent)[positions][None, :, :]) & ~matches
        if permitted is not None:
            left = np.where(
                positions > 0,
                permitted[sequences[:, np.maximum(positions - 1, 0)], wanted],
                True,
            )
            right = np.where(
                positions < length - 1,
                permitted[wanted, sequences[:, np.minimum(positions + 1, length - 1)]],
                True,
            )
            ready = ready & left & right

        counts = matches.sum(axis=2)
        reward *= _levels(landscape, counts.max(axis=1))
        placed += counts.max(axis=1)
        unlockable += (matches | ready).sum(axis=2).max(axis=1)

    return reward * 1e6 + placed * 1e3 + unlockable
