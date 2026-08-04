"""CMA-ES: the classical method that adapts a *distribution* rather than a set.

Every other baseline here carries points -- a population, a current design, a
training set. CMA-ES (Hansen & Ostermeier, *Evol. Comput.* 2001) carries a
probability distribution and updates its shape from the ranking of what it drew.
That makes it the closest thing in classical optimisation to what a GFlowNet
does, and therefore the baseline that tests the part of the claim the others
cannot reach: adapting a distribution over sequences is not itself novel, and a
method whose advantage is "it learns a sampler" has to beat the fifty-year-old
method that also learns a sampler.

What it does not do is sample proportionally to reward. CMA-ES contracts onto a
single optimum by construction -- that is the point of the step-size control --
so it should win on best-found and lose badly on diversity and on distributional
distance. If it does not lose on those, the sampling claim is in trouble.

Discrete sequences via a continuous relaxation
----------------------------------------------

CMA-ES optimises in ``R^d``. The standard adaptation to categorical variables is
to relax: carry a real matrix of shape ``(length, vocabulary)``, treat it as
per-position logits, and read a sequence off it by taking the argmax at each
position. The Gaussian that CMA-ES maintains supplies the exploration, so no
extra sampling temperature is needed -- a position whose logits are close
together flips between tokens under the noise, and one whose logits have
separated stops flipping. The relaxation is where the method's assumptions are
weakest and is worth stating plainly: rank information is fed back into a
Gaussian over logits, and a Gaussian over logits is not a natural model of an
epistatic landscape.

What the relaxation cannot express, and where that has to be fixed
-------------------------------------------------------------------

A separable Gaussian over per-position logits is a **product** distribution, and
per-position argmax is a **product** map: the token at position ``i`` is a
function of the logit block at position ``i`` and of nothing else. A transition
constraint is not a product set -- it couples adjacent positions, and which
tokens position ``i`` may take depends on what position ``i - 1`` took. No mean
and no diagonal covariance can therefore make the image of that decoder land
inside the feasible set, except in the degenerate case where the constraint
happens to factorise. This is a property of the relaxation, not an oversight in
its implementation, and it is the honest limit of the method on a constrained
space: **the search distribution cannot represent feasibility.**

The decoder is a different matter, and it is where the baseline can be left for
dead. Decoding is already a *projection* -- the argmax is followed by a
projection onto the mutation budget -- and a decoder that stops there does not
project onto the other constraint at all. On a sparse instance the consequence
is total rather than merely lossy: the zero mean makes the initial argmax
uniform over the alphabet, a uniform sequence satisfies a long adjacency chain
with vanishing probability, and every design the arm emits is infeasible.
Rejection is no remedy at that density; it is not expensive, it is impossible.

So the projection is completed rather than the method abandoned. Three
constraints have to hold at once and each is local: the adjacency rule is a
**first-order chain** condition on a token pair, the mutation budget is a
cardinality condition, and *constructibility* -- that some ordering of the
substitutions keeps every intermediate feasible -- reduces, by the argument in
[is_constructible][evogfn.env.mutation.MutationEnvironment.is_constructible], to
one further condition per adjacent pair. So the highest-scoring sequence subject
to all three is the Viterbi path of a dynamic program over
``(position, token, counter)``, computed exactly in ``O(n * L * V^2 * K)`` by
``_project_onto_constructible``. It always succeeds, because the anchor itself is
a feasible sequence at zero substitutions and so constrains no ordering, which
makes the set the projection searches non-empty by construction. The cost is wall
clock rather than oracle calls or proposals -- ``proposals_made`` stays at one
per design -- and
[repaired_fraction][evogfn.algorithms.baselines.cmaes.CMAES.repaired_fraction]
reports how often the raw argmax was unbuildable, which is the number that says
how much of the arm's behaviour is the relaxation's and how much the
projection's.

Which set the projection targets is the whole of the comparison
---------------------------------------------------------------

The set has to be the environment's own, and the two candidate answers differ.
[is_reachable][evogfn.env.mutation.MutationEnvironment.is_reachable] is budget
plus endpoint feasibility; the environment's masks additionally require a legal
construction order, which is what
[reachable_terminal_states][evogfn.env.mutation.MutationEnvironment.reachable_terminal_states]
enumerates. **Projecting onto the looser set is not a milder version of the
constraint, it is a different search space**: every masked arm is confined to the
construction graph, so an arm projected onto the endpoint condition alone selects
over designs the others are forbidden to propose, and its regret against an
attainable optimum derived from the construction graph has no floor at zero. A
regret below that floor is a symptom of an arm outside the space, never a result.
The projection therefore reads
[constrains_intermediates][evogfn.env.mutation.MutationEnvironment.constrains_intermediates]
off the environment and enforces the ordering condition exactly where the
environment does, so the arm is confined to the same graph as every other and no
comparison is being drawn across two spaces.

What this still does *not* claim: the distribution spends its mass on infeasible
logit configurations and only ever sees the landscape through the projection, so
what it learns is the composition of the two -- ranks of repaired designs -- and
not the landscape.

Why the arm stays in the suite even where it collapses
-------------------------------------------------------

Two statements, and they must not be merged. **First**, CMA-ES is a standard
baseline for this problem: it appears in Jain et al. (ICML 2022), in
Design-Bench, in FLEXS and in the high-dimensional Bayesian-optimisation
literature, and Design-Bench decodes a continuous relaxation by per-position
argmax exactly as the decoder above does. Its published behaviour is
length-dependent -- competitive on short sequences, degrading as the sequence
grows -- so an unbounded regret on this package's long constrained tasks
corroborates a documented failure mode rather than indicating a broken arm.

**Second**, and separately, nothing in that literature says anything about which
set *this* decoder projects onto. Whether the arm is confined to the same
construction graph as every other is a property of the code above, established by
the tests beside it and by no citation. The two statements are kept apart because
merging them is how a defect in the projection would come to be read as the
published failure mode and go unexamined: a reference explains a bad score, it
never licenses one.

Why the covariance is diagonal
------------------------------

The relaxation has ``d = length * vocabulary`` dimensions, which is 5,120 for the
``L = 256`` protein sequences this package benchmarks on. A full covariance
matrix is then 26 million entries, and CMA-ES needs its eigendecomposition,
costing ``O(d^3) ~ 10^11`` operations per update. That is not a tuning
inconvenience, it is intractable, and it is intractable for the honest reason
that ``d`` is large rather than because of anything about this codebase.

So this is the **separable** variant, sep-CMA-ES (Ros & Hansen, *PPSN* 2008):
the covariance is constrained to be diagonal, its square root is then elementwise
and free, and the learning rates for the rank-one and rank-mu updates are
multiplied by ``(d + 2) / 3`` as that paper prescribes, since a diagonal matrix
has ``d`` rather than ``d(d+1)/2`` parameters to estimate and can afford to move
faster. The cost is real: sep-CMA-ES cannot learn correlations between
coordinates, so it cannot represent epistasis between two positions in its search
distribution. It compensates only in the sense that it reaches a good diagonal
much faster.

Everything else -- the recombination weights, the two evolution paths, the
cumulative step-size adaptation -- follows Hansen's tutorial (arXiv:1604.00772)
with its published default constants, so the baseline is the configuration its
author chose rather than one convenient to us.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines._values import single_objective

#: How a decoded design that the environment cannot build is dealt with.
#:
#: ``"none"`` emits it anyway, which on a constrained instance means emitting
#: nothing usable and is the published method's own behaviour. ``"greedy"``
#: accepts substitutions by descending gain while the design stays legal --
#: the obvious adaptation, and the reported default, so the baseline's score is
#: what a practitioner would get. ``"exact"`` returns the highest-scoring legal
#: design by dynamic programming, which is stronger than anything the literature
#: specifies and is kept for measuring what an engineered decoder is worth
#: rather than for reporting the method.
RepairPolicy = Literal["none", "greedy", "exact"]

_REPAIR_POLICIES: frozenset[str] = frozenset({"none", "greedy", "exact"})

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment

#: Fewest matched measurements that will move the distribution. With one sample
#: the recombination is a copy and the rank-mu update is empty, so the generation
#: carries no information and updating on it only inflates the step size.
_MIN_FOR_UPDATE = 2

#: Bounds on the step size. CMA-ES diverges or collapses on a pathological
#: ranking, and on a landscape returning -inf for infeasible designs pathological
#: rankings happen. Clamping keeps a bad round from producing NaNs that would
#: silently poison every later round of the campaign.
_SIGMA_FLOOR = 1e-12
_SIGMA_CEILING = 1e12

#: Smallest variance a coordinate may hold. At zero it could never move again,
#: and the elementwise whitening below would divide by it.
_VARIANCE_FLOOR = 1e-20

#: Entries the feasibility projection's largest intermediate may hold at once.
#: That array is ``(rows, vocabulary, vocabulary, counter_states)``, so it grows
#: with the batch; rows are processed in blocks small enough to keep it in the
#: tens of megabytes rather than letting a 2,048-design pool decide the peak.
_PROJECTION_BLOCK = 4_000_000


def _projection_counter(length: int, budget: int) -> tuple[int, bool]:
    """Decide what the projection's dynamic program should count, and up to what.

    The mutation budget is a cardinality constraint, so the dynamic program needs
    a counter dimension and its size is what decides whether the projection is
    affordable. Two equivalent framings are available and they cost wildly
    different amounts: "at most ``budget`` substitutions" needs ``budget + 1``
    states, and the identical constraint "at least ``length - budget`` positions
    left at the anchor" needs ``length - budget + 1``. Taking whichever is
    smaller bounds the counter at ``length / 2 + 1`` in the worst case, and the
    cases that actually arise are far better than that: this repository's
    constrained tasks run budgets of 62 of 64 and 248 of 256, which is three and
    nine states respectively rather than 63 and 249.

    Args:
        length: Sequence length.
        budget: Most substitutions from the anchor the environment admits.

    Returns:
        The counter's cap, and whether it counts retained positions. When it
        counts retained positions the counter saturates at the cap and only the
        saturated state is a valid ending, because "at least" is a floor rather
        than a ceiling; when it counts substitutions, exceeding the cap is
        forbidden outright and every state is a valid ending. A cap of zero in
        the retained framing is the unconstrained case: the budget reaches the
        whole sequence and there is nothing to count.
    """
    slack = length - budget
    if budget <= slack:
        return budget, False
    return max(0, slack), True


def _budgeted_argmax(
    logits: npt.NDArray[np.float64],
    parent: Tokens,
    length: int,
    budget: int,
) -> Tokens:
    """Per-position argmax, projected onto the mutation budget alone.

    The budget factorises across positions, so this *is* the exact maximiser of
    the summed logits over the Hamming ball -- there is nothing a dynamic
    program would add. It is the whole decoder when no adjacency rule is set,
    and the unrepaired reference the projection is measured against when one is.

    Args:
        logits: An ``(n, length, vocabulary)`` array of per-position scores.
        parent: The anchor, shape ``(length,)``.
        length: Sequence length.
        budget: Most substitutions from the anchor the environment admits.

    Returns:
        An ``(n, length)`` array within the budget, and feasible only by
        accident.
    """
    chosen = np.asarray(logits.argmax(axis=2), dtype=np.asarray(parent).dtype)
    differing = chosen != parent[None, :]
    counts = differing.sum(axis=1)

    # Confidence that the substitution beats keeping the parent's token.
    # Projecting onto the budget by reverting the *least* confident
    # substitutions keeps the ones the distribution actually asked for; a
    # random projection would discard the search's own signal.
    preference = np.take_along_axis(logits, chosen[:, :, None], axis=2)[:, :, 0]
    margin = preference - logits[:, np.arange(length), parent]

    for row in np.flatnonzero(counts > budget):
        positions = np.flatnonzero(differing[row])
        weakest = positions[np.argsort(-margin[row, positions], kind="stable")[budget:]]
        chosen[row, weakest] = parent[weakest]
    return chosen


def _transition_barrier(
    parent: Tokens,
    permitted: npt.NDArray[np.bool_],
    *,
    ordered: bool,
) -> npt.NDArray[np.float64]:
    """The additive penalty the recursion pays to put one token beside another.

    Two conditions live here, and separating them is the point of the function.
    The first is the environment's adjacency rule, which is a property of the
    pair of tokens alone. The second is *constructibility*: the destination must
    admit an ordering of its substitutions along which every intermediate is
    feasible too, which
    [is_constructible][evogfn.env.mutation.MutationEnvironment.is_constructible]
    shows reduces to one condition per adjacent pair -- and only where both of
    its positions are substituted, since a pair with a position still at the
    anchor passes through no state the destination and the anchor do not already
    vouch for. Where both are substituted, one of them has to go first, and the
    mixed state that order passes through must be permitted.

    A constraint that mentions the anchor's tokens is not a property of the token
    pair, which is why the result carries a position axis where the adjacency
    rule alone would not.

    Args:
        parent: The anchor, shape ``(length,)``.
        permitted: A ``(vocabulary, vocabulary)`` boolean matrix, ``True`` where
            the row's token may be followed by the column's.
        ordered: Whether the environment constrains intermediates, from
            [constrains_intermediates][evogfn.env.mutation.MutationEnvironment.constrains_intermediates].
            When ``False`` the ordering condition is dropped: that environment
            refuses no ordering, so imposing one would narrow the projection to a
            strict subset of what the arm is allowed to propose.

    Returns:
        A ``(length, vocabulary, vocabulary)`` array, entry ``[i, u, t]`` being
        ``0.0`` when token ``t`` at position ``i - 1`` may be followed by token
        ``u`` at position ``i`` and ``-inf`` otherwise. The predecessor is the
        last axis because that is the axis the recursion reduces over, and entry
        ``[0]`` is never read -- the first position has no predecessor.
    """
    anchor = np.asarray(parent)
    length = int(anchor.shape[0])
    vocabulary = int(permitted.shape[0])

    allowed = np.broadcast_to(permitted.T, (length, vocabulary, vocabulary)).copy()
    if ordered and length > 1:
        tokens = np.arange(vocabulary)
        left, right = anchor[:-1], anchor[1:]
        # Whether each candidate token is a substitution at the position it would
        # occupy: the anchor's own token is not, and imposes nothing.
        left_substitutes = tokens[None, :] != left[:, None]
        right_substitutes = tokens[None, :] != right[:, None]
        # The two mixed states, one per order in which the pair could be written.
        left_first = permitted[tokens[None, :], right[:, None]]
        right_first = permitted[left[:, None], tokens[None, :]]
        allowed[1:] &= (
            ~(right_substitutes[:, :, None] & left_substitutes[:, None, :])
            | left_first[:, None, :]
            | right_first[:, :, None]
        )
    return np.where(allowed, 0.0, -np.inf)


#: Candidate substitutions the greedy repair tries per unit of mutation budget.
#: Each costs one batched constructibility check, so this trades decode time
#: against how often the repair gives up below budget on a dense instance.
_GREEDY_ATTEMPTS_PER_SUBSTITUTION = 12


def _greedy_repair(
    logits: npt.NDArray[np.float64],
    parent: Tokens,
    env: MutationEnvironment,
    budget: int,
) -> Tokens:
    """Accept substitutions by descending logit gain, keeping the design legal.

    The straightforward repair, and the one a practitioner reaches for: start at
    the parent -- feasible by definition, and at zero substitutions it constrains
    no ordering -- then take the highest-gain substitution that leaves the design
    constructible, and repeat until the budget is spent or nothing legal is left.

    It is **approximate**, and that is the point of having it. A substitution
    taken early can foreclose a pair that a later, larger gain needed, so the
    result can score below the best legal sequence; the exact projection beside
    it never does. Reporting a baseline through this decoder therefore states
    what the method achieves under an obvious adaptation rather than under an
    engineered one, and the gap between the two decoders is the cost of that
    obviousness rather than a property of the search distribution.

    Legality is re-checked against the environment after each accepted
    substitution rather than argued from the previous state. The condition is
    pairwise, so a substitution can only break the pairs it touches -- but a
    decoder that tracked that itself would be a second implementation of the
    constraint, and the two would drift.

    Args:
        logits: An ``(n, length, vocabulary)`` array of per-position scores.
        parent: The anchor, which every row starts from.
        env: The environment whose graph the result must lie in.
        budget: Substitutions allowed per row.

    Returns:
        An ``(n, length)`` array, every row constructible.
    """
    rows, length, size = logits.shape
    current = np.tile(parent, (rows, 1))
    if budget <= 0 or rows == 0:
        return current

    # Gain of each substitution against staying put. Reverting to the parent's
    # own token is not a substitution, so it is scored at negative infinity
    # rather than at zero, which would let a row spend budget on a change it did
    # not make.
    held = np.broadcast_to(parent[None, :, None], (rows, length, 1))
    gains = logits - np.take_along_axis(logits, held, axis=2)
    gains[:, np.arange(length), parent] = -np.inf
    flat = gains.reshape(rows, -1)

    # Only the strongest candidates are tried. Each attempt costs one batched
    # constructibility check, so scanning the whole (length x vocabulary) list
    # would cost thousands of them per round for substitutions no greedy rule
    # would reach anyway -- the budget is spent long before. A row whose top
    # candidates are all illegal ends with fewer substitutions than the budget
    # allows, which is a real weakness of greedy repair and is left visible
    # rather than papered over by widening the scan.
    attempts = int(min(flat.shape[1], _GREEDY_ATTEMPTS_PER_SUBSTITUTION * budget))
    order = np.argpartition(-flat, attempts - 1, axis=1)[:, :attempts]
    ranked = np.argsort(-np.take_along_axis(flat, order, axis=1), axis=1)
    order = np.take_along_axis(order, ranked, axis=1)

    index = np.arange(rows)
    spent = np.zeros(rows, dtype=np.int64)
    for step in range(attempts):
        live = spent < budget
        if not live.any():
            break
        choice = order[:, step]
        position, token = choice // size, choice % size
        candidate = current.copy()
        offered = live & np.isfinite(flat[index, choice])
        candidate[index[offered], position[offered]] = token[offered]
        legal = env.is_constructible(candidate) & offered
        current[legal] = candidate[legal]
        spent += legal
    return current


def _project_onto_constructible(
    logits: npt.NDArray[np.float64],
    parent: Tokens,
    permitted: npt.NDArray[np.bool_],
    budget: int,
    *,
    ordered: bool = True,
) -> Tokens:
    """Highest-scoring sequence per row that the environment can build.

    This is the constrained counterpart of a per-position argmax. Scoring a
    sequence by the sum of its chosen logits, the unconstrained maximiser is the
    argmax at each position independently; under a first-order adjacency rule the
    maximiser is instead a Viterbi path, and under a mutation budget as well it
    is a Viterbi path over ``(token, counter)`` pairs. Constructibility is the
    third constraint and costs no extra state, because it too reduces to a
    condition on adjacent pairs -- see
    [_transition_barrier][evogfn.algorithms.baselines.cmaes._transition_barrier].
    Nothing is approximated: the returned sequence is the exact maximiser over
    the set the environment can construct, so the projection never discards more
    of the search distribution's signal than the constraint forces it to, and
    never returns a design the environment could not have built.

    **It cannot fail on a well-formed environment.** The anchor is a sequence of
    zero substitutions, so it is inside the budget, it is feasible, and having no
    substitutions at all it constrains no ordering. The constructible set is
    therefore non-empty and the dynamic program always returns something. Rows
    for which it does not -- only reachable by handing this an infeasible anchor
    -- fall back to the anchor itself, which is then equally unbuildable and will
    be reported as such rather than dressed up.

    Args:
        logits: An ``(n, length, vocabulary)`` array of per-position scores.
        parent: The anchor, shape ``(length,)``.
        permitted: A ``(vocabulary, vocabulary)`` boolean matrix, ``True`` where
            the row's token may be followed by the column's.
        budget: Most substitutions from the anchor the environment admits.
        ordered: Whether the environment constrains intermediates and the
            ordering condition therefore applies.

    Returns:
        An ``(n, length)`` array of token indices, every row inside the budget
        and constructible by the environment the arguments describe.
    """
    n, length, vocabulary = logits.shape
    if length == 0:  # pragma: no cover - an empty environment builds nothing
        return np.zeros((n, 0), dtype=np.asarray(parent).dtype)

    cap, retained = _projection_counter(length, budget)
    barrier = _transition_barrier(parent, permitted, ordered=ordered)
    block = max(1, _PROJECTION_BLOCK // max(1, vocabulary * vocabulary * (cap + 1)))
    pieces = [
        _project_block(logits[start : start + block], parent, barrier, cap, retained=retained)
        for start in range(0, n, block)
    ]
    return np.concatenate(pieces) if pieces else np.zeros((0, length), dtype=parent.dtype)


def _project_block(
    logits: npt.NDArray[np.float64],
    parent: Tokens,
    barrier: npt.NDArray[np.float64],
    cap: int,
    *,
    retained: bool,
) -> Tokens:
    """Run the projection's dynamic program over one block of rows.

    Split out from ``_project_onto_constructible`` only so that the
    ``(rows, vocabulary, vocabulary, states)`` intermediate can be bounded; the
    recursion is the whole of the method and lives here.

    Args:
        logits: An ``(m, length, vocabulary)`` block of per-position scores.
        parent: The anchor, shape ``(length,)``.
        barrier: The ``(length, vocabulary, vocabulary)`` penalty from
            [_transition_barrier][evogfn.algorithms.baselines.cmaes._transition_barrier],
            carrying both the adjacency rule and the ordering condition.
        cap: Largest counter value tracked, from
            [_projection_counter][evogfn.algorithms.baselines.cmaes._projection_counter].
        retained: Whether the counter counts positions left at the anchor, in
            which case it saturates at ``cap`` and only the saturated state may
            end a sequence.

    Returns:
        An ``(m, length)`` array of token indices.
    """
    m, length, vocabulary = logits.shape
    states = cap + 1
    counter = np.arange(states, dtype=np.int16)

    back_token = np.zeros((length, m, vocabulary, states), dtype=np.int16)
    back_state = np.zeros((length, m, vocabulary, states), dtype=np.int16)

    # Exactly one token per position is treated differently from the rest -- the
    # anchor's own -- because it is the only one whose choice is not a
    # substitution. So each step is a whole-array default plus a single-column
    # correction, rather than two masked halves.
    score = np.full((m, vocabulary, states), -np.inf)
    opening = int(parent[0])
    if retained:
        score[:, :, 0] = logits[:, 0, :]
        score[:, opening, 0] = -np.inf
        score[:, opening, min(1, cap)] = logits[:, 0, opening]
    else:
        if cap:
            score[:, :, 1] = logits[:, 0, :]
            score[:, opening, 1] = -np.inf
        score[:, opening, 0] = logits[:, 0, opening]

    for position in range(1, length):
        # Best predecessor for every (successor token, counter) pair. Both the
        # adjacency rule and the ordering condition enter the recursion here and
        # nowhere else, which is why the barrier is indexed by position.
        candidates = np.moveaxis(score, 1, -1)[:, None, :, :] + barrier[position][None, :, None, :]
        best, token, held = _advance(
            candidates.max(axis=-1),
            candidates.argmax(axis=-1).astype(np.int16),
            anchor=int(parent[position]),
            counter=counter,
            retained=retained,
        )
        score = best + logits[:, position, :, None]
        back_token[position] = token
        back_state[position] = held

    flat = score.reshape(m, vocabulary * states).copy()
    if retained and cap > 0:
        # Only the saturated counter satisfies "at least `cap` retained".
        unfinished = np.ones(states, dtype=np.bool_)
        unfinished[cap] = False
        flat[:, np.tile(unfinished, vocabulary)] = -np.inf
    return _backtrack(flat, back_token, back_state, parent, states)


def _advance(
    arrived: npt.NDArray[np.float64],
    source: npt.NDArray[np.int16],
    *,
    anchor: int,
    counter: npt.NDArray[np.int16],
    retained: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    """Apply one position's counter bookkeeping to the incoming scores.

    Exactly one token per position is treated differently from every other --
    the anchor's own, because it is the only choice that is not a substitution.
    So a step is a whole-array default plus a one-column correction rather than
    two masked halves, which is what keeps the inner loop free of fancy
    indexing over the vocabulary.

    Args:
        arrived: ``(m, vocabulary, states)`` best score reaching each
            ``(token, counter)`` pair from a permitted predecessor.
        source: The predecessor token behind each of those, same shape.
        anchor: The anchor's token at this position.
        counter: ``[0, 1, ..., cap]``, held once rather than rebuilt per step.
        retained: Whether the counter counts positions left at the anchor.

    Returns:
        The score, predecessor-token and predecessor-counter tables for this
        position, each ``(m, vocabulary, states)``.
    """
    m, vocabulary, states = arrived.shape
    cap = states - 1

    if retained:
        # Default: a substitution, which does not advance a counter of retained
        # positions. The anchor's own token does, and saturates -- once `cap`
        # positions are held the rest are free, and without that the counter
        # would forbid exactly the sequences a loose budget exists to allow.
        best = arrived.copy()
        token = source.copy()
        held = np.broadcast_to(counter, arrived.shape).copy()
        column_best = np.full((m, states), -np.inf)
        column_token = np.zeros((m, states), dtype=np.int16)
        column_held = np.zeros((m, states), dtype=np.int16)
        if states > 1:
            column_best[:, 1:] = arrived[:, anchor, :-1]
            column_token[:, 1:] = source[:, anchor, :-1]
            column_held[:, 1:] = counter[:-1]
        stayed = arrived[:, anchor, cap]
        improves = stayed > column_best[:, cap]
        column_best[improves, cap] = stayed[improves]
        column_token[improves, cap] = source[:, anchor, cap][improves]
        column_held[improves, cap] = cap
    else:
        # Default: a substitution, which does advance a counter of them, and
        # may not push it past the cap. The anchor's own token is free.
        best = np.full((m, vocabulary, states), -np.inf)
        token = np.zeros((m, vocabulary, states), dtype=np.int16)
        held = np.zeros((m, vocabulary, states), dtype=np.int16)
        if states > 1:
            best[:, :, 1:] = arrived[:, :, :-1]
            token[:, :, 1:] = source[:, :, :-1]
            held[:, :, 1:] = counter[None, None, :-1]
        column_best = arrived[:, anchor, :].copy()
        column_token = source[:, anchor, :].copy()
        column_held = np.broadcast_to(counter, (m, states)).copy()

    best[:, anchor, :] = column_best
    token[:, anchor, :] = column_token
    held[:, anchor, :] = column_held
    return best, token, held


def _backtrack(
    terminal: npt.NDArray[np.float64],
    back_token: npt.NDArray[np.int16],
    back_state: npt.NDArray[np.int16],
    parent: Tokens,
    states: int,
) -> Tokens:
    """Walk the recorded predecessors back into sequences.

    Args:
        terminal: ``(m, vocabulary * states)`` final scores, with the states that
            do not satisfy the counter already set to ``-inf``.
        back_token: ``(length, m, vocabulary, states)`` predecessor tokens.
        back_state: The same shape, holding predecessor counter values.
        parent: The anchor, used for rows with no admissible path at all.
        states: Counter states, needed to unflatten ``terminal``.

    Returns:
        An ``(m, length)`` array of token indices.
    """
    m = terminal.shape[0]
    length = back_token.shape[0]
    rows = np.arange(m)

    choice = terminal.argmax(axis=1)
    found = np.isfinite(terminal[rows, choice])
    token_index = (choice // states).astype(np.intp)
    state_index = (choice % states).astype(np.intp)

    decoded = np.empty((m, length), dtype=np.asarray(parent).dtype)
    decoded[:, length - 1] = token_index
    for position in range(length - 1, 0, -1):
        previous_token = back_token[position][rows, token_index, state_index].astype(np.intp)
        state_index = back_state[position][rows, token_index, state_index].astype(np.intp)
        token_index = previous_token
        decoded[:, position - 1] = token_index
    # Only reachable by handing this an infeasible anchor, at which point the
    # environment can build nothing and the anchor is as good an answer as any.
    decoded[~found] = parent
    return decoded


def _permitted_adjacencies(env: MutationEnvironment) -> npt.NDArray[np.bool_] | None:
    """The environment's adjacency rule as a boolean matrix, if it has one.

    Read off the environment's own transition matrix. That attribute is private
    and there is no public accessor, but the alternative is worse: the rule
    cannot be recovered from the public surface, because
    [forward_mask][evogfn.env.mutation.MutationEnvironment.forward_mask] only
    ever reports the rows and columns the anchor's own tokens happen to touch,
    and a projection built on a partially-recovered rule would emit designs the
    environment rejects while believing it had checked them.

    Args:
        env: The environment being sampled.

    Returns:
        A ``(vocabulary, vocabulary)`` boolean matrix, ``True`` where an
        adjacency is allowed, or ``None`` when the environment constrains
        nothing and the cheap per-position argmax is exact.
    """
    matrix: npt.NDArray[np.floating] | None = env._transitions
    return None if matrix is None else np.asarray(matrix) > 0


class CMAES(Sampler):
    """Separable CMA-ES over a continuous relaxation of the sequence space.

    Args:
        env: Supplies the parent, alphabet, mutation budget and feasibility.
        initial_sigma: Starting step size. Hansen advises roughly a third of the
            search domain's width, but logits have no natural width, so 1.0 is
            our choice: it makes the initial argmax uniform over the alphabet,
            and the step-size adaptation corrects the scale within a few
            generations anyway.
        repair: Decode through the constrained projection rather than a plain
            per-position argmax, so that every emitted design is one the
            environment can build: inside the mutation budget, obeying the
            adjacency rule, and reachable by some ordering of its substitutions.
            On by default, and the reason this baseline reports a finite score at
            all on a sparse feasible set; see the module docstring. Passing
            ``False`` restores the unrepaired relaxation, which is the control
            arm for "what a method that ignores the constraint does" and is
            expected to score minus infinity everywhere the constraint bites.
            Ignored where the environment constrains no adjacencies, since the
            plain argmax is then already exact.
        feasible_only: Resample until every proposal is constructible. Left in
            place as the check on ``repair`` rather than as a remedy in its own
            right: rejection at the feasible densities this package benchmarks
            at is not expensive but impossible, and with ``repair`` on nothing
            is ever rejected. Constructibility is the test, not endpoint
            feasibility -- rejecting on the weaker condition would let this
            path keep the designs ``repair`` exists to exclude.
        max_attempts: Resampling rounds before giving up when ``feasible_only``.
        seed: Seeds the Gaussian.

    Raises:
        ValueError: If ``initial_sigma`` is not positive.
    """

    def __init__(  # noqa: PLR0913 - a decode policy sits beside the search parameters
        self,
        env: MutationEnvironment,
        *,
        initial_sigma: float = 1.0,
        repair: RepairPolicy = "greedy",
        feasible_only: bool = False,
        max_attempts: int = 50,
        seed: int = 0,
    ) -> None:
        """Start the distribution uniform over the alphabet at every position."""
        super().__init__()
        if initial_sigma <= 0.0:
            raise ValueError(f"initial_sigma must be positive, got {initial_sigma}")

        if repair not in _REPAIR_POLICIES:
            raise ValueError(
                f"repair must be one of {sorted(_REPAIR_POLICIES)}, got {repair!r}; "
                f"the decoder decides how much of this arm's result is its own"
            )
        self._env = env
        self._repair = repair
        self._permitted = _permitted_adjacencies(env) if repair == "exact" else None
        # Which of the two search spaces the projection targets. Read from the
        # environment rather than assumed, because the two differ by exactly the
        # designs this arm used to be able to reach and the others could not.
        self._ordered = env.constrains_intermediates
        self._feasible_only = feasible_only
        self._max_attempts = max_attempts
        self._rng = np.random.default_rng(seed)
        # How often the raw relaxation emitted something the environment could
        # not build. Reported rather than swallowed: it is the measurement that
        # says whether the projection is a formality or is doing all the work.
        self._decoded = 0
        self._unbuildable = 0

        self._dimension = env.sequence_length * env.alphabet.size
        # A zero mean is the uniform categorical at every position, which is the
        # least committed start available and needs no arbitrary bias toward the
        # parent -- the mutation budget already anchors every sample to it.
        self._mean = np.zeros(self._dimension, dtype=np.float64)
        self._diagonal = np.ones(self._dimension, dtype=np.float64)
        self._sigma = float(initial_sigma)
        self._path_sigma = np.zeros(self._dimension, dtype=np.float64)
        self._path_c = np.zeros(self._dimension, dtype=np.float64)
        self._generation = 0
        # E||N(0, I)||, Hansen's series approximation. Used as the reference the
        # step-size controller compares the evolution path's length against.
        self._expected_norm = math.sqrt(self._dimension) * (
            1.0 - 1.0 / (4.0 * self._dimension) + 1.0 / (21.0 * self._dimension**2)
        )

        # The last batch handed out, so `observe` can find the Gaussian draw a
        # returned sequence came from. Keyed by sequence rather than by row
        # index because the harness scores a *selected subset* of the proposals,
        # in its own order; assuming alignment would attribute each score to
        # some other candidate's draw and update the distribution with noise.
        self._samples = np.zeros((0, self._dimension), dtype=np.float64)
        self._index: dict[bytes, int] = {}

    @property
    def name(self) -> str:
        """Short label, marking any deviation from the default configuration."""
        label = "CMAES"
        if self._repair != "greedy":
            label += f" ({self._repair} repair)" if self._repair != "none" else " (unrepaired)"
        if self._feasible_only:
            label += " (feasible)"
        return label

    @property
    def sigma(self) -> float:
        """Current step size."""
        return self._sigma

    @property
    def repaired_fraction(self) -> float:
        """Share of decoded designs the raw per-position argmax got wrong.

        Measured as the fraction of emissions whose unprojected argmax the
        environment could not construct -- inside the budget, feasible where it
        stands, *and* reachable by some ordering of its substitutions -- so it is
        a statement about the *relaxation* rather than about the projection: at
        0.0 the Gaussian is finding the constructible set on its own and the
        projection is a formality, at 1.0 the Gaussian has never once produced a
        buildable design and every design credited to this arm is the
        projection's work. Reporting a CMA-ES result on a constrained landscape
        without this number beside it overstates the method.

        Returns:
            A fraction in ``[0, 1]``, and ``0.0`` before anything is decoded or
            when ``repair`` is off and nothing is measured.
        """
        return self._unbuildable / self._decoded if self._decoded else 0.0

    @property
    def mean_logits(self) -> npt.NDArray[np.float64]:
        """The distribution's mean, shaped ``(sequence_length, vocabulary)``."""
        reshaped: npt.NDArray[np.float64] = self._mean.reshape(
            self._env.sequence_length, self._env.alphabet.size
        ).copy()
        return reshaped

    def reanchored(self, env: MutationEnvironment) -> CMAES:
        """Carry the search distribution to a re-anchored environment.

        **More survives here than the obvious reading suggests, and it is worth
        being precise about which part.** The relaxation is indexed by
        ``(position, token)``. Neither index is expressed relative to the
        anchor, so the mean's content -- "position 7 prefers token 3" -- is a
        statement about the landscape that the move does not touch, and the same
        is true of the diagonal covariance, the two evolution paths and the step
        size. What is anchor-relative is only the **decoder**: which choices
        count against the mutation budget, and hence which projection the same
        mean now passes through. So the distribution is carried, and the caveat
        is that its calibration was earned under a projection that has changed
        -- a tight budget makes the same mean decode to a different design, a
        loose one barely moves it.

        **What is dropped, and why it cannot be kept.** The pairing between the
        last batch's Gaussian draws and the sequences they decoded to. Those
        sequences came out of the old decoder; under the new one the same draw
        decodes elsewhere, so a score arriving after the move would be
        attributed to a draw that did not produce the design it was measured on.
        The rank-mu update would then be fed noise while continuing to look like
        it was working, which is the failure this class already has a test for.
        Dropping the pairing costs at most one generation of adaptation --
        [Campaign][evogfn.loop.campaign.Campaign] calls ``observe`` before it
        moves the anchor, so in the campaign loop it costs nothing.

        The repair counters are carried too: they are a running measurement of
        how badly the relaxation fits this landscape, and resetting them at each
        anchor would report the last round instead of the campaign.

        Args:
            env: The re-anchored environment. Must describe the same relaxation
                -- same sequence length and same alphabet -- since the mean is a
                vector in it.

        Returns:
            A new sampler over ``env``, carrying the distribution. This one is
            left untouched.

        Raises:
            ValueError: If ``env`` changes the sequence length or the alphabet
                size, which would make the carried mean a vector of the wrong
                dimension. Refused rather than reshaped: a silently truncated
                mean is a search distribution nobody chose.
        """
        dimension = env.sequence_length * env.alphabet.size
        if dimension != self._dimension:
            raise ValueError(
                f"cannot carry a {self._dimension}-dimensional relaxation into an "
                f"environment of dimension {dimension}; the anchor may move but the "
                f"sequence length and alphabet may not"
            )

        moved = CMAES(
            env,
            initial_sigma=self._sigma,
            repair=self._repair,
            feasible_only=self._feasible_only,
            max_attempts=self._max_attempts,
        )
        moved._mean = self._mean.copy()
        moved._diagonal = self._diagonal.copy()
        moved._path_sigma = self._path_sigma.copy()
        moved._path_c = self._path_c.copy()
        moved._generation = self._generation
        # The generator itself, not a fresh one at the same seed: a campaign is
        # reproducible because one stream runs through it, and restarting the
        # stream at every anchor would make round three's draws a copy of round
        # one's.
        moved._rng = self._rng
        moved._proposals_made = self._proposals_made
        moved._decoded = self._decoded
        moved._unbuildable = self._unbuildable
        return moved

    def propose(self, n: int) -> Tokens:
        """Draw ``n`` sequences from the current search distribution.

        With ``repair`` on -- the default -- the decoder already guarantees
        every design is constructible, so the rejection loop below accepts the
        first batch and the two paths cost the same ``n`` proposals. That is the
        point rather than dead code: rejection is retained as the *check* on the
        projection, and it is the path an unrepaired run takes, where it fails
        loudly instead of returning designs the environment cannot build.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array.

        Raises:
            RuntimeError: If ``feasible_only`` and the attempt budget is spent
                before enough feasible candidates are found. Reachable only
                without ``repair``, or from an environment whose own anchor is
                infeasible -- in which case nothing is constructible and saying
                so is the correct outcome.
        """
        if not self._feasible_only:
            sequences, draws = self._sample(n)
            self._count(n)
            self._remember(sequences, draws)
            return sequences

        kept_sequences: list[Tokens] = []
        kept_draws: list[npt.NDArray[np.float64]] = []
        found = 0
        for _ in range(self._max_attempts):
            sequences, draws = self._sample(n)
            self._count(n)
            constructible = self._env.is_constructible(sequences)
            if constructible.any():
                kept_sequences.append(sequences[constructible])
                kept_draws.append(draws[constructible])
                found += int(constructible.sum())
            if found >= n:
                chosen = np.concatenate(kept_sequences)[:n]
                self._remember(chosen, np.concatenate(kept_draws)[:n])
                return chosen
        raise RuntimeError(
            f"could not draw {n} feasible candidates in {self._max_attempts} attempts "
            f"({found} found); rejection sampling has become impractical at this "
            f"feasible density, which is itself the result"
        )

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Move the mean, the evolution paths, the covariance and the step size.

        Only candidates traceable to the most recent
        [propose][evogfn.algorithms.baselines.cmaes.CMAES.propose] contribute:
        CMA-ES updates from the *Gaussian draws* behind the ranking, not from
        the sequences, and a sequence from anywhere else has no draw behind it.
        A batch with fewer than two such candidates leaves the distribution
        untouched rather than updating it on a ranking of one.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which gives
                the rank-based update no single ordering to work from.
        """
        rows, scores = self._matched(sequences, values)
        if rows.size < _MIN_FOR_UPDATE:
            return

        order = np.argsort(-scores, kind="stable")
        selected = self._samples[rows[order]]
        weights, mu_eff = self._recombination_weights(rows.size)
        selected = selected[: weights.size]

        d = self._dimension
        c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
        damping = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)
        c_1 = 2.0 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1.0 - c_1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((d + 2.0) ** 2 + mu_eff),
        )
        # Ros & Hansen's separable speed-up: a diagonal covariance has d free
        # parameters rather than d(d+1)/2, so it may be learned this much faster.
        c_1, c_mu = self._separable_rates(c_1, c_mu)

        self._generation += 1
        step = weights @ selected  # the weighted recombination, in y-space
        self._mean = self._mean + self._sigma * step

        # With a diagonal covariance the whitening C^(-1/2) is elementwise, which
        # is the entire computational reason for the separable variant.
        whitened = step / np.sqrt(self._diagonal)
        self._path_sigma = (1.0 - c_sigma) * self._path_sigma + math.sqrt(
            c_sigma * (2.0 - c_sigma) * mu_eff
        ) * whitened

        path_norm = float(np.linalg.norm(self._path_sigma))
        correction = math.sqrt(max(1e-16, 1.0 - (1.0 - c_sigma) ** (2 * self._generation)))
        # Hansen's h_sigma: stall the rank-one update when the path has grown
        # unusually long, which is the signature of a step size still adapting
        # rather than of a genuine direction worth recording.
        h_sigma = path_norm / correction < (1.4 + 2.0 / (d + 1.0)) * self._expected_norm

        gain = math.sqrt(c_c * (2.0 - c_c) * mu_eff) if h_sigma else 0.0
        self._path_c = (1.0 - c_c) * self._path_c + gain * step
        leak = 0.0 if h_sigma else c_c * (2.0 - c_c)
        rank_mu = weights @ (selected**2)
        self._diagonal = (
            (1.0 - c_1 - c_mu + c_1 * leak) * self._diagonal
            + c_1 * self._path_c**2
            + c_mu * rank_mu
        )
        self._diagonal = np.maximum(self._diagonal, _VARIANCE_FLOOR)

        self._sigma = float(
            np.clip(
                self._sigma
                * math.exp((c_sigma / damping) * (path_norm / self._expected_norm - 1.0)),
                _SIGMA_FLOOR,
                _SIGMA_CEILING,
            )
        )

    def _sample(self, n: int) -> tuple[Tokens, npt.NDArray[np.float64]]:
        """Draw ``n`` points and decode them.

        Returns:
            The decoded sequences and the ``(n, dimension)`` array of draws in
            ``y``-space -- that is, ``(x - mean) / sigma``, which is the form
            every CMA-ES update equation is written in.
        """
        z = self._rng.standard_normal((n, self._dimension))
        y = z * np.sqrt(self._diagonal)[None, :]
        x = self._mean[None, :] + self._sigma * y
        return self._decode(x), y

    def _decode(self, x: npt.NDArray[np.float64]) -> Tokens:
        """Read sequences off the relaxation, projected onto what can be built.

        Two projections, and which are needed depends on the environment. The
        mutation budget always applies and factorises across positions, so the
        cheap independent argmax followed by reverting the least-confident
        surplus substitutions is already its exact maximiser. Neither the
        adjacency rule nor the ordering condition factorises, and where the
        environment sets one the whole decode is handed to
        ``_project_onto_constructible`` instead, which solves every constraint
        jointly and exactly.

        Which rows go to the dynamic program is decided by
        [is_constructible][evogfn.env.mutation.MutationEnvironment.is_constructible]
        and not by
        [is_reachable][evogfn.env.mutation.MutationEnvironment.is_reachable]. The
        weaker test would wave through exactly the designs that are feasible
        where they stand and that no trajectory can build, so the projection
        would never see the rows it exists to repair.

        The unprojected argmax is decoded either way, and counted where it was
        not constructible. That costs one extra ``argmax`` and buys
        [repaired_fraction][evogfn.algorithms.baselines.cmaes.CMAES.repaired_fraction]:
        without it a table would report this arm's score with no way of telling
        whether the search distribution or the projection produced it.

        Args:
            x: An ``(n, dimension)`` array of points in the relaxation.

        Returns:
            An ``(n, sequence_length)`` array inside the environment's graph.
        """
        parent = self._env.parent
        length = self._env.sequence_length
        size = self._env.alphabet.size
        logits = x.reshape(x.shape[0], length, size)

        budget = self._env.max_mutations
        decoded = _budgeted_argmax(logits, parent, length, budget)
        self._decoded += int(decoded.shape[0])
        if self._repair == "none":
            return decoded

        # Counted before either repair runs, and against the same test both
        # repairs have to satisfy, so the share is comparable between the two
        # decoders and says the same thing about the relaxation under each.
        unbuildable = ~self._env.is_constructible(decoded)
        self._unbuildable += int(unbuildable.sum())
        if not unbuildable.any():
            return decoded
        if self._repair == "greedy":
            decoded[unbuildable] = _greedy_repair(logits[unbuildable], parent, self._env, budget)
            return decoded

        # A row whose unconstrained argmax is already constructible needs no
        # projection, and would not be changed by one: it maximises the summed
        # logits over the whole ball, so it maximises them over the feasible
        # part of the ball too. Only the rest are handed to the dynamic program,
        # which is what keeps a loosely constrained landscape cheap.
        assert self._permitted is not None  # noqa: S101 - built for this policy
        decoded[unbuildable] = _project_onto_constructible(
            logits[unbuildable], parent, self._permitted, budget, ordered=self._ordered
        )
        return decoded

    def _remember(self, sequences: Tokens, draws: npt.NDArray[np.float64]) -> None:
        """Record which draw produced which sequence, for the next ``observe``."""
        self._samples = draws
        contiguous = np.ascontiguousarray(sequences)
        # Duplicates collapse onto one draw. Two draws decoding to the same
        # sequence are indistinguishable to the oracle, so attributing the score
        # to either is equally defensible and neither biases the ranking.
        self._index = {row.tobytes(): position for position, row in enumerate(contiguous)}

    def _matched(
        self, sequences: Tokens, values: Fitness
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
        """Line up scored sequences with the draws they came from.

        Returns:
            The sample indices and their finite objective values.

        Raises:
            ValueError: If the values carry more than one objective, which gives
                the rank-based update no single ordering to work from.
        """
        flat = single_objective(values)
        contiguous = np.ascontiguousarray(np.asarray(sequences))
        rows: list[int] = []
        scores: list[float] = []
        for row, value in zip(contiguous, flat, strict=False):
            position = self._index.get(row.tobytes())
            if position is None or not np.isfinite(value):
                continue
            rows.append(position)
            scores.append(float(value))
        return np.asarray(rows, dtype=np.intp), np.asarray(scores, dtype=np.float64)

    def _separable_rates(self, c_1: float, c_mu: float) -> tuple[float, float]:
        """Scale the covariance learning rates for a diagonal matrix.

        Returns:
            The scaled ``(c_1, c_mu)``, jointly capped at 1 so the covariance
            update stays a convex combination.
        """
        factor = (self._dimension + 2.0) / 3.0
        c_1, c_mu = c_1 * factor, c_mu * factor
        total = c_1 + c_mu
        if total > 1.0:
            c_1, c_mu = c_1 / total, c_mu / total
        return c_1, c_mu

    @staticmethod
    def _recombination_weights(n: int) -> tuple[npt.NDArray[np.float64], float]:
        """Hansen's default log-decreasing weights over the better half.

        Args:
            n: Candidates whose scores came back, which plays the role of the
                population size ``lambda``. It is read from the batch rather
                than fixed in the constructor because the harness decides how
                many designs a round screens.

        Returns:
            The normalised weights and the variance-effective selection mass
            ``mu_eff = 1 / sum(w^2)``.
        """
        mu = max(1, n // 2)
        raw = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
        weights = raw / raw.sum()
        return weights, float(1.0 / np.sum(weights**2))
