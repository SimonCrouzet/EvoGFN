r"""Several Ehrlich functions at once, with objective conflict as a dial.

[CH65Landscape][evogfn.landscapes.ch65.CH65Landscape] is the only real
multi-objective landscape here, and a dataset has exactly one degree of objective
conflict and does not say what it is. This landscape supplies the dial, the way
[EhrlichLandscape][evogfn.landscapes.ehrlich.EhrlichLandscape] does for the
single-objective anchors.

## The mechanism: how far the objectives' optima are allowed to disagree

Every objective is its own Ehrlich function, and an Ehrlich function is defined by
the motifs carved out of a **planted optimum** -- a feasible sequence satisfying
all of its motifs simultaneously, by construction. So "how much do two Ehrlich
objectives conflict" is "how much do their planted optima disagree, and can one
sequence satisfy both sets of motifs at once".

`conflict` controls exactly that. All objectives are drawn from one shared
feasible base sequence $b$, and objective $j$'s planted optimum agrees with $b$
over a **prefix of length $\lfloor (1 - \text{conflict}) \cdot L \rceil$** and
then continues as an independent walk of the same Markov chain:

$$
x^*_j = \underbrace{b_{1..m}}_{\text{shared}} \;\Vert\;
\underbrace{w_j}_{\text{drawn per objective}},
\qquad m = \operatorname{round}\big((1 - \text{conflict}) \cdot L\big)
$$

Motifs are carved from $x^*_j$ one per block, as in the single-objective case:

* **`conflict = 0`** -- $m = L$, every $x^*_j$ *is* $b$, so every objective's
  motifs are read off the same sequence and $b$ scores 1.0 on all of them. The
  exact Pareto front is a **single point**, exactly rather than approximately.
* **`conflict = 1`** -- $m = 0$, the planted optima are independent walks, and
  the objectives' motifs are unrelated patterns competing for the same short
  sequence. Satisfying one costs the other and the front **spreads**.
* **in between** -- motifs falling in the shared prefix carry identical
  requirements across objectives and pull together; motifs in the divergent
  suffix pull apart.

Note the dial runs opposite to the naive intuition that "motifs at shared
positions must conflict": shared positions carrying the *same* requirement are
the aligned case, and it is independent draws that fight.

## Feasibility must be one thing, not one thing per objective

The constituent landscapes are required to share a single transition matrix, and
the constructor **checks it rather than assuming it**. Without that `is_feasible`
would have to pick an objective to believe, a sequence could be constructible for
objective 1 and not for objective 2, and the environment -- which masks against
exactly one transition matrix -- would generate designs that one objective
refuses to score. Every downstream number, the hypervolume above all, would be
computed over a set whose membership nobody agrees on.

Sharing is arranged by construction: the constituents are built with the same
seed, alphabet size and transition density, and
[EhrlichLandscape][evogfn.landscapes.ehrlich.EhrlichLandscape] draws its
transition matrix before anything else, so the matrices are identical bit for
bit. The check is still run, because "arranged by construction" is a claim about
code that can change.

## The exact Pareto front

On a small enough instance the whole space can be enumerated, so the true Pareto
front is *known* rather than approximated by whatever the compared methods
happened to find -- hypervolume against a union-of-methods reference front is a
ranking that moves when a method is added.
[exact_pareto_front][evogfn.landscapes.multi_ehrlich.MultiEhrlichLandscape.exact_pareto_front]
provides it, guarded on
[MAX_ENUMERABLE_SIZE][evogfn.landscapes.base.MAX_ENUMERABLE_SIZE] as the other
landscapes guard enumeration.

## Precedent

arXiv:2510.21052, Steinberg et al., *Amortized Active Generation of Pareto Sets*
(A-GPS), uses Ehrlich in a multi-objective benchmark but not in this form: its
"Ehrlich vs. naturalness" task pairs a *single* Ehrlich function with a ProtBert
pseudo-likelihood, $f_1(x) = \text{Ehrlich}(x)$ and
$f_2(x) = e^{-L_{\text{ProtBert}}(x)}$, at $L \in \{15, 32, 64\}$ over the
20-letter alphabet. It does not compose several Ehrlich functions and has no
conflict parameter. Their configurations: $L = 15$ with $k = 3, c = 2, q = 3$;
$L = 32$ with $k = 4, c = 3, q = 4$; $L = 64$ with $k = 4, c = 4, q = 4$.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.landscapes.base import FitnessLandscape
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.metrics.pareto import non_dominated

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from evogfn.core.types import Alphabet, Fitness, Tokens

#: Fewest objectives that make a multi-objective landscape. One objective is not
#: a degenerate case worth supporting here -- it is
#: [EhrlichLandscape][evogfn.landscapes.ehrlich.EhrlichLandscape], which already
#: exists and says so in its type.
MIN_OBJECTIVES = 2


class MultiEhrlichLandscape(FitnessLandscape):
    """Several Ehrlich functions over one sequence, one alphabet, one feasible set.

    Constructed from constituent
    [EhrlichLandscape][evogfn.landscapes.ehrlich.EhrlichLandscape] instances,
    which must agree on their alphabet, sequence length and -- above all --
    transition matrix. Use
    [with_conflict][evogfn.landscapes.multi_ehrlich.MultiEhrlichLandscape.with_conflict]
    to build a family whose objective conflict is a parameter; use this
    constructor directly to compose instances you built yourself, and have the
    agreement checked.

    Args:
        landscapes: Two or more Ehrlich landscapes, in objective order.
        conflict: The dial value these landscapes were built with, recorded for
            reporting. `None` when they came from somewhere else, which is the
            honest answer -- a benchmark record that cannot state its parameters
            cannot be reproduced from a results table.

    Raises:
        ValueError: If fewer than `MIN_OBJECTIVES` landscapes are given, or if
            they disagree on alphabet, sequence length or transition matrix.
    """

    def __init__(
        self,
        landscapes: Sequence[EhrlichLandscape],
        *,
        conflict: float | None = None,
    ) -> None:
        """Compose the landscapes after checking they describe the same space."""
        if len(landscapes) < MIN_OBJECTIVES:
            raise ValueError(
                f"a multi-objective landscape needs at least {MIN_OBJECTIVES} objectives, "
                f"got {len(landscapes)}; for one, use EhrlichLandscape directly"
            )
        _require_one_space(landscapes)
        self._landscapes = tuple(landscapes)
        self._conflict = conflict
        self._front_cache: tuple[Tokens, npt.NDArray[np.float64]] | None = None

    @classmethod
    def with_conflict(  # noqa: PLR0913 - a benchmark's parameters are its definition
        cls,
        *,
        sequence_length: int = 32,
        vocab_size: int = 20,
        n_objectives: int = MIN_OBJECTIVES,
        n_motifs: int = 2,
        motif_length: int = 4,
        quantization: int | None = None,
        max_spacing: int = 3,
        transition_density: float = 0.5,
        conflict: float = 1.0,
        seed: int = 0,
    ) -> MultiEhrlichLandscape:
        r"""Build a family of Ehrlich objectives at a chosen degree of conflict.

        The Ehrlich parameters are shared by every objective, so a sweep over
        `conflict` changes the trade-off and nothing else. See the module
        docstring for the mechanism; in short, objective $j$'s planted optimum
        agrees with a shared base sequence over a prefix of
        $\operatorname{round}((1 - \text{conflict}) \cdot L)$ positions and is
        drawn independently after it.

        Args:
            sequence_length: Length $L$ of every sequence.
            vocab_size: Alphabet size $v$, shared by every objective.
            n_objectives: How many Ehrlich functions to compose. At least
                `MIN_OBJECTIVES`.
            n_motifs: Motifs $c$ per objective.
            motif_length: Tokens $k$ per motif.
            quantization: Reward levels $q$ per motif. Must divide $k$.
            max_spacing: Largest gap between consecutive motif positions.
            transition_density: Fraction of token pairs allowed to be adjacent.
                Shared, since the feasible set is shared.
            conflict: Degree of objective conflict in `[0, 1]`. `0` makes the
                objectives jointly maximisable and collapses the exact Pareto
                front to a single point; `1` draws their optima independently
                and spreads it.
            seed: Seeds the transition matrix, the base sequence and every
                objective's divergence.

        Returns:
            The composed landscape, its constituents sharing one transition
            matrix.

        Raises:
            ValueError: If `conflict` is outside `[0, 1]`, if fewer than
                `MIN_OBJECTIVES` objectives are asked for, or if the Ehrlich
                parameters cannot describe a valid instance.
        """
        if not 0.0 <= conflict <= 1.0:
            raise ValueError(f"conflict must lie in [0, 1], got {conflict}")
        if n_objectives < MIN_OBJECTIVES:
            raise ValueError(
                f"a multi-objective landscape needs at least {MIN_OBJECTIVES} objectives, "
                f"got {n_objectives}; for one, use EhrlichLandscape directly"
            )

        shared = {
            "sequence_length": sequence_length,
            "vocab_size": vocab_size,
            "n_motifs": n_motifs,
            "motif_length": motif_length,
            "quantization": quantization,
            "max_spacing": max_spacing,
            "transition_density": transition_density,
            # The same seed across objectives is what makes the transition
            # matrices identical: EhrlichLandscape draws that matrix first, from
            # a stream determined by (seed, vocab_size, transition_density)
            # alone. Everything after it is redirected per objective below.
            "seed": seed,
        }

        reference = EhrlichLandscape(**shared)  # type: ignore[arg-type]
        base = reference.optimal_sequence
        agreed = round((1.0 - conflict) * sequence_length)

        objectives: list[EhrlichLandscape] = [reference]
        objectives += [
            _DivergentEhrlich(
                agreed_prefix=base[:agreed],
                divergence_seed=(seed, index),
                **shared,
            )
            for index in range(1, n_objectives)
        ]
        return cls(objectives, conflict=conflict)

    @property
    def alphabet(self) -> Alphabet:
        """The alphabet every objective is written in."""
        return self._landscapes[0].alphabet

    @property
    def sequence_length(self) -> int:
        """Length of every sequence this landscape scores."""
        return self._landscapes[0].sequence_length

    @property
    def n_objectives(self) -> int:
        """How many Ehrlich functions are composed."""
        return len(self._landscapes)

    @property
    def objective_names(self) -> tuple[str, ...]:
        """One name per constituent Ehrlich function."""
        return tuple(f"ehrlich_{index}" for index in range(self.n_objectives))

    @property
    def optimum(self) -> Fitness:
        """The **ideal point**: 1.0 on every objective.

        Returns:
            An `(n_objectives,)` array of ones.

        Note:
            Each objective attains 1.0 at its own planted optimum, so the ideal
            point is componentwise exact. Whether any *single* sequence attains
            it is what `conflict` decides -- at `conflict = 0` the shared base
            sequence does, and above that generally nothing does. Like
            [CH65Landscape.optimum][evogfn.landscapes.ch65.CH65Landscape.optimum]
            this is therefore a normaliser and a
            [Tchebycheff][evogfn.rewards.scalarization.Tchebycheff] reference,
            not a target to regret against. For "how close did this run get",
            measure against
            [exact_pareto_front][evogfn.landscapes.multi_ehrlich.MultiEhrlichLandscape.exact_pareto_front].
        """
        return np.ones(self.n_objectives, dtype=np.float64)

    @property
    def landscapes(self) -> tuple[EhrlichLandscape, ...]:
        """The constituent Ehrlich functions, in objective order."""
        return self._landscapes

    @property
    def conflict(self) -> float | None:
        """The conflict dial these objectives were built with, if known."""
        return self._conflict

    @property
    def transition_matrix(self) -> npt.NDArray[np.float64]:
        """The one transition matrix every objective shares.

        Returns:
            A `(v, v)` row-stochastic matrix whose zeros mark forbidden
            adjacencies. Singular because feasibility is a property of the
            *space*, not of an objective.
        """
        return self._landscapes[0].transition_matrix

    @property
    def optimal_sequences(self) -> Tokens:
        """Each objective's planted optimum, one row per objective.

        Returns:
            An `(n_objectives, sequence_length)` array. At `conflict = 0` every
            row is identical, which is the aligned regime stated as concretely
            as it can be.
        """
        return np.stack([landscape.optimal_sequence for landscape in self._landscapes])

    def is_feasible(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Report which sequences the shared Markov chain admits.

        Delegated to the first constituent, which is sound precisely because the
        constructor refused any set of landscapes that disagreed.

        Args:
            sequences: An `(n, sequence_length)` array of token indices.

        Returns:
            An `(n,)` boolean array.

        Raises:
            ValueError: If the input fails validation.
        """
        return self._landscapes[0].is_feasible(sequences)

    def feasible_sequence(self, seed: int = 0) -> Tokens:
        """Draw a feasible sequence, for use as a campaign's starting point.

        Independent of every planted optimum, so it leaks nothing about the
        answer on any objective.

        Args:
            seed: Seeds the walk.

        Returns:
            A feasible sequence of the landscape's length.
        """
        return self._landscapes[0].feasible_sequence(seed)

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Score each objective independently and stack the columns.

        Infeasible sequences score `-inf` on **every** objective rather than on
        some of them, which is only well defined because the feasible set is
        shared.
        """
        columns = [landscape.evaluate(sequences) for landscape in self._landscapes]
        return np.concatenate(columns, axis=1)

    def exact_pareto_front(self, candidates: Tokens | None = None) -> npt.NDArray[np.float64]:
        """The true Pareto front, computed exactly rather than approximated.

        Args:
            candidates: Sequences to search over. Defaults to the entire space,
                which is what makes the front *the* front. Pass an environment's
                [reachable_terminal_states][evogfn.env.mutation.MutationEnvironment.reachable_terminal_states]
                to get the front a campaign under a mutation budget could
                actually attain -- generally a subset, and the more honest target
                for a regret number.

        Returns:
            An `(m, n_objectives)` array of the distinct non-dominated objective
            vectors, sorted so the output is stable across runs.

        Raises:
            ValueError: If `candidates` is omitted and the space is larger than
                [MAX_ENUMERABLE_SIZE][evogfn.landscapes.base.MAX_ENUMERABLE_SIZE],
                or if the candidates fail validation.
        """
        _, values = self.exact_pareto_set(candidates)
        distinct = np.unique(values, axis=0)
        return np.asarray(distinct, dtype=np.float64)

    def exact_pareto_set(
        self, candidates: Tokens | None = None
    ) -> tuple[Tokens, npt.NDArray[np.float64]]:
        """The sequences on the Pareto front, and what they score.

        Several sequences can share one objective vector, so this is generally
        larger than
        [exact_pareto_front][evogfn.landscapes.multi_ehrlich.MultiEhrlichLandscape.exact_pareto_front];
        the difference is the redundancy a diversity metric is measuring.

        Args:
            candidates: Sequences to search over. Defaults to the entire space.

        Returns:
            An `(m, sequence_length)` array of non-dominated sequences and their
            `(m, n_objectives)` values.

        Raises:
            ValueError: If `candidates` is omitted and the space is larger than
                [MAX_ENUMERABLE_SIZE][evogfn.landscapes.base.MAX_ENUMERABLE_SIZE],
                or if the candidates fail validation.
        """
        if candidates is None:
            if self._front_cache is not None:
                cached_sequences, cached_values = self._front_cache
                return cached_sequences.copy(), cached_values.copy()
            searched = self.enumerate()
        else:
            searched = self._validate(candidates)

        values = np.asarray(self.evaluate(searched), dtype=np.float64)
        # An infeasible design scores -inf everywhere. Two of them do not
        # dominate each other, so with no feasible design in the search set they
        # would all be returned as "the front"; excluding them says instead that
        # the front is empty, which is the true answer.
        feasible = np.isfinite(values).all(axis=1)
        searched, values = searched[feasible], values[feasible]

        keep = non_dominated(values)
        front_sequences, front_values = searched[keep], values[keep]
        if candidates is None:
            self._front_cache = (front_sequences, front_values)
        return front_sequences.copy(), front_values.copy()


class _DivergentEhrlich(EhrlichLandscape):
    """An Ehrlich function whose planted optimum branches off a shared prefix.

    Exists so that several Ehrlich objectives can share a transition matrix while
    still having *different* optima. Sharing the matrix means sharing the seed --
    [EhrlichLandscape][evogfn.landscapes.ehrlich.EhrlichLandscape] draws it first,
    from a stream fixed by the seed, the vocabulary size and the density -- and
    sharing the seed would otherwise make the whole landscape identical. So the
    draws *after* the transition matrix are redirected to a per-objective stream,
    for the duration of construction only.

    Args:
        agreed_prefix: Tokens the planted optimum must copy from the shared base
            sequence before diverging. Its length is the dial.
        divergence_seed: Seeds the divergent tail and the motif placement. Made
            distinct per objective by the caller.
        kwargs: Passed to
            [EhrlichLandscape][evogfn.landscapes.ehrlich.EhrlichLandscape],
            identical across objectives.
    """

    def __init__(
        self,
        *,
        agreed_prefix: Tokens,
        divergence_seed: tuple[int, int],
        **kwargs: object,
    ) -> None:
        """Construct with the planted optimum redirected to its own stream."""
        self._agreed_prefix = np.asarray(agreed_prefix, dtype=np.int32)
        # Non-None marks "still constructing". Once the base constructor returns,
        # this class must behave exactly like its parent, or `feasible_sequence`
        # would ignore the seed it was given and hand back the planted optimum --
        # leaking the answer into every campaign's starting point.
        self._divergence: np.random.Generator | None = np.random.default_rng(divergence_seed)
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._divergence = None

    def _sample_feasible(self, rng: np.random.Generator) -> Tokens:
        """Draw the planted optimum, copying the agreed prefix then walking on.

        The junction is drawn from the shared chain's own row, so the whole
        sequence is feasible: the prefix is a prefix of a feasible sequence and
        every step after it has positive transition probability.

        Args:
            rng: The caller's stream, honoured once construction is finished.

        Returns:
            A feasible sequence of the landscape's length.
        """
        if self._divergence is None:
            return super()._sample_feasible(rng)

        length = self.sequence_length
        size = self.alphabet.size
        shared = min(self._agreed_prefix.shape[0], length)

        sequence = np.empty(length, dtype=np.int32)
        sequence[:shared] = self._agreed_prefix[:shared]
        start = shared
        if start == 0:
            sequence[0] = self._divergence.integers(size)
            start = 1
        for position in range(start, length):
            sequence[position] = self._divergence.choice(
                size, p=self._transitions[sequence[position - 1]]
            )
        return sequence

    def _carve_motifs(
        self, source: Tokens, max_spacing: int, rng: np.random.Generator
    ) -> tuple[Tokens, Tokens]:
        """Carve as the parent does, but from this objective's own stream.

        Without this every objective would place its motifs at identical offsets,
        since they share the constructor's stream up to this point. Identical
        placement is not wrong, but it is an accident rather than a decision, and
        it would make objectives differ only in their tokens.

        Args:
            source: The planted optimum to carve from.
            max_spacing: Largest gap between consecutive motif positions.
            rng: The constructor's shared stream, unused during construction.

        Returns:
            The motif tokens and their offsets.
        """
        stream = rng if self._divergence is None else self._divergence
        return super()._carve_motifs(source, max_spacing, stream)


def _require_one_space(landscapes: Sequence[EhrlichLandscape]) -> None:
    """Refuse landscapes that do not describe the same sequences and feasible set.

    The transition matrix is the one that matters. If two objectives disagree
    about which sequences are constructible then "feasible" means something
    different per objective: `is_feasible` has no answer, the environment masks
    against a matrix one objective does not recognise, and the hypervolume is
    computed over a set whose membership is undefined. Checked rather than
    assumed, because the sharing is arranged by a construction that can change.

    Args:
        landscapes: The constituents, in objective order.

    Raises:
        ValueError: If any two disagree on alphabet, sequence length or
            transition matrix.
    """
    reference = landscapes[0]
    matrix = reference.transition_matrix
    for index, landscape in enumerate(landscapes[1:], start=1):
        if landscape.alphabet != reference.alphabet:
            raise ValueError(
                f"objective {index} is written in a different alphabet than objective 0; "
                f"the objectives must score the same sequences"
            )
        if landscape.sequence_length != reference.sequence_length:
            raise ValueError(
                f"objective {index} scores sequences of length "
                f"{landscape.sequence_length}, objective 0 of length "
                f"{reference.sequence_length}; the objectives must score the same sequences"
            )
        if not np.array_equal(landscape.transition_matrix, matrix):
            raise ValueError(
                f"objective {index} carries a different transition matrix than objective 0, "
                f"so 'feasible' would mean something different per objective and the "
                f"composed landscape would have no feasible set to speak of"
            )
