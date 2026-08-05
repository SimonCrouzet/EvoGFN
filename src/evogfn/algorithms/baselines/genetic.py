"""The genetic algorithm: the baseline this project has to beat.

Directed evolution *is* a genetic algorithm, so this is not a strawman to clear
but the incumbent. Two things make that concrete. On PMO -- the field's own
sample-efficiency benchmark -- a vanilla GFlowNet trails Mol GA, and overtakes
it only by absorbing a GA. And the Ehrlich functions this package benchmarks on
were introduced with a tuned GA as their baseline and no GFlowNet evaluated at
all.

Defaults follow Stanton et al.'s reported settings -- ``p_m = 1/L``,
``p_r = 1/L`` -- and holo-bench's ``DiscreteEvolution``, so the comparison is
against the configuration its authors chose rather than one convenient to us.

Rejection sampling and the feasibility claim
--------------------------------------------

[GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm] accepts
``feasible_only``, which resamples offspring until they satisfy the
environment's constraint. This exists to make the feasibility claim falsifiable.

A masked policy is feasible by construction; the interesting question is whether
that is an *advantage*, and rejection sampling is the control that decides it. A
rejection GA wastes no oracle calls -- it discards before evaluating -- so at
equal oracle budget it may well match a masked policy. What it burns instead is
*proposals*, and that cost grows as the feasible fraction falls.

So the honest framing is: if rejection keeps up, the masking advantage is one of
proposal cost rather than sample efficiency, and should be reported that way.
[proposals_made][evogfn.algorithms.base.Sampler.proposals_made] is what makes
the difference visible.

A rejection rate on its own overstates how well rejection is coping, and
[draws_unmutated][evogfn.algorithms.baselines.genetic.GeneticAlgorithm.draws_unmutated]
is why: at ``p_m = 1/L`` the number of substitutions an offspring carries is
Poisson(1), so a large share of every batch is the anchor unchanged, and those
draws are reachable by definition. They pass the filter, hold the rejection rate
down, and search nothing. The three ``draws_*`` counters are reported together
for that reason -- two of them are a rate, and the third is what the rate means.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines._reanchor import reprojected
from evogfn.algorithms.baselines._values import single_objective

if TYPE_CHECKING:
    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment


class GeneticAlgorithm(Sampler):
    """Population-based search by mutation and recombination.

    Args:
        env: Supplies the parent, alphabet, mutation budget and feasibility.
        population_size: Individuals carried between generations.
        mutation_prob: Per-position probability of substitution. Defaults to
            ``1/L``, Stanton et al.'s setting.
        recombine_prob: Per-position probability of taking the other parent's
            token during crossover. Defaults to ``1/L``.
        survival_quantile: Fraction of the population retained as parents each
            generation. The default of 0.25 is our choice, not a published
            value -- holo-bench uses 0.01, which at our population sizes leaves
            too few parents to recombine. Stated because the argument beside it
            *does* carry its authors' value, and a reader is entitled to know
            which is which.
        carry_population: Keep the population when the campaign moves the
            anchor, re-projected onto the new mutation budget, rather than
            founding a fresh one on the new anchor. On by default because it is
            what genuinely transfers, but the choice is a real one: a rebuild is
            a restart at the best design rather than a neutral act of
            forgetting. See
            [reanchored][evogfn.algorithms.baselines.genetic.GeneticAlgorithm.reanchored].
        feasible_only: Resample offspring until constructible. The control for
            the feasibility claim; see the module docstring.
        max_attempts: Resampling rounds before giving up when
            ``feasible_only``.
        seed: Seeds the population and the operators.

    Raises:
        ValueError: If a probability is outside ``[0, 1]`` or a size is not
            positive.
    """

    def __init__(  # noqa: PLR0913 - a GA is defined by its operators' rates
        self,
        env: MutationEnvironment,
        *,
        population_size: int = 256,
        mutation_prob: float | None = None,
        recombine_prob: float | None = None,
        survival_quantile: float = 0.25,
        carry_population: bool = True,
        feasible_only: bool = False,
        max_attempts: int = 50,
        seed: int = 0,
    ) -> None:
        """Seed the population from the parent."""
        super().__init__()
        length = env.sequence_length
        self._env = env
        self._population_size = population_size
        self._mutation_prob = 1.0 / length if mutation_prob is None else mutation_prob
        self._recombine_prob = 1.0 / length if recombine_prob is None else recombine_prob
        self._survival_quantile = survival_quantile
        self._carry_population = carry_population
        self._feasible_only = feasible_only
        self._max_attempts = max_attempts

        if population_size < 1:
            raise ValueError(f"population_size must be at least 1, got {population_size}")
        for label, value in [
            ("mutation_prob", self._mutation_prob),
            ("recombine_prob", self._recombine_prob),
            ("survival_quantile", survival_quantile),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1], got {value}")

        self._rng = np.random.default_rng(seed)
        self._population = np.tile(env.parent, (population_size, 1))
        self._fitness = np.full(population_size, -np.inf)
        self._draws_attempted = 0
        self._draws_rejected = 0
        self._draws_unmutated = 0

    @property
    def name(self) -> str:
        """Short label, marking whether feasibility is enforced."""
        return "GeneticAlgorithm" + (" (rejection)" if self._feasible_only else "")

    @property
    def population(self) -> Tokens:
        """The current population."""
        return self._population.copy()

    def reanchored(self, env: MutationEnvironment) -> GeneticAlgorithm:
        """Carry the population across the move, re-projected onto the new ball.

        The judgement this replaces is that a genetic algorithm is degraded
        either way -- discarded on a rebuild, invalid if carried. The second
        half of that is not right. A population is a set of sequences and a
        sequence does not depend on an anchor; what depends on the anchor is
        only whether each individual is still *inside the budget*, and the
        algorithm already owns the operator for putting an over-budget design
        back inside one, because its own crossover produces them every
        generation. Re-anchoring re-uses it, through ``reprojected``, so a
        carried individual is indistinguishable from one the algorithm could
        have bred.

        **A re-projected individual loses its measurement.** Reverting a
        substitution makes a different sequence, and the fitness on record was
        measured on the old one. Keeping it would let selection promote a design
        on the strength of an assay never run on it -- a fabricated result that
        no later check could catch -- so those entries drop to ``-inf`` and the
        individual has to earn a score again. Individuals that came through
        untouched keep theirs, which is the part that genuinely transfers.
        [carried_fitness][evogfn.algorithms.baselines.genetic.GeneticAlgorithm.carried_fitness]
        reports how many that was.

        **A rebuild is not the neutral act of forgetting it looks like.** Do not
        read this method as an improvement to be assumed. The founding
        population is ``population_size`` copies of the new anchor, so
        rebuilding is a **restart at the best design measured**, with the
        population collapsed onto it. On a loose ball that is a strong
        exploitation move. On a tight ball it destroys the only diversity the
        search had, and the collapsed population cannot rebuild it inside one
        round's budget. Which of the two is right therefore depends on the
        per-round budget, and is why ``carry_population`` exists rather than
        this being wired shut. The rebuild behaviour stays reachable *through
        this method*, so a caller choosing it still gets the random stream
        carried across the move -- the campaign's own factory fallback rebuilds
        at the original seed and therefore re-proposes designs the campaign has
        already measured.

        Args:
            env: The re-anchored environment.

        Returns:
            A genetic algorithm over ``env``, carrying this one's population
            when ``carry_population`` is set and founded on the new anchor
            otherwise. This one's population and fitnesses are not edited,
            though its random stream advances by the draws re-projection needed
            -- the two samplers share one generator, which is what keeps a
            campaign reproducible from a single seed across the move.
        """
        moved = GeneticAlgorithm(
            env,
            population_size=self._population_size,
            mutation_prob=self._mutation_prob,
            recombine_prob=self._recombine_prob,
            survival_quantile=self._survival_quantile,
            carry_population=self._carry_population,
            feasible_only=self._feasible_only,
            max_attempts=self._max_attempts,
        )
        if self._carry_population:
            population, intact = reprojected(env, self._population, self._rng)
            moved._population = population
            moved._fitness = np.where(intact, self._fitness, -np.inf)
        moved._rng = self._rng
        moved._proposals_made = self._proposals_made
        # Carried for the same reason the proposal count is. These are campaign
        # totals, and a campaign that re-anchors three times would otherwise
        # report its last round's draws as though they were the whole run --
        # undercounting silently, and worst on exactly the runs that moved the
        # most, which are the ones anyone would look at.
        moved._draws_attempted = self._draws_attempted
        moved._draws_rejected = self._draws_rejected
        moved._draws_unmutated = self._draws_unmutated
        return moved

    @property
    def draws_attempted(self) -> int:
        """Offspring bred and offered to the feasibility filter.

        Zero unless ``feasible_only``: an arm that emits whatever it breeds
        offers nothing to a filter, and counting its output here would put a
        rejection rate of zero beside a method that never rejects.
        """
        return self._draws_attempted

    @property
    def draws_rejected(self) -> int:
        """How many of those the environment refused as unreachable."""
        return self._draws_rejected

    @property
    def draws_unmutated(self) -> int:
        """Admitted draws that were the anchor itself, carrying no substitution.

        The number that decides how a rejection rate should be read. Offspring
        are mutated at ``p_m = 1/L`` per position, so the count of substitutions
        an offspring carries is Poisson(1) and roughly a third of every batch
        carries none -- and a design identical to the anchor is inside every
        budget and feasible wherever the anchor is, so
        [is_reachable][evogfn.env.mutation.MutationEnvironment.is_reachable]
        admits all of them.

        Those admissions hold the measured rejection rate well below the rate at
        which the sampler produces anything *new*. Read alone the rate then says
        rejection sampling is coping; read beside this it says the surviving
        draws are mostly the parent, which is the finding.
        """
        return self._draws_unmutated

    @property
    def carried_fitness(self) -> int:
        """Individuals holding a measurement that was taken on them.

        Everything a genetic algorithm knows is in this number: it is the
        population where the fitnesses are real. It starts at zero -- the
        founding population is copies of the anchor with nothing measured -- and
        after a re-anchor it is how much of the population survived the move
        without being edited. Reading it across a campaign is how the question
        "does carrying the population beat starting fresh" gets an answer rather
        than an assumption.

        Returns:
            How many individuals carry a finite fitness.
        """
        return int(np.isfinite(self._fitness).sum())

    def propose(self, n: int) -> Tokens:
        """Generate ``n`` offspring from the surviving population.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array.

        Raises:
            RuntimeError: If ``feasible_only`` and the attempt budget is spent
                before enough feasible offspring are found. Returning infeasible
                designs silently would corrupt the comparison this exists for.
        """
        parents = self._survivors()
        if not self._feasible_only:
            offspring = self._breed(parents, n)
            self._count(n)
            return offspring

        collected: list[Tokens] = []
        found = 0
        for _ in range(self._max_attempts):
            batch = self._breed(parents, n)
            self._count(n)
            reachable = self._env.is_reachable(batch)
            # Counted here, before the loop can return and before it can raise.
            # The one run that exhausts is the run whose numbers explain why it
            # exhausted, and accounting written after the raise is accounting
            # that never happens on the run anybody wants it for.
            self._draws_attempted += int(batch.shape[0])
            self._draws_rejected += int((~reachable).sum())
            keep = batch[reachable]
            # Among the *admitted* draws, not among all of them: a rejected draw
            # cost a proposal and nothing else, while an admitted copy of the
            # anchor consumed a well and bought no information.
            self._draws_unmutated += int(
                np.all(keep == self._env.parent[None, :], axis=1).sum() if keep.shape[0] else 0
            )
            if keep.shape[0]:
                collected.append(keep)
                found += keep.shape[0]
            if found >= n:
                return np.concatenate(collected)[:n]
        raise RuntimeError(
            f"could not breed {n} feasible offspring in {self._max_attempts} attempts "
            f"({found} found); rejection sampling has become impractical at this "
            f"feasible density, which is itself the result"
        )

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Merge evaluated offspring into the population, keeping the best.

        Selection is over parents and offspring together, so a generation can
        never lose ground -- the elitism holo-bench's implementation uses.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which gives
                selection no single order to keep the best by.
        """
        flat = single_objective(values)
        combined = np.concatenate([self._population, np.asarray(sequences)])
        scores = np.concatenate([self._fitness, flat])
        # -inf sorts last, so unevaluated founders are displaced by anything real.
        order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))[: self._population_size]
        self._population = combined[order]
        self._fitness = scores[order]

    def _survivors(self) -> Tokens:
        """The top fraction of the population, or all of it if none are scored."""
        count = max(2, int(self._population_size * self._survival_quantile))
        if not np.isfinite(self._fitness).any():
            return self._population[:count]
        order = np.argsort(-self._fitness)[:count]
        return self._population[order]

    def _breed(self, parents: Tokens, n: int) -> Tokens:
        """Produce ``n`` offspring by recombination then mutation."""
        length = self._env.sequence_length
        size = self._env.alphabet.size
        wild_type = self._env.parent

        first = parents[self._rng.integers(0, parents.shape[0], size=n)]
        second = parents[self._rng.integers(0, parents.shape[0], size=n)]
        take_second = self._rng.random((n, length)) < self._recombine_prob
        offspring = np.where(take_second, second, first)

        mutate = self._rng.random((n, length)) < self._mutation_prob
        # Draw from the other tokens rather than from all of them. Sampling
        # uniformly over the whole alphabet redraws the token already present
        # one time in V, so the realised substitution rate would be
        # p_m * (V - 1) / V -- a 5% shortfall at the protein alphabet, which
        # would mean running Stanton et al.'s hyperparameters at 95% of their
        # published value while reporting them as the published value.
        drawn = self._rng.integers(0, size - 1, size=(n, length))
        replacements = drawn + (drawn >= offspring)
        offspring = np.where(mutate, replacements, offspring)

        # The environment admits at most max_mutations differences from the
        # parent. Offspring beyond that are outside its graph, so revert the
        # excess rather than emit sequences no sampler could have produced.
        return self._enforce_budget(offspring, wild_type)

    def _enforce_budget(self, offspring: Tokens, wild_type: Tokens) -> Tokens:
        """Revert surplus mutations so every offspring is inside the graph."""
        budget = self._env.max_mutations
        differing = offspring != wild_type[None, :]
        counts = differing.sum(axis=1)
        for row in np.flatnonzero(counts > budget):
            positions = np.flatnonzero(differing[row])
            surplus = self._rng.choice(positions, size=int(counts[row] - budget), replace=False)
            offspring[row, surplus] = wild_type[surplus]
        return offspring
