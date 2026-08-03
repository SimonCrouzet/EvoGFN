"""AdaLead: the sequence-design lineage's default model-guided baseline.

Sinai et al.'s *AdaLead: A simple and robust adaptive greedy search algorithm for
sequence design* (arXiv 2010.02141), shipped in FLEXS under Apache-2.0 and
described there as the recommended benchmark algorithm. Where the GFlowNet /
offline-MBO literature runs one model-guided evolutionary comparator, this is
usually it.

It is not a second genetic algorithm
------------------------------------

[GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm] is
Stanton et al.'s, from the Ehrlich lineage, and it is **blind**: it breeds, the
oracle scores, selection keeps the best, and no model exists anywhere in the
loop. AdaLead's search is *screened by its own surrogate at every step*. That
changes what the arm is for, not merely how it is tuned:

* Its parents are the elite of the **measured** set -- everything within a
  fraction of the best true value seen, so the seeds are chosen on assay results.
* Its children are accepted or rejected on **predicted** value: a mutant
  survives only if the model scores it at or above the sequence it was mutated
  from, and only survivors are mutated again. The rollout is a hill climb *on
  the surrogate*, run to a model-query budget rather than an oracle budget.
* Only the top of everything the rollout accumulated, ranked by the model, is
  sent to the assay.

So the two arms answer different questions. A genetic algorithm losing says
population search is not enough; AdaLead losing says population search *plus a
surrogate screen* is not enough, which is the comparison a GFlowNet trained
against the same surrogate actually has to win.

It also transfers where the rest of that lineage's model-based methods do not:
CbAS, DbAS and MINs assume a generative prior trained on a large offline
dataset, and COMs assumes a differentiable relaxation scored once. AdaLead
assumes a regressor and a mutation operator, both of which this package has.

Which model, and which of the constants are theirs
--------------------------------------------------

FLEXS takes the model as a constructor argument, so the choice belongs to the
caller rather than to the method, and the default here is the same
[DeepEnsemble][evogfn.surrogate.ensemble.DeepEnsemble] every other model-bearing
arm in this package uses. The arm fits it on its own measurements: the campaign
hands this sampler no surrogate, because the model is *inside* the published
pipeline rather than a screen bolted onto it, and a second model filtering its
output would make the arm a hybrid.

Attributed to the paper's Appendix A.2.2: the elite threshold, the per-position
recombination rate, and a mutation rate of one over the sequence length.
`DEFAULT_ROLLOUT_BATCH` is the reference implementation's shipped default rather
than a published figure. `DEFAULT_RECOMBINATION_PASSES` and
`MODEL_QUERIES_PER_DESIGN` are **ours**, and are named here because the paper
states the recombination *rate* without stating how many passes precede a
rollout, and states the query budget as a free parameter of the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines._values import single_objective
from evogfn.algorithms.baselines.directed_evolution import within_budget
from evogfn.algorithms.baselines.mutagenesis import RandomMutagenesis
from evogfn.surrogate.ensemble import DEFAULT_MEMBERS, DeepEnsemble

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens
    from evogfn.env.mutation import MutationEnvironment
    from evogfn.surrogate.base import Surrogate

#: How far below the best measured value a sequence may sit and still seed a
#: rollout. Sinai et al., Appendix A.2.2: a threshold of 0.05.
DEFAULT_THRESHOLD = 0.05

#: Probability of a crossover at each position during recombination. Sinai et
#: al., Appendix A.2.2: a recombination rate of 0.2, i.e. one in five.
DEFAULT_RECOMBINE_PROB = 0.2

#: Multiplier on the per-position mutation rate, which is ``mu / L``. Sinai et
#: al. state a mutation rate of one over the sequence length, so this is one.
DEFAULT_MUTATION_MULTIPLIER = 1

#: Rollout roots evaluated together. The reference implementation's shipped
#: default for ``eval_batch_size``; the paper does not state it.
DEFAULT_ROLLOUT_BATCH = 20

#: Recombination passes over the parent pool before a rollout. **Ours.** The
#: paper states the per-position crossover rate but not how many times the pool
#: is recombined, and one pass is the least that makes the stated rate act.
DEFAULT_RECOMBINATION_PASSES = 1

#: Model evaluations the rollout may spend per design it is asked for.
#: **Ours.** FLEXS makes the per-round query budget a constructor argument with
#: no default, so there is nothing to inherit. Expressed per design rather than
#: as a flat number so that the arm's model spend tracks the plate it is filling
#: instead of changing meaning when the protocol changes shape.
MODEL_QUERIES_PER_DESIGN = 20

#: Training passes per model fit, matching the regime the rest of this package's
#: ensembles are fitted in.
DEFAULT_EPOCHS = 150

#: Fewest measurements worth fitting a model to. Below this the rollout would be
#: screening against an initialisation, so the arm draws at random instead --
#: which is what a first round is anyway.
_MIN_TRAINING = 2

#: Redraws before a rollout gives up on finding a mutant inside the mutation
#: budget. Bounded because at the edge of that budget most single substitutions
#: leave the graph, and an unbounded search there would not return.
_MUTANT_ATTEMPTS = 8


class AdaLead(Sampler):
    """Elite seeds, recombination, and a rollout screened by its own surrogate.

    One proposal call is one iteration of the published algorithm: take the
    sequences whose *measured* value is within `threshold` of the best, recombine
    them, and roll out mutations from each, keeping a child only where the model
    scores it at or above its root. Everything the rollout accepted is ranked by
    the model and the best are returned.

    The rollout's stopping rule is a **model**-query budget, which is the property
    that makes this arm a fair comparator for a GFlowNet: both spend unlimited
    free evaluations on a surrogate and the same number of real assays on the
    plate, and both report what the free half cost.

    Proposals are ranked best-first, so a campaign taking a prefix of the pool
    takes the designs the arm would have sent to the lab.

    Args:
        env: Supplies the anchor, alphabet, mutation budget and feasibility.
        model: The surrogate the rollout screens against. ``None`` builds this
            package's deep ensemble, which is what FLEXS' model argument is for:
            the choice is the caller's rather than the algorithm's.
        threshold: How far below the best measured value a sequence may sit and
            still seed a rollout. Defaults to `DEFAULT_THRESHOLD`.
        recombine_prob: Per-position crossover probability. Defaults to
            `DEFAULT_RECOMBINE_PROB`.
        mutation_multiplier: Multiplier on the ``1 / L`` per-position mutation
            rate. Defaults to `DEFAULT_MUTATION_MULTIPLIER`.
        recombination_passes: Recombination passes before each rollout. Defaults
            to `DEFAULT_RECOMBINATION_PASSES`, which is ours.
        rollout_batch: Rollout roots advanced together. Defaults to
            `DEFAULT_ROLLOUT_BATCH`.
        seed: Seeds recombination, mutation and the opening random draw.

    Raises:
        ValueError: If a probability is outside ``[0, 1]``, or a count is not
            positive.
    """

    def __init__(  # noqa: PLR0913 - the algorithm is defined by these rates
        self,
        env: MutationEnvironment,
        *,
        model: Surrogate | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        recombine_prob: float = DEFAULT_RECOMBINE_PROB,
        mutation_multiplier: int = DEFAULT_MUTATION_MULTIPLIER,
        recombination_passes: int = DEFAULT_RECOMBINATION_PASSES,
        rollout_batch: int = DEFAULT_ROLLOUT_BATCH,
        seed: int = 0,
    ) -> None:
        """Start with nothing measured and nothing fitted."""
        for label, value in [
            ("threshold", threshold),
            ("recombine_prob", recombine_prob),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must lie in [0, 1], got {value}")
        for label, count in [
            ("mutation_multiplier", mutation_multiplier),
            ("recombination_passes", recombination_passes),
            ("rollout_batch", rollout_batch),
        ]:
            if count < 1:
                raise ValueError(f"{label} must be at least 1, got {count}")
        super().__init__()

        self._env = env
        self._threshold = threshold
        self._recombine_prob = recombine_prob
        self._mutation_multiplier = mutation_multiplier
        self._passes = recombination_passes
        self._rollout_batch = rollout_batch
        self._rng = np.random.default_rng(seed)
        self._model: Surrogate = model or DeepEnsemble(
            n_tokens=env.alphabet.size,
            sequence_length=env.sequence_length,
            n_members=DEFAULT_MEMBERS,
            epochs=DEFAULT_EPOCHS,
            seed=seed,
        )
        # The opening round has no model to screen against, and "sample the
        # library uniformly" is what an explorer does before it has one -- so it
        # is the same object rather than a second copy of the same drawing code.
        self._explorer = RandomMutagenesis(env, seed=seed)

        self._sequences: list[npt.NDArray[np.integer]] = []
        self._values: list[float] = []
        self._measured: set[bytes] = set()
        self._model_calls = 0
        self._stale = True

    @property
    def name(self) -> str:
        """Short label for reporting."""
        return "AdaLead"

    @property
    def is_fitted(self) -> bool:
        """Whether the rollout has a model to screen against yet."""
        return len(self._values) >= _MIN_TRAINING

    @property
    def training_examples(self) -> int:
        """Finite measurements the model is fitted on."""
        return len(self._values)

    @property
    def proxy_calls(self) -> int:
        """Model evaluations spent, which is the arm's free half of the compute.

        Named to match the quantity every surrogate-bearing arm here reports, so
        a table can compare what AdaLead's rollout costs against what a
        GFlowNet's training costs without either being read off a different
        counter.
        """
        return self._model_calls

    def reanchored(self, env: MutationEnvironment) -> AdaLead:
        """Carry the measurements and the fitted model; only the ball moves.

        Both are anchor-free. A measurement is a sequence and a number, and the
        model is a regressor over one-hot sequences whose predictions do not
        mention a parent -- so the elite set, the model, and the ranking it
        produces all cross the move unchanged. What moves is which mutants the
        rollout may keep, which is the point of re-anchoring rather than a cost
        of it: the rollout can now climb further from the wild type than the
        previous ball allowed.

        Rebuilding through the campaign's factory would instead discard every
        measurement and refit the model from nothing each round, which on this
        arm is the difference between a screened search and a random one.

        Args:
            env: The re-anchored environment.

        Returns:
            An AdaLead over ``env``, carrying this one's data and model.
        """
        moved = AdaLead(
            env,
            model=self._model,
            threshold=self._threshold,
            recombine_prob=self._recombine_prob,
            mutation_multiplier=self._mutation_multiplier,
            recombination_passes=self._passes,
            rollout_batch=self._rollout_batch,
        )
        moved._sequences = list(self._sequences)
        moved._values = list(self._values)
        moved._measured = set(self._measured)
        moved._model_calls = self._model_calls
        moved._stale = self._stale
        moved._rng = self._rng
        moved._explorer = self._explorer.reanchored(env)
        moved._proposals_made = self._proposals_made
        return moved

    def propose(self, n: int) -> Tokens:
        """Run one iteration of the algorithm and return its best ``n`` candidates.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array, ranked best-first by the model so
            that a caller taking a prefix takes what the arm would have assayed.
            Falls back to a uniform draw where there is nothing to fit a model on
            or the rollout accepted nothing, which is the correct behaviour for a
            first round rather than a failure.
        """
        if not self.is_fitted:
            return self._draw(n)
        self._refit()
        accepted = self._rollout(n)
        if not accepted:
            return self._draw(n)
        designs = np.stack([design for design, _ in accepted])
        order = np.argsort([-value for _, value in accepted], kind="stable")
        ranked = designs[order]
        self._count(n)
        return np.stack([ranked[index % ranked.shape[0]] for index in range(n)])

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Add measurements to the set the elites are chosen from.

        Args:
            sequences: The evaluated candidates.
            values: An ``(n, 1)`` array of their objective values.

        Raises:
            ValueError: If the values carry more than one objective, which leaves
                the elite threshold with no single scale to be a fraction of.
        """
        flat = single_objective(values)
        for row, value in zip(np.ascontiguousarray(np.asarray(sequences)), flat, strict=False):
            self._measured.add(row.tobytes())
            if not np.isfinite(value):
                # An infeasible or failed assay says nothing about the response
                # surface. Keeping it in `_measured` still stops the rollout
                # spending model queries on a design that has already been tried.
                continue
            self._sequences.append(row.copy())
            self._values.append(float(value))
        self._stale = True

    def _draw(self, n: int) -> Tokens:
        """Uniform library members, charging their cost to this sampler."""
        before = self._explorer.proposals_made
        drawn = self._explorer.propose(n)
        self._count(self._explorer.proposals_made - before)
        return drawn

    def _refit(self) -> None:
        """Fit the model to everything measured, if anything has changed."""
        if not self._stale:
            return
        self._model.fit(np.stack(self._sequences), np.asarray(self._values)[:, None])
        self._stale = False

    def _predict(self, designs: npt.NDArray[np.integer]) -> npt.NDArray[np.float64]:
        """Score designs with the model, charging the calls to `proxy_calls`."""
        mean, _ = self._model.predict(designs)
        self._model_calls += int(designs.shape[0])
        return np.asarray(mean, dtype=np.float64)

    def _elite(self, n: int) -> npt.NDArray[np.integer]:
        """Measured sequences within `threshold` of the best, resized to ``n``.

        The published rule is multiplicative in the best measured value. Written
        as an absolute distance of ``threshold * |best|`` it is the identical set
        wherever values are non-negative -- which every landscape here is -- and
        it stays a *widening* of the elite set rather than a narrowing one where
        they are not, which is the direction a threshold is meant to act in.

        Args:
            n: How many roots the rollout wants. Fewer elites than that are
                repeated, as in the reference implementation, so every rollout
                starts from the same number of roots however converged the
                measured set has become.

        Returns:
            An ``(n, sequence_length)`` array of roots.
        """
        values = np.asarray(self._values, dtype=np.float64)
        best = float(values.max())
        keep = values >= best - self._threshold * abs(best)
        elites = np.stack([self._sequences[index] for index in np.flatnonzero(keep)])
        return np.asarray(np.resize(elites, (n, elites.shape[1])))

    def _rollout(self, n: int) -> list[tuple[npt.NDArray[np.integer], float]]:
        """Recombine the elites and climb the model from each, within its budget.

        Only children that *improve on their own root* enter the candidate set,
        which is the paper's own wording and the reason this is a greedy search
        rather than a model-ranked random walk: a mutant the model rates below
        the sequence it came from is generated, paid for in model queries, and
        then discarded.

        Returns:
            Every design the rollout accepted, with the value the model gave it.
            Empty where nothing improved, which is a real state rather than a
            failure -- the caller falls back to a uniform draw, since a plate has
            to be filled with something and the arm has nothing it prefers.
        """
        budget = MODEL_QUERIES_PER_DESIGN * n
        spent = 0
        accepted: dict[bytes, tuple[npt.NDArray[np.integer], float]] = {}
        seen: set[bytes] = set()
        parents = self._elite(n)

        while spent < budget:
            before = spent
            pool = parents
            for _ in range(self._passes):
                pool = self._recombine(pool)
            for start in range(0, pool.shape[0], self._rollout_batch):
                if spent + self._rollout_batch > budget:
                    break
                roots = pool[start : start + self._rollout_batch]
                root_values = self._predict(roots)
                spent += roots.shape[0]
                nodes = list(enumerate(roots))
                while nodes and spent + self._rollout_batch <= budget:
                    positions, children = self._children(nodes, seen)
                    if not children:
                        break
                    stacked = np.stack(children)
                    values = self._predict(stacked)
                    spent += stacked.shape[0]
                    nodes = []
                    for index, child, value in zip(positions, children, values, strict=True):
                        if value >= root_values[index]:
                            accepted[child.tobytes()] = (child, float(value))
                            nodes.append((index, child))
            if spent == before:
                # A whole pass produced nothing the model had not already been
                # asked about, so the neighbourhood is exhausted and looping
                # again would spin without spending the budget.
                break
        return list(accepted.values())

    def _children(
        self,
        nodes: list[tuple[int, npt.NDArray[np.integer]]],
        seen: set[bytes],
    ) -> tuple[list[int], list[npt.NDArray[np.integer]]]:
        """One in-budget, unseen mutant per surviving node.

        Args:
            nodes: The chains still climbing, each as its root's index in the
                rollout batch and the sequence it currently stands on.
            seen: Designs already generated in this rollout, mutated in place so
                a chain cannot spend the model budget re-scoring them.

        Returns:
            The root indices and the mutants, aligned. Shorter than ``nodes``
            wherever a chain could not produce an in-budget, unseen mutant
            inside `_MUTANT_ATTEMPTS` draws.
        """
        positions: list[int] = []
        children: list[npt.NDArray[np.integer]] = []
        for index, node in nodes:
            for _ in range(_MUTANT_ATTEMPTS):
                child = self._mutant(node)
                key = child.tobytes()
                if key in seen or key in self._measured:
                    continue
                if not within_budget(self._env, child):
                    continue
                seen.add(key)
                positions.append(index)
                children.append(child)
                break
        return positions, children

    def _mutant(self, design: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        """One random mutant at the published per-position rate.

        Replacements are drawn from the *other* tokens rather than from the whole
        alphabet, for the reason
        [GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm]
        gives: a uniform draw over the alphabet redraws the token already present
        one time in ``|alphabet|``, so the realised substitution rate would sit
        below the rate the paper states while being reported as it.
        """
        length = self._env.sequence_length
        size = self._env.alphabet.size
        child = np.ascontiguousarray(design).copy()
        mutate = self._rng.random(length) < self._mutation_multiplier / length
        drawn = self._rng.integers(0, size - 1, size=length)
        replacements = drawn + (drawn >= child)
        child[mutate] = replacements[mutate]
        return child

    def _recombine(self, pool: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        """Multi-point crossover over shuffled pairs, at the published rate.

        The crossover point is a *toggle*: walking the sequence, each position
        flips which parent the offspring is copying with probability
        ``recombine_prob``. That is the reading the paper's own gloss fixes --
        a one-in-five probability of a crossover at each position -- and it is
        what makes the rate a rate rather than a count.

        Offspring carrying more substitutions than the mutation budget admits
        are replaced by the parent they came from, on the same rule that keeps
        every other arm's proposals inside the graph. Feasibility is *not*
        enforced, for the reason it is not enforced on the genetic algorithm
        either: an unmasked method proposing an infeasible design and paying a
        well for it is what the constraint task measures.

        Args:
            pool: An ``(m, sequence_length)`` array of parents.

        Returns:
            An array of the same shape. An odd member out is carried through
            unchanged, having had no partner to cross with.
        """
        if self._recombine_prob == 0.0 or pool.shape[0] < 2:  # noqa: PLR2004 - a pair needs two
            return pool
        shuffled = pool[self._rng.permutation(pool.shape[0])]
        offspring = shuffled.copy()
        pairs = shuffled.shape[0] // 2
        toggles = self._rng.random((pairs, shuffled.shape[1])) < self._recombine_prob
        # A cumulative parity of the crossover points: an odd number of them
        # before a position means that position is copied from the other parent.
        take_second = np.cumsum(toggles, axis=1) % 2 == 1
        first = shuffled[0 : 2 * pairs : 2]
        second = shuffled[1 : 2 * pairs : 2]
        offspring[0 : 2 * pairs : 2] = np.where(take_second, second, first)
        offspring[1 : 2 * pairs : 2] = np.where(take_second, first, second)
        surplus = (offspring != self._env.parent[None, :]).sum(axis=1) > self._env.max_mutations
        offspring[surplus] = shuffled[surplus]
        return offspring

    def __repr__(self) -> str:
        """Name the arm and what it has measured and spent."""
        return f"AdaLead(measured={len(self._values)}, model_calls={self._model_calls})"
