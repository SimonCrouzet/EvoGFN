"""The methodologies under test: samplers crossed with training objectives.

A methodology is whatever turns a task and a seed into a campaign. Keeping that
one callable means a GFlowNet variant, a classical baseline and a baseline given
surrogate access are all the same kind of thing to the harness, so no arm can
quietly receive a different budget, surrogate or starting point than another.

The task fixes the landscape, the protocol and the wild type; the seed fixes the
surrogate initialisation and the sampler's randomness. What is left is the
method.

The GFlowNet variants train against a proxy, never the oracle
--------------------------------------------------------------

Each builds a [ProxyLandscape][evogfn.surrogate.proxy.ProxyLandscape] over the
same surrogate instance the campaign refits, so training costs proxy evaluations
and never oracle calls. The surrogate is constitutive of that pipeline -- it is
what makes a GFlowNet trainable at 384 assays -- rather than an extra it is
handed.

The baselines are the pipelines their papers describe
-----------------------------------------------------

Exactly the components their own papers name, and nothing else. For most that is
bare: no deep ensemble screening their pool, no proxy to optimise against, and a
candidate pool their own paper would recognise. Published CMA-ES has no
surrogate; nor does a feasibility-rejecting genetic algorithm, nor either
site-saturation walk, nor MLDE, which fits its own kernel ensemble. AdaLead fits
its own too.

`alde` is the exception. Its published configuration *is* a five-member
bootstrapped ensemble read through Thompson sampling over an exhaustively
screened library, so the surrogate and the acquisition rule are constitutive of
that pipeline. Taking them away would leave an arm called ALDE that is not ALDE.

Three MLDE arms, because one budget and one landscape both fail to fit
-----------------------------------------------------------------------

MLDE's published protocol is 480 assays against the 384 every other arm spends,
and its training set alone is this suite's whole budget. So `mlde` is the
compressed arm, trained on one plate instead of four, and `mlde-over-budget` is
the published protocol, screening every plate the task affords and designing one
more. Beating a baseline at its own budget is a stronger claim than beating it at
ours; reporting only the compressed arm would rest the headline on a comparator
we had shortened.

Both gate their handover on **usable** measurements, and an infeasible assay
carries no fitness to regress on. On ``feasibility`` the mean feasible share of a
random screen is 0.053, so 384 assays buy about 20 training examples against
`mlde`'s 96 and `mlde-over-budget`'s 384: neither handover happens, and only
``fitted`` in the stored record says so.

`mlde+earlyfit` is the third arm, at the same oracle budget and at a training
size small enough that the handover can happen -- **ours**, which is what the
``+`` says here as it does for ``cmaes+dp``. Between the two rows a reader can
separate an arm that suits constrained spaces badly from an arm that never
fitted. Eight measurements is a real fit and very nearly no model selection, so a
favourable number from it bounds a trained MLDE from below; see
[evogfn.algorithms.baselines.mlde][].

What is not here
----------------

**Simulated annealing** appears in no baseline table of either lineage this suite
is measured against.
[SimulatedAnnealing][evogfn.algorithms.baselines.annealing.SimulatedAnnealing]
remains available; a registry entry is what would put it in a results row.

**δ-CS** (Kim et al., ICML 2025) is excluded for three reasons, and it is the
comparator a reviewer is most likely to name. Its step one draws from a
rank-based reweighted prior over an **offline dataset** of thousands of labelled
sequences, where a campaign here starts from a wild type and zero measurements.
Its δ is per position of the whole sequence, where this suite's state is a set of
substitutions inside a mutation budget -- the two readings differ by roughly a
factor of six in how much they destroy, and picking one is a choice made on the
baseline's behalf. And it is itself a trajectory-balance GFlowNet plus a
destroy-and-rebuild operator, so an arm we implemented and placed in the
baselines column would be a comparison between two of our implementations. If it
is ever run it belongs beside `gfn-tb` at its own budget, not in `BASELINES`.

Pool size is part of the method
-------------------------------

A genetic algorithm's pool is its population, and Stanton et al. run population
== evaluation batch == one plate. CMA-ES's is ``lambda``. Hill climbing proposes
a neighbourhood of the current point, and the site-saturation walks propose
exactly the designs their protocol names. MLDE's is an exhaustive library, and it
excludes measured designs internally because that is its protocol; ALDE's is the
same. AdaLead's is one plate, its screening happening inside its own rollout.
These differ by three orders of magnitude, so one global
``max(2048, batch * 4)`` could not be right for more than one of them -- and a
pool that large always holds enough distinct candidates to fill a plate however
badly a method has converged, which hides convergence rather than reporting it.

`genetic-feasible`'s rejection burden and the threshold at which it declares
rejection sampling impractical both key on how many candidates it is asked for,
so its behaviour moves with this.

Every methodology can follow a moved anchor
-------------------------------------------

A task that re-anchors moves its
[MutationEnvironment][evogfn.env.mutation.MutationEnvironment] to the best design
measured so far at the end of every round, and the campaign refuses at
construction if the sampler cannot follow. Two ways to follow, and the campaign
prefers the first:

* the sampler implements
  [reanchored][evogfn.loop.campaign.ReanchorableSampler.reanchored] and says what
  should happen to its own state -- a trained policy survives, a CMA-ES mean
  decoded relative to the old parent does not;
* the campaign rebuilds it from a factory, which is always correct and always
  forgetful.

Every methodology supplies a factory, so no task can re-anchor and then fail at
construction. The factories close over whatever the rebuild must not lose -- the
*same* `SequencePolicy`, so a GFlowNet keeps its trained weights, and the *same*
`ProxyLandscape`, so it keeps its link to the surrogate the campaign refits.

What a rebuild does lose is the sampler's own accounting: `proxy_calls` and the
round count restart, and
[Campaign.sampler][evogfn.loop.campaign.Campaign.sampler] returns the rebuilt
object, so a stored ``proxy_calls`` under re-anchoring counts the last anchor's
rounds rather than the campaign's.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evogfn.acquisition.rules import Greedy, Thompson, TopK
from evogfn.algorithms.base import Sampler
from evogfn.algorithms.baselines.adalead import AdaLead
from evogfn.algorithms.baselines.cmaes import CMAES
from evogfn.algorithms.baselines.directed_evolution import (
    replicated_recombination,
    replicated_walk,
)
from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
from evogfn.algorithms.baselines.mlde import MLDE
from evogfn.algorithms.baselines.mutagenesis import HillClimbing, RandomMutagenesis
from evogfn.algorithms.gflownet.genetic_gfn import GeneticConfig
from evogfn.algorithms.gflownet.objectives import ContrastiveBalance, TrajectoryBalance
from evogfn.algorithms.gflownet.sampler import GFlowNetSampler
from evogfn.algorithms.gflownet.training import TrainingConfig
from evogfn.algorithms.inner_loop import ProxyOptimising
from evogfn.env.mutation import MutationEnvironment, TerminalFeasibilityEnvironment
from evogfn.loop.campaign import Campaign
from evogfn.models.policy import AnchorConditionedPolicy, SequencePolicy
from evogfn.rewards.base import TemperedReward
from evogfn.surrogate.ensemble import DeepEnsemble
from evogfn.surrogate.proxy import ProxyLandscape

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.acquisition.base import Acquisition
    from evogfn.algorithms.gflownet.objectives import GFlowNetObjective
    from evogfn.benchmark.protocol import Protocol
    from evogfn.benchmark.tasks import Task
    from evogfn.core.types import Fitness, Tokens
    from evogfn.landscapes.base import FitnessLandscape

#: A methodology turns a task and a seed into a runnable campaign.
Methodology = Callable[["Task", int], Campaign]

#: How a classical arm's sampler is built: from the environment it searches, the
#: seed it runs at, and the protocol it runs under.
#:
#: The protocol is there for the one thing an environment cannot say. A published
#: DE protocol has a design count fixed by the *library* -- so many sites, so many
#: substitutions each -- which bears no relation to the plates a campaign offers,
#: and an arm that has to lay its designs out across those plates cannot do so
#: without knowing how many there are. Most builders ignore it, and say so by
#: naming the parameter as unused rather than by omitting it: one signature keeps
#: `classical` a single factory, and a second one would be a second place for the
#: surrogate and pool rules to drift.
ClassicalBuilder = Callable[["MutationEnvironment", int, "Protocol"], Sampler]

#: How an arm's acquisition rule is built from the campaign's seed. Seeded rather
#: than shared, because [Thompson][evogfn.acquisition.rules.Thompson] draws, and
#: an unseeded draw would make a campaign irreproducible from its own seed.
AcquisitionBuilder = Callable[[int], "Acquisition"]

#: A resolved arm setting, in the types a JSON record round-trips unchanged.
#: Anything richer -- an objective instance, a sampler class -- is recorded by
#: name, since a record is read back long after the object is gone.
ArmParameter = str | float | bool

#: Reward exponent. Jain et al. and the MOGFN papers use 3.
DEFAULT_BETA = 3.0

#: Gradient steps per campaign round. Free in oracle terms, not in wall clock.
DEFAULT_TRAINING_STEPS = 300

#: Candidates a *library* method generates per round before selection. This is
#: MLDE's regime and the regime of any arm whose job is to be screened: there
#: has to be something for the model to filter, and Wittmann et al.'s protocol
#: ranks an exhaustive library on purpose. It is not a default for everything:
#: an arm asks for it by name in `BASELINES`, and everything else gets a plate.
DEFAULT_POOL = 2048

#: Pool size meaning "exactly the plate this task measures", resolved against
#: the task's own batch size rather than written as 96. A protocol sweep varies
#: the plate, and a pool pinned at a literal would stop tracking it -- the arm
#: would then be a GA whose population is only accidentally its evaluation batch,
#: which is the one property Stanton et al.'s setting has.
PLATE_POOL = 0

#: Genetic-GFN's offline mixing ratio, after Jain et al. (2022).
DEFAULT_MIX = 0.5

#: Width of the policy trunk. Inherited rather than chosen, and it overrides
#: [SequencePolicy][evogfn.models.policy.SequencePolicy]'s own default of 256.
#: Named here rather than left as a literal at the two call sites so that the
#: selection phase can scan it against the value the headline arms actually run
#: -- a scan whose grid did not contain the shipped setting would report a curve
#: with the reported configuration missing from it.
DEFAULT_HIDDEN_DIM = 128


#: Trajectories per gradient step. Named rather than repeated as a literal at
#: the two training call sites because it is the second half of a GFlowNet
#: arm's proxy budget: a round costs ``steps * TRAINING_BATCH`` proxy
#: evaluations, so two arms that agree on ``steps`` and disagree on this do not
#: cost the same, and any comparison between them is confounded by compute.
TRAINING_BATCH = 64


@dataclass(frozen=True)
class Arm:
    """A methodology together with the settings it resolved to.

    A methodology is a closure, and a closure says nothing about itself. Two
    arms built from one factory at two reward exponents are the same code and
    the same object type; what separates them lives in captured variables that
    nothing downstream can read. So a stored result could only name its arm,
    and every question of the form "what did this row run at" became a parse of
    that name -- which is how a reader of ``beta`` came to be reading a step
    count, in silence, the day a naming scheme grew a component.

    This carries the answer instead. It is provenance and only provenance: the
    parameters are written into
    [RunRecord][evogfn.benchmark.store.RunRecord] and never consulted when
    deciding whether a cached result may be reused. Staleness stays the code
    fingerprint's job, on purpose -- a second, configuration-based invalidation
    rule would be a second thing to be wrong, and being wrong there costs a
    table that silently mixes runs.

    Being callable rather than owning the call is what keeps it invisible to
    every caller: `Methodology` is a `Callable`, an instance of this satisfies
    it, and a plain function still does. A methodology defined outside this
    module therefore keeps working and simply records nothing, which is the
    honest reading rather than a gap.

    Attributes:
        run: The methodology, called unchanged.
        parameters: What it closed over, by name.
        resolve: What it can only close over *once the task is known*, or
            ``None`` for an arm whose settings are the same everywhere. A width
            matched to the task's own sequence length and alphabet is the case
            this exists for: the number is not a property of the arm, it is a
            property of the arm on a task, and an arm carrying one static width
            would record the same figure on every shape and be wrong on all but
            one of them.
    """

    run: Methodology
    parameters: dict[str, ArmParameter]
    resolve: Callable[[Task], dict[str, ArmParameter]] | None = None

    def __call__(self, task: Task, seed: int) -> Campaign:
        """Build the campaign, adding nothing to what the methodology does.

        Args:
            task: The task to run against.
            seed: The campaign's seed.

        Returns:
            The campaign the wrapped methodology builds.
        """
        return self.run(task, seed)

    def parameters_for(self, task: Task) -> dict[str, ArmParameter]:
        """What this arm resolved to *on this task*, for the record.

        The store keys a record by ``(task, arm)``, so a per-task setting is
        already in a cell of its own and cannot collide with the same arm's
        setting elsewhere. What it must not do is go unrecorded: a control whose
        width is resolved from the task is the one arm where "which configuration
        produced this row" cannot be answered from the arm's name.

        Args:
            task: The task the record is being written for.

        Returns:
            The static settings, overlaid with whatever the task resolves. The
            overlay direction is deliberate -- a resolved width replaces the
            nominal one rather than appearing beside it, because two widths in
            one record is a reader's problem to adjudicate and only one of them
            trained.
        """
        if self.resolve is None:
            return dict(self.parameters)
        return {**self.parameters, **self.resolve(task)}


def _objective_parameters(objective: GFlowNetObjective | None) -> dict[str, ArmParameter]:
    """How an objective instance describes itself in a stored record.

    ``lam`` is read off the objective rather than taken as an argument because
    that is where it is resolved: callers hand `gflownet` a constructed
    objective, so a lambda passed to
    [SubTrajectoryBalance][evogfn.algorithms.gflownet.flow_objectives.SubTrajectoryBalance]
    is inside the object by the time this module sees it. Reading it back is
    what keeps the record describing what ran rather than what was requested.

    Args:
        objective: The training objective, or ``None`` for the sampler's own
            default.

    Returns:
        The objective's name, plus its length weighting where it has one.
        ``None`` records as ``"default"`` rather than as the name of whatever
        the sampler currently defaults to: a record must not claim a choice
        nobody made.
    """
    resolved: dict[str, ArmParameter] = {
        "objective": "default" if objective is None else type(objective).__name__
    }
    lam = getattr(objective, "lam", None)
    if lam is not None:
        resolved["lam"] = float(lam)
    return resolved


def _anchor_seed(seed: int, generation: int) -> int:
    """A distinct, reproducible seed for each anchor a campaign moves to.

    A sampler rebuilt from a factory starts from its constructor, which means it
    starts from its seed -- and a sampler re-seeded identically every round
    proposes the identical pool every round. The campaign then deduplicates
    almost all of it and stalls, spending every round after the first on the
    tail of a batch it had already generated.

    Reproducibility is not given up to fix it. The stream is a pure function of
    the campaign's seed and how many times its anchor has moved, both of which
    are fixed by the run, so a re-run reproduces it exactly.

    Args:
        seed: The campaign's seed.
        generation: How many times the anchor has moved, from zero.

    Returns:
        The seed to build the sampler for that anchor with. Generation zero
        returns `seed` unchanged, so a task that never re-anchors is bit-for-bit
        unaffected by the mechanism.
    """
    if generation == 0:
        return seed
    return int(np.random.SeedSequence([seed, generation]).generate_state(1)[0])


def _parts(
    task: Task,
    seed: int,
    *,
    terminal_feasibility: bool = False,
    bootstrap: bool = False,
    label_noise_std: float = 0.0,
) -> tuple[object, MutationEnvironment, DeepEnsemble]:
    """Everything a campaign needs that is not the method itself.

    Built identically for every methodology on a given task and seed, which is
    what makes the comparison paired rather than merely simultaneous.

    The landscape's feasibility rule is handed to the environment, which is what
    makes masked sampling possible at all. Omitting it does not raise: it
    silently switches feasibility-by-construction off, so every proposal scores
    minus infinity and the surrogate has nothing finite to fit.

    Args:
        task: Fixes the landscape, the wild type and the search radius.
        seed: Seeds the surrogate's initialisation.
        terminal_feasibility: Build
            [TerminalFeasibilityEnvironment][evogfn.env.mutation.TerminalFeasibilityEnvironment]
            instead, which defers the transition constraint from every
            intermediate to the terminal. Off by default and reachable only from
            an arm that names it, because the two classes describe *different
            search spaces*: an arm that acquired this silently would not be
            comparable to the arm it is tabled against, and nothing in the
            numbers would say so.
        bootstrap: Fit each ensemble member to a resample of the measurements
            rather than to all of them. Off by default and reachable only from an
            arm that names it, on the same reasoning: it changes what the spread
            *means*, so two arms ranked on uncertainty either side of this flag
            are not ranked on the same quantity.
        label_noise_std: Forwarded to
            [DeepEnsemble][evogfn.surrogate.ensemble.DeepEnsemble]. Zero is
            every shipped arm; the surrogate-quality intervention is the only
            caller that sets this.

    Returns:
        The landscape, the environment and an unfitted surrogate. The ensemble
        takes [DeepEnsemble][evogfn.surrogate.ensemble.DeepEnsemble]'s own
        default width, which is the five members Jain et al. use and the five
        ALDE's bench configuration names -- so the arm that requires exactly five
        inherits it here rather than restating it and risking the two drifting.
    """
    landscape = task.landscape()
    env = _environment(task, landscape, terminal_feasibility=terminal_feasibility)
    surrogate = DeepEnsemble(
        n_tokens=landscape.alphabet.size,
        sequence_length=landscape.sequence_length,
        epochs=150,
        bootstrap=bootstrap,
        label_noise_std=label_noise_std,
        seed=seed,
    )
    return landscape, env, surrogate


def _environment(
    task: Task, landscape: FitnessLandscape, *, terminal_feasibility: bool = False
) -> MutationEnvironment:
    """The graph a campaign on this task walks.

    Split out of `_parts` so that anything asking a question *about* the graph --
    what it constrains, whether a mechanism defined on it can act -- asks the
    environment a campaign would actually build, rather than a second
    reconstruction of it that could drift. `constrains_construction` is the caller
    that makes this matter: it decides whether a rung is run or reproduced, and a
    decision taken against a differently-built environment would be a decision
    about a graph nothing runs.

    Args:
        task: Fixes the wild type and the search radius.
        landscape: The oracle, already built -- a caller that has one does not
            pay for a second, and on the empirical landscapes that is a dataset
            load.
        terminal_feasibility: Defer the transition constraint to the stop action.

    Returns:
        The environment.
    """
    build = TerminalFeasibilityEnvironment if terminal_feasibility else MutationEnvironment
    # A task that declares a predicate gets it; every task that does not gets
    # the landscape's transition matrix, which is the predicate it always meant.
    # Never both -- the environment refuses two spellings of one argument, so a
    # declaration silently disagreeing with a matrix cannot arise.
    predicate = None if task.feasibility is None else task.feasibility(landscape)
    return build(
        task.parent(landscape),
        landscape.alphabet,
        max_mutations=task.max_mutations,
        transitions=None if predicate is not None else _feasibility_of(landscape),
        feasibility=predicate,
    )


def constrains_construction(task: Task) -> bool:
    """Whether anything on this task constrains the *order* substitutions are made in.

    The condition the ``+terminal`` mechanism is defined against. That rung defers
    the feasibility rule from every intermediate state to the stop action, so what
    it can possibly change is the set of construction orders the graph permits. On
    a landscape with no transition matrix there is no such rule to defer:
    [TerminalFeasibilityEnvironment][evogfn.env.mutation.TerminalFeasibilityEnvironment]
    and [MutationEnvironment][evogfn.env.mutation.MutationEnvironment] then
    describe the identical graph, and a campaign under the flag is the campaign
    without it, edge for edge and seed for seed.

    Read off the environment rather than off a list of task names, and that is the
    whole design of this function. The empirical four-site landscapes are the two
    tasks this is currently true of, and a list saying so would keep saying so
    after somebody gave one of them a transition matrix -- at which point the
    mechanism becomes measurable and the rung would go on being reproduced from an
    arm it no longer equals. Asking the environment cannot be stale: it is the
    same object the campaign runs against, built by the same function.

    Args:
        task: The task in question.

    Returns:
        ``True`` when the graph enforces adjacency during construction, which is
        exactly when deferring it to the terminal can change something.
    """
    return _environment(task, task.landscape()).constrains_intermediates


def _feasibility_of(landscape: object) -> npt.NDArray[np.floating] | None:
    """The landscape's transition matrix, when it has one.

    Returns:
        The matrix whose zeros mark forbidden adjacent token pairs, or ``None``
        for a landscape with no constructibility constraint.
    """
    matrix = getattr(landscape, "transition_matrix", None)
    return None if matrix is None else np.asarray(matrix)


def _campaign(  # noqa: PLR0913 - a campaign is defined by its protocol
    task: Task,
    landscape: object,
    env: MutationEnvironment,
    build: Callable[[MutationEnvironment], Sampler],
    surrogate: DeepEnsemble | None,
    *,
    protocol: Protocol | None = None,
    pool_size: int = DEFAULT_POOL,
    distinct_batch: bool = False,
    acquisition: Acquisition | None = None,
) -> Campaign:
    """Assemble a campaign under the task's protocol, anchored where it says.

    Args:
        task: Fixes the protocol, the search radius and whether the anchor moves.
        landscape: The oracle.
        env: The environment the sampler was built against, handed to the
            campaign so the ledger records which design each round searched from
            and so the anchor has something to move.
        build: Rebuilds the sampler for a moved anchor. Called once here for the
            opening round, so the sampler the campaign starts with and the ones
            it rebuilds come from one place and cannot drift apart.
        surrogate: Model fitted to the measurements, or ``None`` for the
            unassisted ablation.
        protocol: The rounds and plate this campaign runs, or ``None`` for the
            task's own. Only an arm whose *published* protocol does not fit the
            task's passes anything here, and it names itself accordingly -- see
            `classical`'s ``extra_rounds``. Everything else takes the task's, so
            the harness's guarantee that one protocol reaches every arm holds by
            default and is departed from only where a row says so.
        pool_size: Candidates the sampler is asked for per proposal call, or
            `PLATE_POOL` for exactly the plate. Passed in rather than computed
            here because it is the method's published population, not a harness
            setting: one formula over every arm would give a genetic algorithm a
            pool many times its own population and hand a library method the
            same number as a hill climber.
        distinct_batch: Fill the plate with distinct designs rather than with
            proposals. The plate rule is a method property in exactly the way
            the pool is, and the arm that sets it exists to measure what the
            other rule costs.
        acquisition: How predictions and uncertainty become one score, or
            ``None`` for [Greedy][evogfn.acquisition.rules.Greedy]. Greedy is the
            right default rather than merely the conventional one: every other
            arm here either has no surrogate at all or -- like MLDE -- publishes
            a protocol that ranks on the prediction and nothing else. An arm
            whose own paper names a different rule states it, which is exactly
            the axis the MLDE/ALDE comparison turns on.

    Returns:
        The campaign, which refuses at construction if the task asks to
        re-anchor and anything needed for it is missing.
    """
    resolved = task.protocol if protocol is None else protocol
    return Campaign(
        landscape=landscape,  # type: ignore[arg-type]
        sampler=build(env),
        surrogate=surrogate,
        acquisition=acquisition or Greedy(),
        selector=TopK(),
        rounds=resolved.rounds,
        batch_size=resolved.batch_size,
        pool_size=(resolved.batch_size if pool_size == PLATE_POOL else pool_size),
        distinct_batch=distinct_batch,
        environment=env,
        reanchor=task.reanchor,
        sampler_factory=build,
    )


def _policy(
    env: MutationEnvironment,
    *,
    hidden_dim: int,
    learn_flow: bool,
    seed: int,
    anchor_conditioned: bool = False,
) -> SequencePolicy:
    """A policy sized to an environment's action space.

    One place rather than three, because the sizing is the part that must not
    drift: a policy whose head is a different width from the environment's
    action count emits logits for actions that do not exist, and nothing
    downstream raises.

    Args:
        env: The environment the policy proposes into. Only its shape is read,
            and the shape is anchor-independent, so a policy built against one
            anchor is correctly sized for every other anchor of the same task.
        hidden_dim: Width of the trunk.
        learn_flow: Whether to build a flow head.
        seed: Seeds the initialisation.
        anchor_conditioned: Feed the anchor alongside the state, making the
            policy a function over anchors rather than a policy for the one it
            was built at. Off by default: it widens the trunk's input, so the
            two are not the same network and a table that mixed them would be
            comparing capacities as well as conditioning. This is the one
            argument for which the environment's *anchor* is read and not only
            its shape -- and the anchor moves, so the sampler re-binds it rather
            than this being the last word on it.

    Returns:
        The policy.
    """
    if anchor_conditioned:
        return AnchorConditionedPolicy(
            anchor=env.parent,
            n_actions=env.n_actions,
            sequence_length=env.sequence_length,
            n_tokens=env.alphabet.size,
            hidden_dim=hidden_dim,
            learn_flow=learn_flow,
            seed=seed,
        )
    return SequencePolicy(
        n_actions=env.n_actions,
        sequence_length=env.sequence_length,
        n_tokens=env.alphabet.size,
        hidden_dim=hidden_dim,
        learn_flow=learn_flow,
        seed=seed,
    )


class _RebuiltOnMove(Sampler):
    """A GFlowNet stripped of its ability to follow a moved anchor.

    The rebuild half of the amortisation ablation, and the mechanism is a
    deliberate omission: the campaign resolves
    [ReanchorableSampler][evogfn.loop.campaign.ReanchorableSampler] first and
    falls back to its factory, so a wrapper that does not implement
    ``reanchored`` is how an arm asks to be rebuilt. Deleting the hook from
    [GFlowNetSampler][evogfn.algorithms.gflownet.sampler.GFlowNetSampler] would
    do the same thing to every arm in the suite; hiding it behind a wrapper
    makes it a property of one arm, which is what an ablation needs it to be.

    What the rebuild discards is the point of the arm. The factory that
    replaces this builds a *fresh* policy, so the weights the previous anchor
    trained are gone and the next ball is learned from nothing -- which is the
    position a genetic algorithm is structurally in, its operator being the
    same before and after a move.

    What the rebuild must not discard is the accounting. ``proxy_calls`` and
    the teacher's tallies are campaign totals, and a rebuilt sampler starts
    them at zero; read off the last one alone they would report the final
    anchor's rounds as the arm's whole compute, which is exactly the number a
    reader would use to check that this arm and the carried one cost the same.
    So the counters are summed over every sampler the factory has built rather
    than read off the current one.

    Args:
        inner: The sampler for the current anchor.
        built: Every sampler the arm's factory has built, this one last. Shared
            with the factory rather than copied, so a wrapper made now still
            totals correctly over samplers made later.
    """

    def __init__(self, inner: GFlowNetSampler, built: list[GFlowNetSampler]) -> None:
        """Wrap the current sampler without running it."""
        super().__init__()
        self._inner = inner
        self._built = built

    @property
    def name(self) -> str:
        """Short label marking that this arm is rebuilt at every move."""
        return f"{self._inner.name} (rebuilt)"

    @property
    def inner(self) -> GFlowNetSampler:
        """The sampler for the current anchor, for inspection."""
        return self._inner

    @property
    def proposals_made(self) -> int:
        """Candidates generated across every anchor this arm has searched."""
        return sum(sampler.proposals_made for sampler in self._built)

    @property
    def proxy_calls(self) -> int:
        """Proxy evaluations spent across every anchor, so compute is comparable."""
        return sum(sampler.proxy_calls for sampler in self._built)

    @property
    def bred_designs(self) -> int:
        """Genetic offspring the policies were asked to construct a path to."""
        return sum(sampler.bred_designs for sampler in self._built)

    @property
    def unconstructible_designs(self) -> int:
        """How many of those had no construction order at all."""
        return sum(sampler.unconstructible_designs for sampler in self._built)

    @property
    def unconstructible_fraction(self) -> float:
        """Share of bred designs no policy could construct.

        Returns:
            The share in ``[0, 1]``, and ``0.0`` when nothing was bred -- a
            share of nothing, which is why it is meaningless without
            `bred_designs` beside it.
        """
        bred = self.bred_designs
        return self.unconstructible_designs / bred if bred else 0.0

    def propose(self, n: int) -> Tokens:
        """Propose from the current anchor's sampler.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array.
        """
        return self._inner.propose(n)

    def observe(self, sequences: Tokens, values: Fitness) -> None:
        """Pass measurements through to the current anchor's sampler.

        Args:
            sequences: The candidates that were measured.
            values: Their measured objective values.
        """
        self._inner.observe(sequences, values)

    def __repr__(self) -> str:
        """Name the wrapped sampler and how many anchors have been built."""
        return f"_RebuiltOnMove({self._inner!r}, anchors={len(self._built)})"


def classical(  # noqa: PLR0913 - each argument names one published component
    build: ClassicalBuilder,
    *,
    surrogate: bool = False,
    proxy_access: bool = False,
    pool_size: int = PLATE_POOL,
    distinct_batch: bool = False,
    acquisition: AcquisitionBuilder | None = None,
    bootstrap: bool = False,
    extra_rounds: int = 0,
) -> Methodology:
    """A classical baseline, as published or with a named piece added.

    The defaults are the bare pipeline: no surrogate, no proxy, greedy ranking,
    an ensemble whose spread comes from initialisation alone, and a pool the size
    of the plate. Defaulting the other way would put a deep ensemble in front of
    every classical baseline in the suite -- a step in none of their papers --
    and make the headline table a comparison between hybrids nobody proposed.

    So every argument below is something a caller has to ask for by name. Two
    quite different kinds of caller do: an *ablation*, which adds a rung to a
    published pipeline and says so in its arm name, and a *published pipeline
    that contains the component already*, which asks for it because its own paper
    does. The arguments cannot tell those apart, and nothing here tries to --
    which is which is recorded in the arm's parameters and asserted in the
    registry below, where the claim can be read against the source.

    Args:
        build: Makes the sampler from an environment, a seed and the protocol it
            will run under. See `ClassicalBuilder` for why the protocol is in
            that signature even though most builders ignore it.
        surrogate: Whether a surrogate screens the proposal pool. For an
            ablation this is the ``+screen`` rung -- the model filters what gets
            measured and the search itself stays blind. For ``alde`` it is the
            method.
        proxy_access: Whether the sampler may also *optimise* against the
            surrogate, as the GFlowNet does. This is the ``+search`` rung, and it
            is what separates "the surrogate won" from "the constructive sampler
            won" -- a real question, and an attribution question, so it belongs
            to a named ablation rather than to every arm silently.
        pool_size: Candidates per proposal call, defaulting to one plate. A
            screened arm needs more than a plate or there is nothing to screen.
        distinct_batch: Fill the plate with distinct designs rather than with
            proposals.
        acquisition: Builds the rule that turns predictions and uncertainty into
            one score, from the campaign's seed. ``None`` is greedy ranking.
            Exposed because "greedy versus a rule that reads the uncertainty" is
            the axis separating the two published active-learning protocols this
            suite runs, and an axis a caller cannot set is an axis nobody can
            measure.
        bootstrap: Fit each ensemble member to a resample of the measurements.
            Inert without ``surrogate``, since there is then no ensemble to fit.
        extra_rounds: Plates this arm runs **beyond the task's own budget**,
            because its published protocol does not fit inside that budget. This
            is the one argument here that breaks the harness's "every arm sees
            the same protocol" guarantee, so it is deliberately the crudest
            possible knob -- whole rounds at the task's own plate, never a
            different plate and never a different radius per round -- and an arm
            that sets it must say so in its name.

            The only arm that does is MLDE, whose protocol is
            `PUBLISHED_TRAINING_SIZE` screened variants plus one designed plate.
            On this suite's four-plate protocol one extra plate resolves to
            exactly Wittmann et al.'s 480, with exactly their 384-variant
            training split; on a task of another shape it is the same protocol at
            that task's scale, screen everything then design one plate more.

            Two consequences a reader has to be told about, since neither is
            visible in the stored ``protocol`` field -- which is the *task's*
            repr, written by [run_task][evogfn.benchmark.suite.run_task] and not
            by the arm. First, the record's ``oracle_calls`` exceeds the budget
            that field names, and that gap *is* the disclosure. Second, on a
            re-anchoring task an extra round is also an extra round of *reach*:
            the task's audited attainable optimum is taken over
            [Task.search_budget][evogfn.benchmark.tasks.Task.search_budget]
            rounds, so a regret stored for this arm is taken against a bound
            derived for a shorter campaign and can come out negative. That is a
            real limitation of running an over-budget arm inside a per-task
            audit, not a bug in either.

    Returns:
        A methodology, carrying the settings above so that a record written by
        it says what it ran at rather than leaving that to be read out of the
        arm's name.
    """

    def methodology(task: Task, seed: int) -> Campaign:
        landscape, env, ensemble = _parts(task, seed, bootstrap=bootstrap)
        # Resolved once and handed to both the builder and the campaign. A
        # builder that laid its designs out across the task's plate count while
        # the campaign ran a different one would be a walk whose site order and
        # whose rounds disagreed, and nothing would raise.
        protocol = (
            task.protocol
            if extra_rounds == 0
            else replace(task.protocol, rounds=task.protocol.rounds + extra_rounds)
        )
        # One proxy for the whole campaign, closed over rather than rebuilt: it
        # wraps the surrogate instance the campaign refits in place, and a fresh
        # one per anchor would still see the same model but would make that
        # dependence look accidental. Its shape does not change with the anchor.
        proxy = (
            ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)
            if proxy_access
            else None
        )

        generation = count()

        def make(anchored: MutationEnvironment) -> Sampler:
            """Build the baseline against whichever anchor the campaign is at."""
            sampler = build(anchored, _anchor_seed(seed, next(generation)), protocol)
            return sampler if proxy is None else ProxyOptimising(sampler, proxy=proxy)

        return _campaign(
            task,
            landscape,
            env,
            make,
            ensemble if surrogate else None,
            protocol=protocol,
            pool_size=pool_size,
            distinct_batch=distinct_batch,
            acquisition=None if acquisition is None else acquisition(seed),
        )

    return Arm(
        methodology,
        {
            "family": "classical",
            # The sampler by the name of the function that builds it, which is
            # what separates `genetic` from `genetic-feasible`: same class,
            # different constructor arguments, and a record naming only the
            # class could not tell the rejection-sampling arm from the plain
            # one.
            "sampler": str(getattr(build, "__name__", "unknown")).strip("_").replace("_", "-"),
            "surrogate": surrogate,
            "proxy_access": proxy_access,
            # As configured, and `PLATE_POOL` is stored as the zero it is: it
            # means "the task's plate", which is resolved per task and is
            # therefore already stated by the protocol the record carries.
            # Resolving it here would need a task this arm has not been handed.
            "pool_size": pool_size,
            "distinct_batch": distinct_batch,
            # By name rather than by instance, and resolved from an arbitrary
            # seed: a record is read back long after the object is gone, and what
            # a reader needs from it is which *rule* ranked the pool, not which
            # draw it made.
            "acquisition": "Greedy" if acquisition is None else type(acquisition(0)).__name__,
            "bootstrap": bootstrap,
            # Rounds beyond the task's, so a record says by how much its arm
            # exceeded the budget the `protocol` field beside it names. Zero for
            # every arm that ran the task's own protocol, which is all but one.
            "extra_rounds": extra_rounds,
        },
    )


def gflownet(  # noqa: PLR0913 - an arm is defined by its hyperparameters
    objective: GFlowNetObjective | None = None,
    *,
    steps: int = DEFAULT_TRAINING_STEPS,
    beta: float = DEFAULT_BETA,
    learn_flow: bool = False,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    carry_policy: bool = True,
    terminal_feasibility: bool = False,
    anchor_conditioned: bool = False,
    match_anchor_capacity: bool = False,
    pool_size: int = DEFAULT_POOL,
    label_noise_std: float = 0.0,
) -> Methodology:
    """A GFlowNet trained against the surrogate proxy.

    Args:
        objective: How balance violation is measured. Defaults to trajectory
            balance.
        steps: Gradient steps per round.
        beta: Reward exponent.
        learn_flow: Whether to build a flow head. Required by the
            detailed-balance family and useless to the others, so it is set by
            the caller alongside the objective rather than guessed.
        hidden_dim: Width of the policy trunk. Exposed because it is capacity,
            and capacity is the one axis where "the default was fine" and "the
            policy could not represent the target" produce the same flat table:
            a comparison at a single width cannot tell them apart. Its default
            is the shipped setting, so leaving it alone changes nothing.
        carry_policy: Whether the trained weights survive a moved anchor. On is
            the shipped method and is what *amortisation* means here: the policy
            arrives at the new ball already knowing something about the
            landscape. Off rebuilds it from scratch at every move, which is the
            position a genetic algorithm is structurally in -- its operator is
            the same before and after the move -- and so is the control the
            claim needs. It changes nothing on a task whose anchor never moves,
            because nothing is ever rebuilt there.
        terminal_feasibility: Enforce the transition constraint on the design
            rather than on every intermediate, through
            [TerminalFeasibilityEnvironment][evogfn.env.mutation.TerminalFeasibilityEnvironment].
            Off is the shipped method. On, the arm searches a strictly larger
            space -- every feasible design within the budget, rather than only
            those with a feasible construction order -- and pays for it with
            trajectories that dead-end on infeasible designs, which the assay
            scores ``-inf``. It is inert on a landscape with no transition
            matrix, so an arm that sets it differs from one that does not only
            where feasibility is actually constrained.
        anchor_conditioned: Feed the anchor to the policy, through
            [AnchorConditionedPolicy][evogfn.models.policy.AnchorConditionedPolicy].
            Off is the shipped method, which sees only the resulting sequence and
            so cannot tell a substitution from an inherited residue. On, the
            policy fits a function over anchors, which is something only an arm
            with a policy can do at all: a genetic algorithm's representation of
            where it is *is* its population.
        pool_size: Candidates screened per proposal call, or `PLATE_POOL` for
            exactly the plate and no screening at all. Exposed because the two
            decomposition rungs below are defined by it: screening spends the
            *surrogate*, which is not the campaign's budget -- the oracle budget
            is the plate, and it is identical for every arm here whatever this is
            set to. So a rung that screens more is not being given a larger
            budget, it is being given more compute against a free model, which is
            what makes "how far does masking plus screening get on its own"
            answerable rather than rhetorical.
        match_anchor_capacity: Widen the plain trunk, per task, until it carries
            what an anchor-conditioned trunk of `hidden_dim` would carry on that
            task's own shape. Off is every arm but the capacity control. It has
            to be a flag rather than a width passed in, because the width is not
            knowable at the call site: the tier runs five tasks of three shapes
            and the matching width differs on each, so a number chosen here would
            be right on one of them and silently wrong -- in the direction that
            manufactures the effect the control exists to rule out -- on the rest.
            See `matched_capacity`.
        label_noise_std: Forwarded to `_parts`, which forwards it to
            [DeepEnsemble][evogfn.surrogate.ensemble.DeepEnsemble]. Zero
            reproduces the shipped arm. This is the surrogate-quality
            intervention: the correlation between surrogate-oracle Pearson r
            and whether masking or learning carries the GFlowNet's margin is
            readable off stored campaigns without touching this, but only
            deliberately degrading fit quality turns that correlation into a
            causal claim.

    Returns:
        A methodology, carrying its settings for the record. The capacity control
        also carries a resolver, since its width is a property of the arm *on a
        task* and a single recorded number would describe two thirds of its rows
        wrongly.

    Raises:
        ValueError: If capacity matching is asked of an anchor-conditioned arm.
            The control exists to be the conditioned arm's size *without* the
            conditioning; an arm that is both would be matched against itself and
            would take a width the search cannot even define.
    """
    if match_anchor_capacity and anchor_conditioned:
        raise ValueError(
            "match_anchor_capacity widens a plain trunk to an anchor-conditioned one's size, "
            "so an arm that is already anchor-conditioned has nothing to match; an arm setting "
            "both would be the capacity control for itself"
        )

    def width_on(task: Task) -> int:
        """The trunk width this arm trains at on this task."""
        if not match_anchor_capacity:
            return hidden_dim
        return matched_capacity_for(task, hidden_dim, learn_flow=learn_flow).hidden_dim

    def methodology(task: Task, seed: int) -> Campaign:
        width = width_on(task)
        landscape, env, ensemble = _parts(
            task,
            seed,
            terminal_feasibility=terminal_feasibility,
            label_noise_std=label_noise_std,
        )
        # Built once and closed over when the policy is carried, so a rebuild
        # for a moved anchor keeps the trained weights. It survives the move
        # because its action space -- length * |alphabet| + 1 indices -- and its
        # input, the state sequence, are both properties of the space rather
        # than of the anchor. Only the masks move.
        carried = (
            _policy(
                env,
                hidden_dim=width,
                learn_flow=learn_flow,
                seed=seed,
                anchor_conditioned=anchor_conditioned,
            )
            if carry_policy
            else None
        )
        proxy = ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)

        generation = count()
        built: list[GFlowNetSampler] = []

        def make(anchored: MutationEnvironment) -> Sampler:
            """Build the sampler against whichever anchor the campaign is at."""
            stream = _anchor_seed(seed, next(generation))
            # A fresh policy per anchor when nothing is carried, seeded from the
            # same stream the sampler is. At the opening anchor that stream *is*
            # the campaign's seed, so the two arms start from identical weights
            # and diverge only where the anchor moves -- which is the only place
            # the axis is supposed to act.
            sampler = GFlowNetSampler(
                anchored,
                carried
                if carried is not None
                else _policy(
                    anchored,
                    hidden_dim=width,
                    learn_flow=learn_flow,
                    seed=stream,
                    anchor_conditioned=anchor_conditioned,
                ),
                proxy=proxy,
                reward=TemperedReward(beta=beta),
                # Identical whether or not the policy is carried, which is what
                # makes the two comparable: a round costs `steps` gradient steps
                # and `steps * TRAINING_BATCH` proxy evaluations either way, and
                # every round retrains, so an arm that starts each ball from
                # nothing is given exactly as much compute to recover with.
                # `steps or 1` keeps the config well-formed where the arm
                # does not train at all; `train` is what actually decides, and
                # the recorded `steps` stays 0 so a reader of the row sees the
                # arm for what it is.
                config=TrainingConfig(steps=steps or 1, batch_size=TRAINING_BATCH, seed=stream),
                train=steps > 0,
                objective=objective,
                seed=stream,
            )
            if carried is not None:
                return sampler
            built.append(sampler)
            return _RebuiltOnMove(sampler, built)

        return _campaign(task, landscape, env, make, ensemble, pool_size=pool_size)

    def resolved(task: Task) -> dict[str, ArmParameter]:
        """What the capacity control came out at on this task, for the record.

        The residual is the reason this is stored rather than left derivable. It
        is the arm's entire warrant -- a control that is not smaller than the arm
        it controls for can only understate that arm's effect -- and it is
        different on every shape the tier runs. Put on the record it is checkable
        from the results alone, by a reader who has neither the architecture nor
        this module in front of them; left implicit, "the control was the right
        size" is an assertion about code somebody would have to re-run to test.
        """
        match = matched_capacity_for(task, hidden_dim, learn_flow=learn_flow)
        return {
            # Overwrites the nominal width, which on this arm is the *conditioned*
            # trunk being matched and never the one that trained.
            "hidden_dim": match.hidden_dim,
            "capacity_parameters": match.parameters,
            "capacity_target": match.target,
            "capacity_residual": match.residual,
        }

    return Arm(
        methodology,
        {
            "family": "gflownet",
            **_objective_parameters(objective),
            "steps": steps,
            "beta": beta,
            "learn_flow": learn_flow,
            "hidden_dim": hidden_dim,
            "pool_size": pool_size,
            "carry_policy": carry_policy,
            "terminal_feasibility": terminal_feasibility,
            "anchor_conditioned": anchor_conditioned,
            "match_anchor_capacity": match_anchor_capacity,
            "label_noise_std": label_noise_std,
        },
        resolved if match_anchor_capacity else None,
    )


def genetic_gflownet(
    objective: GFlowNetObjective | None = None,
    *,
    steps: int = DEFAULT_TRAINING_STEPS,
    beta: float = DEFAULT_BETA,
    mix: float = DEFAULT_MIX,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
) -> Methodology:
    """A GFlowNet taught by a genetic algorithm, after Kim et al. (2024).

    The variant most likely to matter here: directed evolution is a genetic
    algorithm and the Ehrlich benchmark's own baseline is one, so a method that
    absorbs the GA into the policy's training signal rather than competing with
    it is the shape this problem invites.

    Args:
        objective: How balance violation is measured.
        steps: Gradient steps per round.
        beta: Reward exponent.
        mix: Share of each training batch bred by the genetic teacher rather
            than sampled from the policy. This is the knob the method is *about*
            -- at zero it is an ordinary GFlowNet and at one the policy only
            ever sees the GA's offspring -- so it is exposed rather than left at
            the config default.
        hidden_dim: Width of the policy trunk, as in
            [gflownet][evogfn.benchmark.methods.gflownet]. Kept on the same
            default so that ``mix = 0`` really is a plain GFlowNet: the `mix`
            axis brackets its own claim by reducing to `gfn-tb` at zero, and a
            different width here would quietly make those two arms different
            methods while the sweep reported them as one axis.

    Returns:
        A methodology.
    """

    def methodology(task: Task, seed: int) -> Campaign:
        landscape, env, ensemble = _parts(task, seed)
        policy = _policy(env, hidden_dim=hidden_dim, learn_flow=False, seed=seed)
        proxy = ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length)
        generation = count()

        def make(anchored: MutationEnvironment) -> Sampler:
            """Build the sampler and a fresh teacher for the current anchor.

            The policy carries over and the genetic teacher does not, which is
            the right split rather than an oversight: a population bred around
            the old parent can sit wholly outside the new mutation budget, so
            re-seeding it at the new anchor is what keeps the teacher's
            offspring inside the space the policy is being taught to sample.
            """
            stream = _anchor_seed(seed, next(generation))
            return GFlowNetSampler(
                anchored,
                policy,
                proxy=proxy,
                reward=TemperedReward(beta=beta),
                config=TrainingConfig(steps=steps, batch_size=TRAINING_BATCH, seed=stream),
                objective=objective,
                genetic=GeneticAlgorithm(anchored, seed=stream),
                genetic_config=GeneticConfig(offspring=64, mix=mix, warmup=max(steps // 10, 1)),
                seed=stream,
            )

        return _campaign(task, landscape, env, make, ensemble, pool_size=DEFAULT_POOL)

    return Arm(
        methodology,
        {
            "family": "genetic-gflownet",
            **_objective_parameters(objective),
            "steps": steps,
            "beta": beta,
            "mix": mix,
            "hidden_dim": hidden_dim,
            "pool_size": DEFAULT_POOL,
        },
    )


def _random(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    return RandomMutagenesis(env, seed=seed)


def _hill_climb(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    """Random single-substitution neighbours of the incumbent, with restarts.

    The *other* thing the field calls directed evolution: HDBO's greedy
    incumbent search, which draws a random substitution anywhere in the sequence
    and keeps no record of which positions it has already changed. Kept clearly
    apart from
    [SingleStepWalk][evogfn.algorithms.baselines.directed_evolution.SingleStepWalk],
    which saturates a site before committing to it and then never returns to it.
    Collapsing the two would leave the suite with one arm where two different
    published comparators are expected.
    """
    return HillClimbing(env, seed=seed)


def _genetic(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    return GeneticAlgorithm(env, seed=seed)


def _cmaes(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    """CMA-ES under the straightforward decoder, which is what gets reported.

    A separable Gaussian over per-position logits cannot represent a transition
    constraint, so on a constrained instance the raw decode is not buildable and
    something has to repair it. Which repair is chosen is a choice *we* make on
    the baseline's behalf, and it is not specified anywhere in the literature --
    so the reported arm takes the obvious one: accept substitutions by
    descending gain while the design stays legal.

    The exact alternative is a dynamic program that returns the best legal
    design outright. It scores better, and that is precisely why it is not here:
    it is a stronger decoder than any published account of this method uses, and
    a baseline reported through machinery of our own devising is no longer the
    published method. It is kept beside this one, as a study rather than as a
    baseline, so the difference between them can be measured.
    """
    return CMAES(env, repair="greedy", seed=seed)


def _cmaes_exact(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    """CMA-ES handed an exact constrained decoder, for the study only.

    Not a baseline, and it must not enter a table that reports published
    methods. What it measures is how much of a constrained-decoding result
    belongs to the search distribution and how much to the decoder -- a question
    worth answering, and a different question from how the published method
    performs.
    """
    return CMAES(env, repair="exact", seed=seed)


def _mlde(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    """Machine-learning-directed evolution, the method protein engineers run.

    The most important baseline here after the genetic algorithm: it is what
    Wittmann et al. actually do, at almost exactly this budget, and its whole
    claim is reaching the answer in hundreds of assays rather than thousands.
    """
    return MLDE(env, seed=seed)


def _mlde_as_published(env: MutationEnvironment, seed: int, protocol: Protocol) -> Sampler:
    """MLDE at Wittmann et al.'s own split: screen every plate but the last.

    The published protocol is a *training size*, not a budget: screen 384
    variants at random, fit the ensemble to them, and spend one more plate on
    what it predicts best. Compressed into this suite's 384 assays that becomes
    one training plate instead of four, which its own module records as a
    handicap -- the paper's outcome improves with training-set size, so a
    96-variant MLDE is a weaker MLDE.

    So the training size is read off the protocol rather than pinned at
    [PUBLISHED_TRAINING_SIZE][evogfn.algorithms.baselines.mlde.PUBLISHED_TRAINING_SIZE].
    Every plate but the last is screening; the last is the designed plate. On
    the four-plate protocol this task's arm is registered against, that *is*
    384 and 96 -- the published numbers, reached by construction rather than by
    two constants agreeing, which is what keeps the arm on Wittmann et al.'s
    split if the shared protocol ever moves.

    Pinning 384 outright would be worse in exactly one way and it is the way that
    matters: on any task whose whole campaign is shorter than 384 the sampler
    would never reach its training size, would propose at random for every round,
    and would still be tabled as MLDE. A random arm under a supervised arm's name
    is the failure this whole module is arranged against.

    One thing this does *not* fix, because it belongs to the sampler and to
    `mlde` equally. The handover is gated on **usable** measurements:
    [MLDE.observe][evogfn.algorithms.baselines.mlde.MLDE.observe] drops an
    infeasible assay, having no fitness to regress on. So on a landscape with a
    transition constraint the screening plates yield fewer training examples than
    they cost, the model takes over later than the protocol says, and where the
    infeasible share is large it never takes over at all -- at which point the
    row is a random baseline under a supervised method's name. Reading
    ``is_fitted`` off the finished sampler is what distinguishes the two, and
    nothing in the spend or the plate count does.
    """
    return MLDE(env, training_size=protocol.batch_size * (protocol.rounds - 1), seed=seed)


def _mlde_adapted(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    """**Our** adaptation of MLDE: a training size a constrained screen returns.

    Not Wittmann et al.'s protocol, and not `_mlde`'s compression of it either.
    The published method assumes an assay comes back with a number. On a
    landscape where 95% of wells come back with nothing to regress on, its
    supervised phase is unreachable at *any* budget a laboratory would run -- the
    training set grows at a twentieth of the rate the budget shrinks, so its 384
    usable measurements would cost upwards of seven thousand assays. `mlde` and
    `mlde-over-budget` both therefore spend the whole campaign screening at
    random on such a task, and the two rows they produce are random baselines
    under a supervised method's name. Both stay, because that is the finding.

    This arm lowers the training size to
    [ADAPTED_TRAINING_SIZE][evogfn.algorithms.baselines.mlde.ADAPTED_TRAINING_SIZE]
    so that the handover happens, at the same oracle budget as every other arm.
    It is how the question *would MLDE be competitive here if it ever got to be
    MLDE?* can be asked at all, and it is **not** a claim about what the
    published method does: the value is derived from the suite's own measured
    feasible share rather than from anything in the paper, and it is a twelfth of
    a training sample the module beside it already calls a compression.

    Two properties it deliberately does not have. It is not per-task: one stated
    number, not one read off each landscape's feasibility, since a configuration
    tuned on the task it is scored on is a family of configurations rather than
    one. And it is not over budget: the deviation is a parameter of ours, not an
    assay of ours, so it stays inside the protocol every other arm runs and only
    `mlde-over-budget` departs from that.
    """
    return MLDE.adapted(env, seed=seed)


def _single_step(env: MutationEnvironment, seed: int, protocol: Protocol) -> Sampler:
    """Traditional directed evolution, to ALDE's specification.

    The comparator every "MLDE beats DE" claim in the wet-lab literature is
    measured against, and the half of the two lineages' shared intersection this
    suite would otherwise be missing.

    One walk costs a fraction of a plate, so the arm runs as many walks
    concurrently as the plate holds, each at its own site order. The plate is
    what it reads from the protocol; the walk itself needs nothing from it, since
    a walk advances one site per set of results and the mutation budget decides
    how far down its order it can get.
    """
    return replicated_walk(env, batch_size=protocol.batch_size, seed=seed)


def _recomb(env: MutationEnvironment, seed: int, protocol: Protocol) -> Sampler:
    """Li et al.'s recombination arm: saturate every site, then combine winners.

    Reads the whole protocol, not just the plate. The arm's design count is fixed
    by how many sites it saturates, which their own budget formula states as a
    free parameter -- so how many sites it can afford, how to lay the saturations
    out so a plate is left for the recombinant, and how many replicates the
    campaign can carry are all questions only the campaign's shape can answer.
    """
    return replicated_recombination(
        env, rounds=protocol.rounds, batch_size=protocol.batch_size, seed=seed
    )


def _adalead(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    """FLEXS' recommended benchmark algorithm, at the paper's own rates.

    It carries its own surrogate, which is the whole difference from `_genetic`:
    the rollout keeps a mutant only where the model scores it at or above the
    sequence it came from, so the search is screened at every step rather than
    only at the plate. Given no campaign surrogate for that reason -- a second
    model filtering its output would make the arm a hybrid.
    """
    return AdaLead(env, seed=seed)


def _feasible_genetic(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    """A genetic algorithm that rejection-samples until its offspring are legal.

    The control for the feasibility claim. Where masking is free, rejection
    sampling costs proposals, and what those proposals cost on a sparse feasible
    set is the quantity this arm exists to expose.

    It is asked for one plate at a time, like the genetic algorithm it is a
    variant of, so ``max_attempts`` bounds 200 attempts at breeding 96 legal
    offspring rather than at breeding 2048. The number of proposals it burns per
    measured design is therefore the method's, not the harness's, which is what
    the claim needs it to be.
    """
    return GeneticAlgorithm(env, seed=seed, feasible_only=True, max_attempts=200)


def _masked_genetic(env: MutationEnvironment, seed: int, _protocol: Protocol) -> Sampler:
    """A genetic algorithm whose variation operator only proposes legal moves.

    The second control for the feasibility claim, alongside `_feasible_genetic`.
    Construction is available to a GA too: this arm decides an offspring the
    same way `genetic` does and then builds it edge by edge through the same
    `forward_mask` a masked GFlowNet policy is scored against, dropping an
    individual edit where the mask forbids it rather than discarding the whole
    offspring the way rejection does. If this still loses to the GFlowNet, the
    gap is attributable to learning rather than to rejection's mechanics.
    """
    return GeneticAlgorithm(env, seed=seed, construct_feasible=True)


#: The classical baselines, each as its own paper published it. Directed
#: evolution *is* a genetic algorithm, so these are the incumbents rather than
#: strawmen to be cleared -- and an incumbent is a whole pipeline, which is what
#: a lab chooses between. Every arm here carries exactly the components its own
#: source names, and a pool that source would recognise.
#:
#: The set is drawn from the two baseline tables this work is read against, and
#: the two overlap in only two places -- a random floor and a greedy incumbent
#: search -- so most rows answer to one lineage or the other:
#:
#: =====================  ===================================================
#: arm                    what expects it
#: =====================  ===================================================
#: ``random``             both; the near-universal floor
#: ``hill-climb``         the in-silico lineage's greedy incumbent search
#: ``single-step``        the wet-lab lineage's DE walk, and the other half
#:                        of the intersection
#: ``recomb``             the wet-lab lineage's independent-site DE variant
#: ``mlde``               the supervised protein-engineering reference,
#:                        compressed to the shared budget
#: ``mlde-over-budget``   the same reference at the budget its paper spends,
#:                        which is one plate more than this suite gives anyone
#: ``alde``               its active-learning successor, on this suite's own
#:                        landscapes and budget shape
#: ``genetic``            the Ehrlich lineage's only baseline, and the arm the
#:                        GA criterion says must be beaten or conceded
#: ``adalead``            the sequence-design lineage's default model-guided
#:                        comparator
#: ``cmaes``              the continuous relaxation the in-silico lineage runs
#: ``genetic-feasible``   ours; the control for the feasibility claim by
#:                        rejection
#: ``genetic-masked``     ours; the control for the feasibility claim by
#:                        construction -- a GA masked the same way a GFlowNet
#:                        policy is
#: ``mlde+earlyfit``      ours; the supervised reference at a training size a
#:                        constrained screen can actually return, which is the
#:                        only configuration of it that ever fits there
#: =====================  ===================================================
#:
#: The four ``genetic+`` arms are a **ladder on one representative baseline**
#: rather than a silent default on all of them. Each rung adds exactly one thing
#: to the rung above it, so the table reads as a decomposition:
#:
#: ===================  =====================================================
#: arm                  what it adds
#: ===================  =====================================================
#: ``genetic``          nothing; the model does not exist
#: ``genetic+screen``   the model filters the pool; the search stays blind
#: ``genetic+search``   the sampler optimises against the model
#: ``genetic+distinct`` the plate is filled with distinct designs
#: ===================  =====================================================
#:
#: ``random+screen`` is the same first rung on the floor, which is what says
#: whether a screen helps at all or only helps a method that was already
#: searching. A silent deep ensemble on every arm would answer none of these
#: questions and would make every one of them a hybrid.
#:
#: ``mlde+earlyfit`` wears the same ``+`` for the other reason it is used here,
#: the one ``cmaes+dp`` carries: not a rung added to a pipeline but a parameter
#: of that pipeline replaced by one of ours. It is a *smaller* method than the
#: arm it sits beside rather than a larger one, and the mark is there for the
#: same purpose either way -- so nobody reads the row as the published method.
#:
#: Note what ``genetic+screen`` and ``alde`` are *not*: the same arm. They share
#: a surrogate over a library pool and differ in the sampler, the acquisition
#: rule and the ensemble's estimator -- and only one of them is a pipeline
#: somebody published.
BASELINES: dict[str, Methodology] = {
    "random": classical(_random),
    "hill-climb": classical(_hill_climb),
    # Site saturation, one site at a time, fixing as it goes. Bare and
    # plate-pooled: the walk proposes exactly the designs its protocol names, so
    # there is nothing for a screen to screen and no pool to widen.
    "single-step": classical(_single_step),
    "recomb": classical(_recomb),
    "genetic": classical(_genetic),
    "genetic-feasible": classical(_feasible_genetic),
    "genetic-masked": classical(_masked_genetic),
    "cmaes": classical(_cmaes),
    # AdaLead's model is inside its own rollout, so the campaign hands it none
    # and its pool is one plate: what it proposes has already been screened and
    # ranked by the arm itself.
    "adalead": classical(_adalead),
    # MLDE's pool is its library and it excludes measured designs internally --
    # its published protocol, not a favour from the harness. Shrinking it to a
    # plate would leave an arm called MLDE that is not MLDE.
    "mlde": classical(_mlde, pool_size=DEFAULT_POOL),
    # The same method at the budget its own paper spends: 384 screened plus a
    # designed plate, which is 480 against everyone else's 384. It is here
    # *because* it does not fit -- an arm compressed to a quarter of its training
    # set is a weaker arm, and a headline resting on the compressed row alone
    # would be resting on a comparator we had shortened. Named for the fact
    # rather than for the number: one extra plate is over budget on every task in
    # the suite, while "480" is only the resolved total where the plate is 96.
    #
    # Two things a reader of its row has to hold on to. The stored `protocol`
    # field is the task's, so it names 384 while this arm spent 480 -- the gap
    # shows as `oracle_calls`, and `extra_rounds` in the arm's own parameters
    # says where it came from. And on a re-anchoring task the extra plate is
    # extra reach as well as extra assays, so this arm's regret is taken against
    # an attainable bound audited for a shorter campaign; read its `best` rather
    # than its `regret` where the two disagree.
    "mlde-over-budget": classical(_mlde_as_published, pool_size=DEFAULT_POOL, extra_rounds=1),
    # Ours, at everybody's budget. The two arms above cannot fit on a landscape
    # whose wells mostly return nothing -- their handover counts usable
    # measurements, and at a feasible share of 0.053 a 384-assay campaign buys
    # about 20 of them against training sizes of 96 and 384. This one trains on
    # eight, which three screening plates return on 98.7% of seeds, so the
    # ensemble gets to design at least one plate and the row says something about
    # the method rather than about a random screen wearing its name.
    #
    # The `+` is the same disclosure `cmaes+dp` carries: a published pipeline
    # with a component of ours in it is not that pipeline. What is ours here is
    # the training size, which is derived from this suite's own measured
    # feasibility and appears nowhere in Wittmann et al. -- so this row must
    # never be quoted as MLDE, and its pair with `mlde` is what separates "a
    # constrained space suits this method badly" from "this method never fitted".
    "mlde+earlyfit": classical(_mlde_adapted, pool_size=DEFAULT_POOL),
    # ALDE: the same library screen, and then the three things its own authors
    # name as what it adds over MLDE -- rounds, an uncertainty-bearing surrogate,
    # and a non-greedy rule to read that uncertainty with. Their bench
    # configuration is one-hot encodings, a five-member DNN ensemble with
    # bootstrapping, and Thompson sampling; the encoding and the width are
    # `DeepEnsemble`'s own, the other two are named here. Rounds come free from
    # the campaign, which is what a campaign is.
    "alde": classical(
        _random,
        surrogate=True,
        bootstrap=True,
        acquisition=lambda seed: Thompson(seed=seed),
        pool_size=DEFAULT_POOL,
    ),
    # Ablations. Each keeps the library pool because a screen with nothing to
    # screen is not a screen: at a pool of one plate the model would rank 96
    # candidates into 96 wells and change nothing.
    "random+screen": classical(_random, surrogate=True, pool_size=DEFAULT_POOL),
    "genetic+screen": classical(_genetic, surrogate=True, pool_size=DEFAULT_POOL),
    "genetic+search": classical(
        _genetic, surrogate=True, proxy_access=True, pool_size=DEFAULT_POOL
    ),
    # The plate rule rather than the model: same algorithm, same population,
    # same blindness, and 96 distinct designs instead of 96 proposals. It cannot
    # be recovered from `genetic`'s ledger, because dropping a repeat changes
    # which design takes that well and the two campaigns diverge from round one.
    "genetic+distinct": classical(_genetic, distinct_batch=True),
}

#: GFlowNet objectives, each behind the same interface. Comparing them is a
#: configuration change rather than a rewrite, which is the point of the seam.
#: Decoder variants of CMA-ES, kept out of `BASELINES` on purpose.
#:
#: The reported arm decodes greedily because that is the adaptation a
#: practitioner would write. This one decodes exactly, which is stronger than
#: the literature specifies, so a table containing both would be reporting our
#: engineering under a citation to Hansen. It stays here until the comparison
#: between the two has been made and can be described as what it is. Named
#: with the ``+`` the ablation rungs use, because it is the same kind of thing:
#: a published pipeline plus a component we added, which is not that pipeline.
DECODER_STUDY: dict[str, Methodology] = {
    "cmaes+dp": classical(_cmaes_exact),
}


OBJECTIVES: dict[str, Methodology] = {
    "gfn-tb": gflownet(TrajectoryBalance()),
    "gfn-contrastive": gflownet(ContrastiveBalance(prune_threshold=0.1)),
    "genetic-gfn": genetic_gflownet(TrajectoryBalance()),
}


def flow_objectives() -> dict[str, Methodology]:
    """The detailed-balance family, which needs a policy with a flow head.

    Separate from `OBJECTIVES` because they require ``learn_flow`` and
    would raise rather than silently degrade if handed a policy without one.

    Returns:
        Methodologies by name.
    """
    from evogfn.algorithms.gflownet.flow_objectives import (  # noqa: PLC0415
        DEFAULT_LAMBDA,
        DetailedBalance,
        ForwardLookingDetailedBalance,
        SubTrajectoryBalance,
    )

    return {
        "gfn-db": gflownet(DetailedBalance(), learn_flow=True),
        # `lam` named from its own module's constant rather than repeated as a
        # literal here. The selection phase scans this axis and defaults to the
        # same constant, so the arm the headline runs and the centre of the scan
        # are the same value by construction; two 0.9s could drift apart and the
        # resulting scan would report a curve the shipped arm is not on.
        "gfn-subtb": gflownet(SubTrajectoryBalance(lam=DEFAULT_LAMBDA), learn_flow=True),
        "gfn-fldb": gflownet(ForwardLookingDetailedBalance(), learn_flow=True),
    }


#: The GFlowNet settings this project inherited rather than chose, and the
#: values to measure them at: ``steps`` from what runs in acceptable time,
#: ``beta`` from Jain et al. (2022), ``mix`` from Kim et al.'s (2024) offline
#: ratio. A headline comparison against baselines tuned to their own papers, run
#: at settings nobody tuned, measures the settings -- so what this grid is for is
#: establishing that the reported configuration is not a bad one, and saying by
#: how much it could be beaten.
#:
#: Values bracket each default above and below rather than extending in one
#: direction, so a monotone column is legible as "the grid is too narrow" rather
#: than being mistaken for an optimum.
SENSITIVITY_GRID: dict[str, tuple[float, ...]] = {
    "steps": (100.0, float(DEFAULT_TRAINING_STEPS), 900.0),
    "beta": (1.0, DEFAULT_BETA, 10.0),
    "mix": (0.0, DEFAULT_MIX, 1.0),
}


def sensitivity() -> dict[str, Methodology]:
    """One arm per hyperparameter value, varying one axis at a time.

    One at a time rather than a full grid: the full cross of
    `SENSITIVITY_GRID` is 27 arms where this is 9, and the question being asked
    is whether any single setting is badly chosen -- not where the joint optimum
    sits, which this benchmark has nothing like the seed count to locate.

    The arm at each default duplicates a configuration the objectives
    diagnostic already runs -- ``steps-300`` and ``beta-3`` are ``gfn-tb``,
    ``mix-0.5`` is ``genetic-gfn`` -- and is re-run anyway. It costs two extra
    arms on the cheapest landscape in the suite, and it buys an axis that reads
    as a curve on its own terms rather than as two measured points plus a
    cross-reference to another tier's table.

    Returns:
        Methodologies by name, named ``<axis>-<value>`` so a report groups them.
    """
    arms: dict[str, Methodology] = {}
    for axis, values in SENSITIVITY_GRID.items():
        for value in values:
            name = f"{axis}-{value:g}"
            if axis == "steps":
                arms[name] = gflownet(TrajectoryBalance(), steps=int(value))
            elif axis == "beta":
                arms[name] = gflownet(TrajectoryBalance(), beta=value)
            else:
                # `mix` belongs to the genetic teacher, so it varies the arm
                # that has one. Its endpoints are the method's own limits: at 0
                # this is plain `gfn-tb` and at 1 the policy never samples for
                # itself, which brackets the claim that the hybrid beats both.
                arms[name] = genetic_gflownet(TrajectoryBalance(), mix=value)
    return arms


def anchor_arms() -> dict[str, Methodology]:
    """The three arms of the anchor mechanism study.

    Two of the names are taken out of the tables the suite already runs rather
    than rebuilt here, and that is deliberate: the store keys on ``(task,
    arm)``, so an arm re-declared under the same name from a second expression
    would be the same cell only for as long as the two expressions agreed.
    Looked up, they cannot disagree, and a cell the suite has already paid for
    stays paid for.

    ``gfn-tb-rebuilt`` differs from ``gfn-tb`` in exactly one argument.
    Everything else -- the objective, the reward exponent, the gradient steps
    per round, the trunk width, the candidate pool -- is left at the default the
    other arm also takes, so the two cannot drift apart in anything that costs
    compute. That matters more here than anywhere else in the suite: every round
    retrains regardless, so both arms spend ``rounds * steps`` gradient steps
    and ``rounds * steps * TRAINING_BATCH`` proxy evaluations, and the rebuilt
    arm is therefore losing or winning on transfer rather than on budget. An arm
    that started each ball from nothing *and* got fewer steps to recover with
    would measure the two together.

    Plain trajectory balance rather than the genetic hybrid, because the hybrid
    would move two mechanisms at once: its teacher is re-founded at every anchor
    in both arms by design -- a population bred around the old parent can sit
    wholly outside the new mutation budget -- so a rebuilt Genetic-GFN would
    differ from a carried one in the policy while both had already discarded the
    teacher.

    Returns:
        Methodologies by name.
    """
    return {
        "gfn-tb": OBJECTIVES["gfn-tb"],
        "gfn-tb-rebuilt": gflownet(TrajectoryBalance(), carry_policy=False),
        "genetic": BASELINES["genetic"],
    }


#: Where the selection phase records the configuration this project ships.
#:
#: The *same relative path* ``experiments/run_suite.py``'s ``selected_gflownet``
#: reads, and the same
#: string rather than a path resolved from ``__file__``, deliberately. A library
#: resolving a data file against the working directory is poor practice in
#: general; here the property that has to hold is stronger than that objection.
#: The ladder's base rung and the headline table's GFlowNet arm must be *the same
#: arm*, and two readers that resolve the same file differently -- one finding it
#: from the repository root, one not finding it from somewhere else -- break
#: exactly that. Resolving it identically to the only other reader is what makes
#: disagreement impossible rather than unlikely.
SELECTED_CONFIGURATION = Path("results/selected.json")

#: Every axis the selection moves. Absent and null are different claims -- null
#: says the chosen objective reads no such knob, absent says the file predates
#: the stage that chose it -- so a missing key raises rather than defaulting, on
#: the same reasoning as the reader in ``experiments/run_suite.py``: a ladder
#: built from a file missing an axis would run the default on it and report the
#: result as the shipped configuration.
_SELECTED_AXES = ("objective", "arm", "beta", "steps", "lam", "mix", "hidden_dim")


def _parameter_count(
    hidden_dim: int,
    *,
    sequence_length: int,
    vocab_size: int,
    learn_flow: bool,
    anchor_conditioned: bool,
) -> int:
    """Parameters a policy of this width carries on this shape, by construction.

    Built and counted rather than computed from a formula. The closed forms are
    easy to write and easy to get subtly wrong -- an embedding table, a bias, a
    flow head -- and a control mis-sized by a formula error is a control that
    reports capacity as conditioning.

    Args:
        hidden_dim: Width of the trunk.
        sequence_length: Positions the policy reads and writes. Half of what
            fixes both the trunk's input width and the action head's output
            width, so a count is a statement about *one* shape.
        vocab_size: Tokens in the alphabet, the other half.
        learn_flow: Whether the policy carries a flow head.
        anchor_conditioned: Whether the anchor is fed alongside the state.

    Returns:
        The number of learnable parameters. The anchor itself is a buffer rather
        than a parameter and so is correctly not counted: it is data the policy
        reads, not capacity it has.
    """
    shape = {
        "n_tokens": vocab_size,
        "sequence_length": sequence_length,
        "n_actions": sequence_length * vocab_size + 1,
        "hidden_dim": hidden_dim,
        "learn_flow": learn_flow,
        "seed": 0,
    }
    policy = (
        AnchorConditionedPolicy(anchor=np.zeros(sequence_length, dtype=np.int64), **shape)  # type: ignore[arg-type]
        if anchor_conditioned
        else SequencePolicy(**shape)  # type: ignore[arg-type]
    )
    return sum(int(p.numel()) for p in policy.parameters())


@dataclass(frozen=True, slots=True)
class CapacityMatch:
    """A plain trunk width, and how near it came to the size it had to match.

    The residual is carried rather than recomputed by whoever wants it, because
    the *sign* of it is the safety property the whole control rests on and a
    reader has no other way to check it. A row of a headline table that says
    "conditioning helped" is only attributable if the plain policy beside it was
    not the smaller network; a residual printed with the row is that claim,
    stated as a measurement, at the shape it was measured on.

    Attributes:
        hidden_dim: The plain trunk width this resolved to.
        parameters: What the plain policy carries at that width.
        target: What the anchor-conditioned policy carries, and therefore what
            was being matched.
    """

    hidden_dim: int
    parameters: int
    target: int

    @property
    def residual(self) -> float:
        """Signed share by which the control over- or under-shoots its target.

        Returns:
            ``(parameters - target) / target``. Positive is the safe direction
            and is what the search guarantees; negative would mean the control
            is the smaller network, which manufactures the effect it exists to
            rule out.
        """
        return (self.parameters - self.target) / self.target

    def __repr__(self) -> str:
        """The width, and the residual a reader has to see beside it."""
        return (
            f"hidden_dim={self.hidden_dim} at {self.parameters} parameters "
            f"against {self.target} ({self.residual:+.2%})"
        )


@cache
def matched_capacity(
    hidden_dim: int, *, sequence_length: int, vocab_size: int, learn_flow: bool
) -> CapacityMatch:
    """Trunk width at which a plain policy carries a conditioned one's capacity.

    Anchor conditioning widens the trunk's *input* -- the state embedding, the
    anchor embedding and one difference indicator per position -- so an
    anchor-conditioned policy has more parameters than a plain one of the same
    trunk width. At the diagnostic shape and ``hidden_dim = 64`` that is 179,715
    against 112,131, a 60% difference in capacity. Any gain the ``+anchor`` rung
    shows is confounded with that until a plain policy of matching size has been
    measured, and no amount of care about the other axes removes the confound.

    **No width matches exactly**, since the counts are quadratics in the width
    that do not agree at any integer. The rule is therefore the *narrowest plain
    trunk that is not smaller* than the conditioned one, which fixes the residual
    on the safe side: a control that slightly over-resources the comparator can
    only understate conditioning's effect, while one that under-resources it
    manufactures the effect the arm exists to rule out.

    The shape is an argument, and that is the point
    -----------------------------------------------

    A parameter count depends on the sequence length and the alphabet, so "the
    plain policy that matches the conditioned one" is a statement about *one*
    sizing and is false at every other. Resolved once at a fixed shape -- which
    is what this used to do -- the control was correct on the landscape the rungs
    were first run on and wrong everywhere the tier subsequently went: at the
    shipped width of 64 and a flow head, matching the 32-position diagnostic gives
    width 101, and that same 101 carries **1.63% fewer** parameters than the
    conditioned arm at the 64-position tasks and 20.5% more at the four-site
    empirical ones. The shortfall is the fatal direction: an under-resourced
    control cannot rule out capacity, so the row would have flattered conditioning
    on exactly the tasks the claim is drawn from. Per shape, the residuals are
    +0.13% at 32 positions (width 101), +0.37% at 64 (width 103) and +1.07% at
    four (width 88) -- all above zero, which is the property that has to hold.

    Searched rather than solved, and every candidate is built and counted, so a
    change to the architecture moves this instead of leaving it stating a
    residual that stopped being true.

    Args:
        hidden_dim: Width of the anchor-conditioned trunk to match.
        sequence_length: Positions in the task's designs.
        vocab_size: Tokens in the task's alphabet.
        learn_flow: Whether both policies carry a flow head. The objective
            decides this, so it is read from the base arm rather than assumed:
            matched at the wrong setting the two counts differ by a head.

    Returns:
        The width and what it achieved, so the residual travels with the number
        instead of having to be re-derived by anything that reports it.
    """
    shape = {"sequence_length": sequence_length, "vocab_size": vocab_size, "learn_flow": learn_flow}
    target = _parameter_count(hidden_dim, **shape, anchor_conditioned=True)  # type: ignore[arg-type]
    low = hidden_dim
    high = max(hidden_dim, 1) * 2
    while _parameter_count(high, **shape, anchor_conditioned=False) < target:  # type: ignore[arg-type]
        low, high = high, high * 2
    # Invariant: `low` is short of the target and `high` is not, so the answer is
    # the first width above `low`, and the loop cannot return an untested width.
    while high - low > 1:
        middle = (low + high) // 2
        if _parameter_count(middle, **shape, anchor_conditioned=False) < target:  # type: ignore[arg-type]
            low = middle
        else:
            high = middle
    return CapacityMatch(
        hidden_dim=high,
        parameters=_parameter_count(high, **shape, anchor_conditioned=False),  # type: ignore[arg-type]
        target=target,
    )


def matched_capacity_for(task: Task, hidden_dim: int, *, learn_flow: bool) -> CapacityMatch:
    """The capacity match at the shape this task's policies are built on.

    The one place the task's shape is read for this purpose, so the width a
    campaign trains at and the width its record reports come from the same call
    rather than from two that could disagree.

    Args:
        task: The task the control will run on.
        hidden_dim: Width of the conditioned trunk being matched.
        learn_flow: Whether both policies carry a flow head.

    Returns:
        The match.
    """
    landscape = task.landscape()
    return matched_capacity(
        hidden_dim,
        sequence_length=int(landscape.sequence_length),
        vocab_size=int(landscape.alphabet.size),
        learn_flow=learn_flow,
    )


@dataclass(frozen=True)
class LadderBase:
    """The rung every other rung of `variant_arms` is one step above.

    Carries the base arm *and* the parts it was built from, which is redundant
    only in appearance. The arm is what the base rung runs, and it is built
    exactly as the headline table builds its GFlowNet arm so the two share a
    store cell; the parts are what the other rungs are built from, since a
    methodology is a closure and a closure cannot be asked what objective it
    holds. Rungs assembled from these fields differ from the base in the flag
    they are named for and in nothing else, and a test asserts that rather than
    the reader having to trust it.

    Attributes:
        name: The base arm's name, and therefore the prefix every rung takes.
        arm: The base rung itself.
        objective: The training objective the rungs rebuild with.
        learn_flow: Whether the objective needs a flow head.
        beta: Reward exponent.
        steps: Gradient steps per round.
        hidden_dim: Width of the policy trunk.
    """

    name: str
    arm: Methodology
    objective: GFlowNetObjective
    learn_flow: bool
    beta: float
    steps: int
    hidden_dim: int

    def rung(self, **flags: bool | int) -> Methodology:
        """One rung: the base configuration with the named mechanism turned on.

        Args:
            **flags: Arguments of [gflownet][evogfn.benchmark.methods.gflownet]
                that this rung sets -- the two mechanism flags, or the widened
                trunk of the capacity control.

        Returns:
            A methodology.
        """
        settings: dict[str, bool | int | float] = {
            "beta": self.beta,
            "steps": self.steps,
            "learn_flow": self.learn_flow,
            "hidden_dim": self.hidden_dim,
        }
        settings.update(flags)
        return gflownet(self.objective, **settings)  # type: ignore[arg-type]


def _objective_instance(name: str, lam: float | None) -> tuple[GFlowNetObjective, bool]:
    """The objective a selection record names, and whether it needs a flow head.

    Args:
        name: An objective from `OBJECTIVES` or `flow_objectives`.
        lam: Sub-trajectory balance's length weighting, or ``None`` for its own
            default.

    Returns:
        The objective instance and its ``learn_flow`` requirement.

    Raises:
        KeyError: If the name is not a known objective.
        ValueError: If the objective breeds. Genetic-GFN is built by
            [genetic_gflownet][evogfn.benchmark.methods.genetic_gflownet], which
            takes neither mechanism flag, so neither rung of this ladder is
            definable for it -- and building the ladder on plain trajectory
            balance instead, silently, would report a mechanism study of a
            configuration nobody selected.
    """
    from evogfn.algorithms.gflownet.flow_objectives import (  # noqa: PLC0415
        DEFAULT_LAMBDA,
        DetailedBalance,
        ForwardLookingDetailedBalance,
        SubTrajectoryBalance,
    )

    if name == "gfn-tb":
        return TrajectoryBalance(), False
    if name == "gfn-contrastive":
        return ContrastiveBalance(prune_threshold=0.1), False
    if name == "gfn-db":
        return DetailedBalance(), True
    if name == "gfn-subtb":
        return SubTrajectoryBalance(lam=DEFAULT_LAMBDA if lam is None else lam), True
    if name == "gfn-fldb":
        return ForwardLookingDetailedBalance(), True
    if name == "genetic-gfn":
        raise ValueError(
            "the selected configuration is Genetic-GFN, which breeds and is built by "
            "genetic_gflownet(); neither ladder mechanism is defined for it, so the "
            "ladder cannot be built on the shipped configuration"
        )
    raise KeyError(f"unknown objective {name!r}")


def shipped_base() -> LadderBase:
    """The configuration this project ships, as the ladder's base rung.

    Read from `SELECTED_CONFIGURATION` rather than restated, and built through
    the selection phase's own
    `_build_objective`, so the base
    rung and the arm ``experiments/run_suite.py`` puts in the headline table are
    the same arm resolved from the same file by the same builder. Restating the
    chosen settings here instead -- ``gfn-subtb`` at beta 0.1, 300 steps, lambda
    0.9, width 64 -- would be one literal per axis that nothing would ever check,
    and the ladder would go on claiming to study the shipped configuration for as
    long as it took a selection to move.

    Imported inside the function because `selection` imports this module: the
    cycle is real and the lazy import is what breaks it. It costs nothing in
    staleness terms, ``benchmark.selection`` already being one of the entry
    points every record's fingerprint is expanded from.

    Returns:
        The base rung. With no selection recorded this is ``gfn-tb`` -- not as a
        placeholder but because that is what the headline table reports in that
        state: ``methods_for`` falls back to the untuned arms, so a ladder based
        on anything else would be the one describing a configuration nobody runs.

    Raises:
        ValueError: If the file records a selection that stopped partway, or one
            whose name and settings describe different configurations, or one
            whose objective has no ladder. Raised rather than falling back to the
            untuned defaults, which would silently answer a question about
            trajectory balance under the name of the shipped method.
    """
    from evogfn.benchmark.selection import Configuration  # noqa: PLC0415

    if not SELECTED_CONFIGURATION.exists():
        return LadderBase(
            name="gfn-tb",
            # Looked up rather than rebuilt, so the base rung is the identical
            # object -- and therefore the identical store cell -- that the
            # objectives diagnostic has already paid a hundred seeds for.
            arm=OBJECTIVES["gfn-tb"],
            objective=TrajectoryBalance(),
            learn_flow=False,
            beta=DEFAULT_BETA,
            steps=DEFAULT_TRAINING_STEPS,
            hidden_dim=DEFAULT_HIDDEN_DIM,
        )

    choice = json.loads(SELECTED_CONFIGURATION.read_text())
    if missing := [key for key in _SELECTED_AXES if key not in choice]:
        raise ValueError(
            f"{SELECTED_CONFIGURATION} is missing {', '.join(missing)}, so the selection it "
            f"records is unfinished; run experiments/select_configuration.py to completion, "
            f"or delete the file to build the ladder on the untuned defaults"
        )
    configuration = Configuration(
        objective=str(choice["objective"]),
        beta=float(choice["beta"]),
        steps=int(choice["steps"]),
        hidden_dim=int(choice["hidden_dim"]),
        lam=None if choice["lam"] is None else float(choice["lam"]),
        mix=None if choice["mix"] is None else float(choice["mix"]),
    )
    # The recorded name is not decoration: it is the store key every confirmation
    # campaign was written under, so a file whose settings and name disagree
    # would build one configuration and read another's results.
    if configuration.name != str(choice["arm"]):
        raise ValueError(
            f"{SELECTED_CONFIGURATION} names arm {choice['arm']!r} but its settings build "
            f"{configuration.name!r}; one of the two describes a campaign that never happened"
        )
    objective, learn_flow = _objective_instance(configuration.objective, configuration.lam)
    return LadderBase(
        name=configuration.name,
        arm=configuration.build(),
        objective=objective,
        learn_flow=learn_flow,
        beta=configuration.beta,
        steps=configuration.steps,
        hidden_dim=configuration.hidden_dim,
    )


def variant_arms(base: LadderBase | None = None) -> dict[str, Methodology]:
    """The GFlowNet ladder: two mechanisms, each one rung above what ships.

    The same shape as the ``genetic+`` ladder in `BASELINES` and for the same
    reason. Each rung adds exactly one thing to a base arm that is *resolved*
    rather than re-declared, so the store's ``(task, arm)`` cell for the base is
    shared with every table that already paid for it, and the comparison is
    against the identical configuration by construction rather than by two
    expressions agreeing. Writing ``B`` for the base arm's name:

    ====================  =======================================================
    arm                   what it adds
    ====================  =======================================================
    ``B``                 nothing; feasibility is masked at every intermediate
    ``B+terminal``        feasibility is required of the design, not of the order
    ``B+anchor``          the policy is told which parent it is evolving from
    ``B+terminal+anchor`` both
    ``B+wide``            neither; the capacity ``+anchor`` also brought
    ====================  =======================================================

    The base is what this project ships, not ``gfn-tb``
    ---------------------------------------------------

    It used to be ``gfn-tb``, and that was wrong in a way this repository can
    demonstrate rather than merely suspect. The shipped configuration is
    sub-trajectory balance, chosen by a pre-declared rule over 3,100 campaigns;
    and the selection study measured the reward exponent on trajectory balance
    and then again on sub-trajectory balance and **the curve reversed direction
    between them**. An effect measured on one objective is therefore not known to
    transfer to the other here, which makes a ladder built on trajectory balance
    a study of an arm nobody runs -- and makes the claim that its both-on rung
    "is the configuration the method would ship if both rungs win" a non sequitur,
    since the configuration that ships is not on the ladder at all.

    Whether to take the base as a parameter or read the selection here is a real
    question and it is decided by *which drift is possible*, not by which
    dependency reads better. A module that reaches into a results file is coupled
    to it; a parameter with a restated default is coupled to nothing and drifts
    the moment a selection moves, which is precisely the failure being fixed. So
    the base is a parameter -- a caller with a configuration in hand passes it,
    and tests can hand it one without a file on disk -- and the *default* is the
    recorded selection, read through `shipped_base` from the same file and by the
    same builder ``experiments/run_suite.py`` uses. The one caller in the suite
    passes nothing, so the default is what the tier runs, and it cannot disagree
    with the headline table about what ships.

    Why there are four rungs and not three
    --------------------------------------

    The both-on rung is not decoration. Unlike the ``genetic+`` ladder, whose
    rungs are nested -- each contains the one above it -- these two mechanisms
    are *orthogonal*: one changes the graph the policy walks, the other changes
    what the policy reads at each state. Three arms would report two main effects
    and leave the interaction unmeasured, which is exactly where the interesting
    answer could be. Conditioning on the anchor is what lets a policy tell an
    inherited residue from a substitution, and terminal feasibility is what makes
    the substitution orders unconstrained; a policy that can see which positions
    are its own is the one with something to do with that freedom. It is also the
    configuration the method would ship if both rungs win -- which is a statement
    about this ladder only now that its base is the shipped arm -- so it has to
    have been run rather than inferred by addition.

    Why there is a fifth arm that is not a rung
    -------------------------------------------

    ``B+wide`` is a control, not a step. Anchor conditioning widens the
    trunk's input, so ``+anchor`` is *both* a conditioning change and a capacity
    change, and a win there is unattributable until a plain policy of the same
    size has been measured. Its width is resolved by `matched_capacity`, which
    builds candidates and counts them, **per task**: at the shipped width of 64
    that is 101 on a 32-position landscape, 103 on a 64-position one and 88 on
    the four-site empirical ones, each over its target rather than under -- by
    0.13%, 0.37% and 1.07% -- since a control given slightly more capacity can
    only understate the mechanism it exists to isolate.

    Per task and not once, because the tier runs five tasks of three shapes and a
    parameter count is a statement about one of them. Sized at the diagnostic
    shape alone, this arm carried 1.63% *fewer* parameters than ``+anchor`` on the
    64-position tasks -- an under-resourced control, which cannot rule out
    capacity and so flatters conditioning on precisely the tasks the claim is
    drawn from. The width therefore does not appear in the arm's name or in its
    static settings; it is resolved when the task is known and written onto every
    record, beside the residual it achieved there.

    It carries the base arm's prefix for the same reason the rungs do, and it is
    the arm that needed it most. Its *width* is derived from the base
    configuration -- and now from the task as well -- so under the bare name it
    used to have, a stored record from one selection and a record sized against a
    different selection were the same ``(task, arm)`` cell and indistinguishable
    in the store. The rungs were rebuilt to stand on the recorded selection
    precisely to close that drift; a control left outside the naming scheme kept
    it open for the one arm whose number is not on the ladder anywhere else. The
    task half of the drift needs no name: the store keys by ``(task, arm)``
    already, so two shapes are already two cells.

    Kept out of `OBJECTIVES` deliberately. That mapping is read as a set of
    *training objectives* -- the configuration sweep resolves each of its keys to
    an objective instance by name -- and none of these varies the objective; what
    differs is the graph one of them walks, the input another one reads and the
    width of a third.

    Two rungs are not measurable on every task
    ------------------------------------------

    ``+terminal`` has something to defer only where something constrains
    construction. On a landscape with no transition matrix it and the base are the
    identical graph, and ``+terminal+anchor`` and ``+anchor`` likewise. Those rows
    are reproduced rather than run -- see `reproduced_rungs`, which decides it from
    the environment and not from a list of tasks. This function still builds all
    five arms everywhere, because *what the arm is* does not depend on the task;
    what depends on the task is whether running it would measure anything.

    No rung is available to a classical arm, and that is a property of the
    methods rather than of this module: conditioning needs a policy to condition,
    and nothing in `BASELINES` has one.

    Args:
        base: The rung everything else is one step above, or ``None`` for the
            recorded selection.

    Returns:
        Methodologies by name.
    """
    base = shipped_base() if base is None else base
    return {
        base.name: base.arm,
        f"{base.name}+terminal": base.rung(terminal_feasibility=True),
        f"{base.name}+anchor": base.rung(anchor_conditioned=True),
        f"{base.name}+terminal+anchor": base.rung(
            terminal_feasibility=True, anchor_conditioned=True
        ),
        f"{base.name}+wide": base.rung(match_anchor_capacity=True),
        # The decomposition the headline claim rests on, in two steps. Neither
        # rung trains: `steps=0` leaves the policy at its random initialisation,
        # so both draw from the *masked construction graph* and differ from the
        # base arm in learning alone.
        #
        # `+untrained` screens nothing -- `PLATE_POOL` fills the plate directly
        # -- so it is masking and nothing else, and it spends zero surrogate
        # calls. It is the arm behind the observation that an untrained masked
        # draw already beats published baselines' final answers.
        #
        # `+untrained@pN` adds the screen at pool `N`. The sweep exists because
        # the surrogate is **not** the campaign's budget: the oracle budget is
        # the plate and is identical on every one of these rungs, so screening
        # more is more compute against a free model rather than a larger budget.
        # A single pool would leave "would more screening have closed the gap?"
        # unanswered and unanswerable from the table; three say where it
        # saturates, which is the honest form of the control.
        f"{base.name}+untrained": base.rung(steps=0, pool_size=PLATE_POOL),
        f"{base.name}+untrained@p2048": base.rung(steps=0, pool_size=2048),
        f"{base.name}+untrained@p8192": base.rung(steps=0, pool_size=8192),
        f"{base.name}+untrained@p32768": base.rung(steps=0, pool_size=32768),
    }


def reproduced_rungs(base: LadderBase | None = None, *, task: Task) -> dict[str, str]:
    """Rungs whose campaign on this task *is* another rung's, and which one.

    ``+terminal`` defers the feasibility rule from every intermediate state to the
    stop action. Where nothing constrains construction there is no rule to defer,
    so the two environments describe the identical graph and the rung reproduces
    the base bit for bit; ``+terminal+anchor`` stands in the same relation to
    ``+anchor``. Running them would spend the campaigns twice and put two pairs of
    identical rows in a headline table, which a reader may quote as "we tested the
    mechanism here and it made no difference" -- a different and false claim from
    "there was nothing here to test".

    So they are reproduced rather than run, and **nothing is stored**. A stored
    copy is indistinguishable from a measurement the moment it is written: it
    lands in the same ``(task, arm)`` cell shape, carries the same fingerprint and
    the same seeds, and every reader of the store -- the instance analysis, the
    statistics, a script somebody writes next year -- would count it as an
    independent campaign. Marking it would only move the problem, since the mark
    then has to be honoured by every one of those readers and the one that forgets
    is the one that double-counts. Reproducing at report time keeps the store a
    record of what was *measured*, and puts the copy where the claim is made,
    where it can be labelled in the same line a reader takes the number from.

    Derived from the environment rather than from a list of task names -- see
    `constrains_construction` -- so a task that gains a transition matrix stops
    being reproduced and starts being run, with no edit here.

    Args:
        base: The ladder's base rung, or ``None`` for the recorded selection.
        task: The task the rungs would run on.

    Returns:
        Rung name to the arm whose campaign it repeats, or empty where the
        mechanism has something to act on and every rung must be run.
    """
    if constrains_construction(task):
        return {}
    base = shipped_base() if base is None else base
    return {
        f"{base.name}+terminal": base.name,
        f"{base.name}+terminal+anchor": f"{base.name}+anchor",
    }


def default_methodologies() -> dict[str, Methodology]:
    """Every methodology, for a full sweep.

    Returns:
        Baselines, then GFlowNet objectives, in a stable order so a report
        reads the same way each run.
    """
    return {**BASELINES, **OBJECTIVES, **flow_objectives()}


def rng_for(seed: int) -> np.random.Generator:
    """A generator for anything a methodology needs beyond its components."""
    return np.random.default_rng(seed)
