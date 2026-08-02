"""A GFlowNet behind the same interface as the classical baselines.

This is what makes the comparison a config change rather than two harnesses.
The campaign asks for proposals; this sampler retrains its policy against the
current proxy and samples from it. Everything the loop knows about a genetic
algorithm it also knows about this.

Where the oracle budget does *not* go
-------------------------------------

Training happens against a
[ProxyLandscape][evogfn.surrogate.proxy.ProxyLandscape], never the oracle. A
thousand gradient steps at batch 64 is 64,000 reward evaluations; charging those
would exhaust a realistic campaign budget of a few hundred before the first
round returned. Because the proxy holds no oracle, the separation is structural
rather than a matter of remembering.

Retraining, not fine-tuning
---------------------------

The policy is retrained from its current parameters each round rather than reset.
The proxy changes between rounds -- it is refitted on strictly more data -- so
the target distribution moves, and a policy already near the previous target is
a better starting point than a fresh one. This is warm-starting, and it is why a
round costs a fraction of what training from scratch would.

The first round has no proxy to train against: nothing has been measured yet, so
the surrogate is unfitted. The policy samples from its initialisation, which
under a masked environment is close to uniform over the *feasible* set. That is
not a fallback -- it is the feasibility-by-construction property doing its work
before any model exists, and it is the fairest possible seed design.

Following the campaign's anchor
-------------------------------

A campaign re-anchors between rounds, and a sampler either says what should
happen to its own state or is rebuilt from a factory. The rebuild is correct
here -- the factory closes over the same policy, so the weights survive -- but it
is not free: it restarts the sampler's own accounting, so a ``proxy_calls`` read
off the campaign afterwards counts the last anchor's rounds rather than the
campaign's, understating the method's compute by roughly the number of times it
moved. That is a number reported next to the oracle budget, and undercounting it
would flatter exactly the method under test.
[reanchored][evogfn.algorithms.gflownet.sampler.GFlowNetSampler.reanchored] is
what makes the campaign prefer the informed path instead.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from evogfn.algorithms.base import Sampler
from evogfn.algorithms.gflownet.genetic_gfn import train_genetic_gfn
from evogfn.algorithms.gflownet.sampling import sample_trajectories
from evogfn.algorithms.gflownet.training import TrainingConfig, train_trajectory_balance
from evogfn.models.policy import AnchorConditionedPolicy

if TYPE_CHECKING:
    from evogfn.algorithms.baselines.genetic import GeneticAlgorithm
    from evogfn.algorithms.gflownet.genetic_gfn import GeneticConfig
    from evogfn.algorithms.gflownet.objectives import GFlowNetObjective
    from evogfn.core.types import Tokens
    from evogfn.env.base import SequenceEnvironment
    from evogfn.env.mutation import MutationEnvironment
    from evogfn.models.policy import SequencePolicy
    from evogfn.rewards.base import Reward
    from evogfn.surrogate.proxy import ProxyLandscape


class GFlowNetSampler(Sampler):
    """Trains a GFlowNet against a proxy and samples proportionally to it.

    Args:
        env: The construction graph. Feasibility lives here, as masks.
        policy: The policy to train, modified in place across rounds.
        proxy: The surrogate-backed landscape training optimises against.
            Never the oracle.
        reward: Transforms proxy values into log rewards.
        config: Training settings applied on every retrain.
        objective: How balance violation is measured. Defaults to trajectory
            balance.
        genetic: A genetic algorithm to use as the policy's teacher. With one,
            training runs through
            [train_genetic_gfn][evogfn.algorithms.gflownet.genetic_gfn.train_genetic_gfn]
            -- the GA recombines the best of a rank-based buffer and its
            offspring are replayed into the training batch. Kim et al. report
            this closing the gap to Mol GA on PMO, and directed evolution *is* a
            genetic algorithm, so it is the variant most likely to matter here.
        genetic_config: How much guidance to apply. Ignored without ``genetic``.
        seed: Seeds proposal sampling, independently of training.
    """

    def __init__(  # noqa: PLR0913 - the sampler is defined by its parts
        self,
        env: SequenceEnvironment,
        policy: SequencePolicy,
        *,
        proxy: ProxyLandscape,
        reward: Reward,
        config: TrainingConfig | None = None,
        objective: GFlowNetObjective | None = None,
        genetic: GeneticAlgorithm | None = None,
        genetic_config: GeneticConfig | None = None,
        seed: int = 0,
    ) -> None:
        """Store the training setup without running it."""
        super().__init__()
        self._env = env
        self._policy = policy
        self._bind_anchor(env, policy)
        self._proxy = proxy
        self._reward = reward
        self._config = config or TrainingConfig()
        self._objective = objective
        self._genetic = genetic
        self._genetic_config = genetic_config
        self._generator: torch.Generator | None = None
        self._seed = seed
        self._rounds_trained = 0
        self._proxy_calls = 0
        self._bred_designs = 0
        self._unconstructible_designs = 0

    @staticmethod
    def _bind_anchor(env: SequenceEnvironment, policy: SequencePolicy) -> None:
        """Point an anchor-conditioned policy at the graph it is about to walk.

        [AnchorConditionedPolicy][evogfn.models.policy.AnchorConditionedPolicy]
        reads the anchor as an input, so it holds one, and a held anchor can
        disagree with the environment's. The disagreement is silent -- the policy
        conditions on a parent nobody is searching from, the loss stays finite,
        and the only symptom is a worse result -- so it is made impossible here
        instead of documented as a caution. This is the one place an environment
        and a policy are married, and a campaign that moves its anchor arrives
        here again by constructing a new sampler.

        An unconditioned policy is left alone entirely, which is what keeps the
        variant an added arm rather than a change to every existing one.

        Args:
            env: The graph the sampler will walk. Only its anchor is read, and
                only when there is a policy that wants one; an environment with
                no parent -- one that builds from nothing -- is left alone.
            policy: The policy to bind.
        """
        if not isinstance(policy, AnchorConditionedPolicy):
            return
        parent = getattr(env, "parent", None)
        if parent is not None:
            policy.set_anchor(parent)

    @property
    def name(self) -> str:
        """Short label naming the objective the policy was trained under."""
        objective = type(self._objective).__name__ if self._objective else "TrajectoryBalance"
        teacher = " + GA" if self._genetic is not None else ""
        return f"GFlowNet ({objective}){teacher}"

    @property
    def rounds_trained(self) -> int:
        """How many times the policy has been retrained."""
        return self._rounds_trained

    @property
    def proxy_calls(self) -> int:
        """Reward evaluations spent on the proxy.

        Free in budget terms and expensive in compute terms. Reported so the
        trade against a baseline that does no training is visible rather than
        implied.
        """
        return self._proxy_calls

    @property
    def bred_designs(self) -> int:
        """Genetic offspring the policy was asked to construct a path to.

        A campaign total, and zero without a genetic teacher -- nothing was
        bred, so nothing could fail to be built. That is a statement about the
        configuration rather than a measurement, which is why the share below
        must never be read without this beside it.
        """
        return self._bred_designs

    @property
    def unconstructible_designs(self) -> int:
        """How many of those the policy had no way to construct.

        Each was feasible and inside the mutation budget -- ``is_reachable``
        admits it -- and yet every ordering of its mutations passes through a
        state the environment forbids, so no trajectory of this graph ends
        there and replay could return no path.
        """
        return self._unconstructible_designs

    @property
    def unconstructible_fraction(self) -> float:
        """Share of bred designs the policy could not construct.

        The feasible-but-unreachable gap measured on designs a run actually
        produced, rather than by enumerating a toy instance. It is a property of
        the landscape's constraint and not of this sampler, which is what makes
        it worth storing rather than logging.

        Returns:
            The share in ``[0, 1]``, and ``0.0`` when nothing was bred. A share
            alone cannot distinguish those two cases, so read it with
            `bred_designs`.
        """
        if not self._bred_designs:
            return 0.0
        return self._unconstructible_designs / self._bred_designs

    def reanchored(self, env: MutationEnvironment) -> GFlowNetSampler:
        """Carry the policy, the random stream and the accounting to ``env``.

        **The policy survives, and the reason is structural rather than
        convenient.** An action index below ``length * |alphabet|`` means "set
        position ``a // |alphabet|`` to token ``a % |alphabet|``", and the
        network's input is the state sequence itself. Neither the action layout
        nor the input mentions the parent, so a policy that has learned "token 1
        at position 3 earns reward" has learned something about the landscape and
        not about the anchor. What is anchored is only the **masks** -- which
        actions are legal from a state, and how much budget is left -- and those
        are read from ``env`` on every step rather than stored. So the weights
        transfer intact, and the object is shared rather than copied: training
        already mutates it in place across rounds, and the campaign's factory
        fallback closes over the same instance, so copying here would introduce a
        second set of weights that no other path has.

        **An anchor-conditioned policy mentions the parent in its input, and is
        re-bound rather than exempted.** The paragraph above says the network's
        input does not mention the parent, and for
        [AnchorConditionedPolicy][evogfn.models.policy.AnchorConditionedPolicy]
        that is false by design: the anchor is fed to it. Its *weights* are still
        anchor-free -- the anchor arrives as input, not as a parameter -- so they
        transfer for exactly the same reason, and what has to move is the input.
        Constructing the moved sampler is what moves it, so a policy is never
        walking one graph while conditioned on another.

        **``log Z`` is the one anchor-relative parameter, and it is carried
        deliberately.** It estimates the total flow through the DAG -- the
        partition function of the reward over the reachable set -- and the
        reachable set is precisely what moved, so the carried value is stale by
        construction. It is kept anyway: ``log Z`` enters the loss and never the
        forward distribution, so a stale value biases nothing that gets proposed;
        it is a single scalar carrying the learning rate an order of magnitude
        above the policy's, which is what re-converges it inside the next
        retrain; and the reward *scale* largely transfers across a move, so the
        stale estimate is a better starting point than a reset. What it must not
        be is *read* as the new ball's partition function before that retrain.

        **The accounting survives, which is the failure this hook exists for.**
        ``proxy_calls`` is a campaign total: restarting it at each anchor would
        report the last anchor's rounds as the arm's whole compute, and it is a
        printed column beside the oracle budget, so the undercount would land in
        the results table as a cost this method did not pay. The bred and
        unconstructible counts are carried for a sharper version of the same
        reason: the reachability gap is a property of the *ball being searched*,
        so a move is exactly when it changes, and a count restarted at each
        anchor would report the last ball's gap as the campaign's. ``rounds_trained``
        matters twice over -- it is the reported round count, and it is the
        offset that gives each retrain its own seed (``config.seed +
        rounds_trained``). Reset it and the first round after a move replays the
        trajectories of round zero, so a round of training costs full compute and
        teaches the policy nothing it has not already seen. The torch generator
        is carried for the same reason and not re-seeded: one stream through the
        campaign is what keeps it reproducible, and a restarted stream re-draws
        designs the campaign has already measured.

        **The genetic teacher is moved, not carried and not rebuilt.** It is the
        one piece of state here that would be silently wrong if it came across
        untouched: a
        [GeneticAlgorithm][evogfn.algorithms.baselines.genetic.GeneticAlgorithm]
        reverts surplus substitutions to *its own* environment's parent, so a
        teacher left at the old anchor would breed around a design nobody is
        searching from any more, and
        [train_genetic_gfn][evogfn.algorithms.gflownet.genetic_gfn.train_genetic_gfn]
        filters its offspring through the new environment -- so the teacher would
        quietly contribute nothing while the run continued to look like
        Genetic-GFN. It is moved through its own hook rather than re-founded here
        because whether a population is carried or refounded is the GA's own
        question, answered by its ``carry_population`` flag and the reasoning in
        its docstring; deciding it a second time in this module would override a
        caller's choice from the outside.

        **What is not carried is what does not exist.** This sampler holds no
        measurements: ``observe`` is the inherited no-op, and everything it
        learns arrives through the proxy, which is a surrogate over sequences and
        so anchor-free. The rank-based buffer is built inside each retrain and
        never crosses a round boundary, let alone a move; were it ever hoisted
        onto the sampler it would need re-projecting, since a stored sequence is
        anchor-free but its membership of the new ball is not.

        Args:
            env: The re-anchored environment. Must describe the same space --
                same sequence length, same action layout -- since the policy's
                heads are sized to it.

        Returns:
            A sampler over ``env``, sharing this one's policy and random stream
            and continuing its counts. This one is left usable but should not be
            used: it would train the same weights against the old anchor's masks.

        Raises:
            ValueError: If ``env`` changes the sequence length or the size of the
                action space, which would leave the policy emitting logits for
                actions that no longer exist. Refused rather than reshaped: a
                policy silently mis-indexed against its environment proposes
                designs no one chose, and nothing downstream would raise.
        """
        if env.sequence_length != self._env.sequence_length or env.n_actions != self._env.n_actions:
            raise ValueError(
                f"cannot carry a policy over {self._env.n_actions} actions on sequences of "
                f"length {self._env.sequence_length} into an environment of {env.n_actions} "
                f"actions on length {env.sequence_length}; the anchor may move but the "
                f"sequence length and alphabet may not"
            )

        moved = GFlowNetSampler(
            env,
            self._policy,
            proxy=self._proxy,
            reward=self._reward,
            config=self._config,
            objective=self._objective,
            genetic=None if self._genetic is None else self._genetic.reanchored(env),
            genetic_config=self._genetic_config,
            seed=self._seed,
        )
        moved._generator = self._sampling_generator()
        moved._rounds_trained = self._rounds_trained
        moved._proxy_calls = self._proxy_calls
        moved._proposals_made = self._proposals_made
        # The teacher's replay tallies are a running measurement of how much of
        # what the GA bred the policy could be taught on, so they are a campaign
        # total for the same reason the proxy spend is: restarted at each anchor
        # they would report the last anchor's rounds as the whole run's.
        moved._bred_designs = self._bred_designs
        moved._unconstructible_designs = self._unconstructible_designs
        return moved

    def propose(self, n: int) -> Tokens:
        """Retrain against the current proxy, then sample ``n`` designs.

        Args:
            n: How many candidates to return.

        Returns:
            An ``(n, sequence_length)`` array of terminal states.
        """
        if self._proxy.is_ready:
            # A distinct seed per round, or every round replays the same
            # trajectories and the later rounds teach nothing.
            config = replace(self._config, seed=self._config.seed + self._rounds_trained)
            # The unguided branch first so that `result` is typed by the general
            # result and the genetic one narrows into it, rather than the other
            # way round.
            if self._genetic is None:
                result = train_trajectory_balance(
                    self._env,
                    self._policy,
                    self._proxy,
                    self._reward,
                    config,
                    objective=self._objective,
                )
            else:
                result = train_genetic_gfn(
                    self._env,
                    self._policy,
                    self._proxy,
                    self._reward,
                    config,
                    genetic=self._genetic,
                    genetic_config=self._genetic_config,
                    objective=self._objective,
                )
                # Only a run with a teacher scores designs the policy did not
                # itself produce, so this is the only branch that can meet one
                # it cannot construct. Accumulated rather than replaced: the
                # quantity wanted is the campaign's share, not the last round's.
                self._bred_designs += result.bred_designs
                self._unconstructible_designs += result.unconstructible_designs
            self._proxy_calls += result.oracle_calls
            self._rounds_trained += 1

        trajectories = sample_trajectories(
            self._env, self._policy, n, epsilon=0.0, generator=self._sampling_generator()
        )
        self._count(n)
        return np.asarray(trajectories.terminal)

    def _sampling_generator(self) -> torch.Generator:
        """One generator across rounds, so proposals do not repeat themselves."""
        if self._generator is None:
            self._generator = torch.Generator().manual_seed(self._seed)
        return self._generator

    def __repr__(self) -> str:
        """Name the sampler and how much training it has had."""
        return f"GFlowNetSampler(rounds_trained={self._rounds_trained})"
