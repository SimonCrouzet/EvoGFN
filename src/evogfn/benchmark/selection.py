"""Choosing the configuration the headline benchmark reports.

Every classical baseline in this suite runs at hyperparameters its own authors
tuned -- the genetic algorithm uses the Ehrlich paper's mutation and
recombination rates, MLDE the regime Wittmann et al. actually run. A GFlowNet
run at inherited defaults against that field is not being compared to it; the
comparison measures our configuration, and it measures it in the direction that
flatters the baselines.

So a configuration is *selected*, and this module is the rule that selects it.
Writing the rule down before the numbers arrive is the point. A criterion chosen
after seeing the table is not a criterion, and "best regret, except on the task
where diversity looked better" is how a sweep becomes a story.

Where selection happens, and where it must not
----------------------------------------------

Selection runs on the diagnostic landscape. The headline tasks never see it.
This is what keeps the phase from being a tuning run on the test set -- the
configuration is fixed before a single headline campaign is scored, and it is
fixed against a landscape that carries no claim.

The rule
--------

**Lowest mean regret, with diversity breaking statistical ties.**

The tie-break is not a tidy-up. This project's claim is diverse, feasible,
high-fitness variants, not just high-fitness ones; a rule that read regret alone
would happily select a configuration that optimises well and samples badly, and
the diversity column of the headline table would then have to live with whatever
that produced. Ties are decided by measurement rather than by preference.

A tie is *statistical*, not numerical. Two arms tie when a paired comparison
cannot separate them, which is a statement about the evidence rather than about
the decimal places -- an arm ahead by 0.006 with a confidence interval spanning
zero has not won anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from evogfn.algorithms.gflownet.objectives import ContrastiveBalance, TrajectoryBalance
from evogfn.benchmark.methods import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_TRAINING_STEPS,
    genetic_gflownet,
    gflownet,
)
from evogfn.benchmark.statistics import compare

if TYPE_CHECKING:
    from collections.abc import Mapping

    from evogfn.benchmark.methods import Methodology


class Scored(Protocol):
    """The two numbers the rule reads from a stored campaign.

    Structural rather than a `RunRecord`, because the rule is arithmetic over
    two fields and nothing else about a record concerns it. That keeps the rule
    testable against hand-built cases -- a selection procedure whose behaviour
    can only be exercised by running campaigns is one whose edge cases go
    unexercised.
    """

    @property
    def regret(self) -> float:
        """Distance from what the task was audited to contain."""

    @property
    def diversity(self) -> float:
        """Mean pairwise distance among the top designs."""


@dataclass(frozen=True, slots=True)
class Selection:
    """Which arm was chosen, and what the choice rested on.

    Attributes:
        chosen: The winning arm's name.
        reason: Why it won, in a form that can be pasted into a caption.
        tied: Arms a paired comparison could not separate from the leader,
            including the leader. More than one name here means the choice was
            settled on diversity rather than on fitness.
        regret: Mean regret per arm.
        diversity: Mean top-K diversity per arm.
    """

    chosen: str
    reason: str
    tied: tuple[str, ...]
    regret: Mapping[str, float]
    diversity: Mapping[str, float]

    def __repr__(self) -> str:
        """Name the winner and the ground it won on."""
        return f"{self.chosen} ({self.reason})"


def _means(
    records: Mapping[str, Mapping[int, Scored]], seeds: list[int]
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean regret and diversity per arm over the shared seeds."""
    regret: dict[str, float] = {}
    diversity: dict[str, float] = {}
    for name, held in records.items():
        values = np.array([held[s].regret for s in seeds], dtype=np.float64)
        finite = values[np.isfinite(values)]
        # An arm that failed on some seeds is scored on the ones it survived,
        # and an arm that failed on all of them is not eligible to be chosen.
        regret[name] = float(finite.mean()) if finite.size else float("inf")
        spread = np.array([held[s].diversity for s in seeds], dtype=np.float64)
        usable = spread[np.isfinite(spread)]
        diversity[name] = float(usable.mean()) if usable.size else float("-inf")
    return regret, diversity


def _separated_on_diversity(
    records: Mapping[str, Mapping[int, Scored]],
    seeds: list[int],
    name: str,
    reference: str,
) -> bool:
    """Whether one arm's diversity beats another's by more than the noise.

    Args:
        records: Stored records per arm.
        seeds: The shared seeds, so the comparison is paired.
        name: The wider arm.
        reference: The arm it must beat.

    Returns:
        Whether a paired comparison separates them. Higher is better here,
        unlike regret.
    """
    mine = np.array([records[name][s].diversity for s in seeds], dtype=np.float64)
    theirs = np.array([records[reference][s].diversity for s in seeds], dtype=np.float64)
    if not (np.isfinite(mine).all() and np.isfinite(theirs).all()):
        return False
    return bool(compare(name, mine, theirs, higher_is_better=True).significant)


def select(records: Mapping[str, Mapping[int, Scored]]) -> Selection:
    """Apply the selection rule to a stage's stored results.

    Args:
        records: Stored records per arm, keyed by seed. Only seeds every arm
            holds are used, so the paired comparisons that decide ties are
            genuinely paired.

    Returns:
        The chosen arm and the evidence for it.

    Raises:
        ValueError: If there is nothing to choose between, or no seed is shared
            by every arm -- either would otherwise produce a confident-looking
            choice resting on a comparison that never happened.
    """
    if not records:
        raise ValueError("no arms to select from")
    shared = sorted(set.intersection(*(set(held) for held in records.values())))
    if not shared:
        raise ValueError(f"arms {sorted(records)} share no seed, so nothing here can be paired")

    regret, diversity = _means(records, shared)
    leader = min(regret, key=lambda name: regret[name])
    if not np.isfinite(regret[leader]):
        raise ValueError("every arm failed on every shared seed")

    # Everything a paired comparison cannot separate from the leader is still in
    # the running, including arms with a worse mean: losing by less than the
    # noise is not losing.
    tied = [leader]
    reference = np.array([records[leader][s].regret for s in shared], dtype=np.float64)
    for name in sorted(records):
        if name == leader:
            continue
        mine = np.array([records[name][s].regret for s in shared], dtype=np.float64)
        if not np.isfinite(mine).all() or not np.isfinite(reference).all():
            continue
        if not compare(name, mine, reference, higher_is_better=False).significant:
            tied.append(name)

    if len(tied) == 1:
        return Selection(
            chosen=leader,
            reason=(
                f"lowest mean regret {regret[leader]:.4f} over {len(shared)} seeds, "
                f"and separated from every other arm"
            ),
            tied=tuple(tied),
            regret=regret,
            diversity=diversity,
        )

    # The tie-break has to clear the same bar it just applied to regret. Two
    # arms tied on fitness whose diversity differs by less than the evidence can
    # resolve have not been separated by diversity either, and picking the
    # nominally-higher one is discrimination on noise. Without the guard the rule
    # can hand the choice to an arm that is worse on regret on the strength of a
    # diversity margin narrower than its own replicate spread.
    #
    # A tightening, never a loosening: it can only make the tie-break fire less
    # often. Where diversity cannot discriminate, the fallback is the leader on
    # regret, which is where the rule started.
    widest = max(tied, key=lambda name: diversity[name])
    if widest != leader and not _separated_on_diversity(records, shared, widest, leader):
        return Selection(
            chosen=leader,
            reason=(
                f"tied on regret with {len(tied) - 1} other arm(s) over "
                f"{len(shared)} seeds, and the diversity spread among them "
                f"({min(diversity[n] for n in tied):.2f} to "
                f"{max(diversity[n] for n in tied):.2f}) is itself within noise, "
                f"so the tie-break cannot separate them either; lowest mean "
                f"regret {regret[leader]:.4f} decides"
            ),
            tied=tuple(sorted(tied)),
            regret=regret,
            diversity=diversity,
        )

    chosen = widest
    return Selection(
        chosen=chosen,
        reason=(
            f"tied on regret with {len(tied) - 1} other arm(s) over {len(shared)} "
            f"seeds ({', '.join(sorted(tied))}); highest diversity "
            f"{diversity[chosen]:.2f} among them, at regret {regret[chosen]:.4f} "
            f"against the leader's {regret[leader]:.4f}"
        ),
        tied=tuple(sorted(tied)),
        regret=regret,
        diversity=diversity,
    )


# The arm builders live here rather than in `methods.py` on purpose.
# `methods.py` is a campaign entry point, so the result store fingerprints its
# whole import closure; adding even a purely additive function to it restamps
# every stored record as stale and throws away nights of compute. Nothing below
# is reachable from a campaign -- these build arms for the selection phase only
# -- so it belongs on this side of that boundary.


#: Reward exponents the selection phase scans, once an objective has been
#: chosen. The grid brackets the default from both sides on purpose. A scan
#: whose best value sits at its own edge cannot distinguish "this exponent is
#: right" from "this exponent is the largest one offered", so the range has to
#: reach far enough above the default for regret to turn.
#:
#: The values below 1 close the same hole at the other end. Diversity is the
#: axis the tie-break actually decides on, so a grid floored at 1 cannot say
#: whether 1 is the optimum or merely the edge. Below 1 the target R(x)^beta
#: flattens, approaching uniform over the reachable set as beta approaches 0, so
#: regret has to turn upward somewhere down there; where it turns is the thing
#: worth knowing.
#:
#: Widening a grid is only legitimate under the conditions that hold here: the
#: rule is fixed before the numbers arrive, the landscape carries no claim, and
#: the rule is regret-first -- an exponent that buys diversity at a real regret
#: cost is not eligible, since only statistical ties go to diversity. Widening
#: until the answer is agreeable would not be legitimate, so this grid is fixed
#: and the whole curve gets reported either way.
SELECTION_BETAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 10.0, 30.0, 100.0)


#: Objectives that read a sub-trajectory length weighting. Everything else
#: ignores one, which is why `_build_objective` refuses rather than accepts it.
_TAKES_LAM = frozenset({"gfn-subtb"})


def _build_objective(
    name: str,
    beta: float,
    steps: int = DEFAULT_TRAINING_STEPS,
    *,
    lam: float | None = None,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
) -> Methodology:
    """One arm: the named training objective at the given hyperparameters.

    Args:
        name: An objective from `OBJECTIVES` or `flow_objectives`.
        beta: The reward exponent to build it with.
        steps: Gradient steps per round.
        lam: Sub-trajectory balance's weight per unit sub-trajectory length,
            which is the knob that objective *is*: it interpolates detailed
            balance as it approaches zero and trajectory balance as it grows,
            so the objective that won selection is a one-parameter family we
            have only ever evaluated at one point. ``None`` takes
            [SubTrajectoryBalance][evogfn.algorithms.gflownet.flow_objectives.SubTrajectoryBalance]'s
            own default, read from there rather than restated here so the two
            cannot drift apart while both claim to be the shipped setting.
        hidden_dim: Width of the policy trunk, passed to every objective because
            capacity is a property of the policy rather than of the loss.

    Returns:
        A methodology.

    Raises:
        KeyError: If the name is not a known objective. Raised rather than
            defaulted, because a typo that silently produced trajectory balance
            would make a beta scan report the wrong objective's curve.
        ValueError: If ``lam`` is given for an objective that has no length
            weighting. Accepting and ignoring it is the worse failure: a scan
            over ``lam`` would then build one identical arm per value, and a
            column of identical rows reads as "the weighting does not matter"
            rather than as "the weighting was never applied".
    """
    from evogfn.algorithms.gflownet.flow_objectives import (  # noqa: PLC0415
        DEFAULT_LAMBDA,
        DetailedBalance,
        ForwardLookingDetailedBalance,
        SubTrajectoryBalance,
    )

    if lam is not None and name not in _TAKES_LAM:
        raise ValueError(
            f"objective {name!r} has no sub-trajectory length weighting, so lam={lam} would "
            f"be silently dropped; lam applies to {', '.join(sorted(_TAKES_LAM))}"
        )

    if name == "gfn-tb":
        return gflownet(TrajectoryBalance(), beta=beta, steps=steps, hidden_dim=hidden_dim)
    if name == "gfn-contrastive":
        return gflownet(
            ContrastiveBalance(prune_threshold=0.1), beta=beta, steps=steps, hidden_dim=hidden_dim
        )
    if name == "genetic-gfn":
        return genetic_gflownet(TrajectoryBalance(), beta=beta, steps=steps, hidden_dim=hidden_dim)
    if name == "gfn-db":
        return gflownet(
            DetailedBalance(), beta=beta, steps=steps, learn_flow=True, hidden_dim=hidden_dim
        )
    if name == "gfn-subtb":
        return gflownet(
            SubTrajectoryBalance(lam=DEFAULT_LAMBDA if lam is None else lam),
            beta=beta,
            steps=steps,
            learn_flow=True,
            hidden_dim=hidden_dim,
        )
    if name == "gfn-fldb":
        return gflownet(
            ForwardLookingDetailedBalance(),
            beta=beta,
            steps=steps,
            learn_flow=True,
            hidden_dim=hidden_dim,
        )
    raise KeyError(f"unknown objective {name!r}")


def beta_arms(objective: str) -> dict[str, Methodology]:
    """The reward-exponent scan for one objective.

    The second stage of selection. Running it only for the objective that won
    the first stage is what keeps the phase affordable, and it is also why the
    two stages cannot be collapsed: the winning objective is not known until the
    first has finished.

    What this deliberately does not do is scan beta for every objective. The
    best exponent for trajectory balance need not be the best for
    sub-trajectory balance, so a scan run only for the winner can miss an
    interaction -- an objective that loses at beta = 3 and would have won at
    beta = 30. That is a known limit of a two-stage design rather than an
    oversight, and it is the reason the first stage fixes beta at the default
    the objectives were compared at rather than at an arbitrary value.

    Args:
        objective: The objective to scan, named as in `OBJECTIVES`.

    Returns:
        Methodologies by name, one per exponent in `SELECTION_BETAS`.
    """
    return {
        f"{objective}-beta-{beta:g}": _build_objective(objective, beta) for beta in SELECTION_BETAS
    }


#: Gradient steps per round, scanned once the objective and exponent are fixed.
#: This is the GFlowNet's *proxy* budget -- steps x batch_size proxy evaluations
#: per round -- and it is a knob we chose rather than anything the architecture
#: dictates, so it has to be measured rather than inherited.
#:
#: Hyperparameters cannot be assumed to transfer across objectives, so this is
#: scanned for the objective selection settled on rather than carried over from
#: a scan run against a different one.
#:
#: The values below cluster where the knee plausibly is rather than spanning
#: orders of magnitude: a grid whose points are an order of magnitude apart
#: cannot locate a knee sitting between two of them. The question is no longer
#: "does more help" but "how little is enough". That matters beyond tidiness:
#: proxy spend is a reported column in the results table, so halving the answer
#: halves a number the paper prints.
SELECTION_STEPS: tuple[int, ...] = (50, 100, 150, 200, 300)


def steps_arms(objective: str, beta: float) -> dict[str, Methodology]:
    """The gradient-step scan, once objective and exponent are settled.

    Args:
        objective: The objective stage A chose.
        beta: The reward exponent stage B chose.

    Returns:
        Methodologies by name, one per entry in `SELECTION_STEPS`.
    """
    return {
        f"{objective}-beta-{beta:g}-steps-{steps}": _build_objective(objective, beta, steps)
        for steps in SELECTION_STEPS
    }
