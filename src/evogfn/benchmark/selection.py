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

Screening, and what a screen is not allowed to do
-------------------------------------------------

The first stages of this phase move one axis at a time, which cannot see
interactions and cannot afford to visit every axis at all. The last stage
therefore samples a *joint* space at few seeds and then re-measures a shortlist
at the full seed count.

**The screen nominates. It never ranks, and nothing from it is a result.** A
ranking over few seeds on a coarse discrete objective is mostly noise, so the
screen's only output is a set of candidates; the confirmation, at the full seed
count and through `select` above, is the measurement. No number the screen
produces appears in a table, a caption, or a claim.

That boundary is what makes widening the search safe rather than a fishing
expedition, and it rests on one further guarantee: the configuration already in
hand is confirmed alongside the nominees whatever the screen thought of it. A
screen can therefore only ever *add* a better configuration, never displace the
current one on evidence too thin to displace anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Protocol

import numpy as np

from evogfn.algorithms.gflownet.objectives import ContrastiveBalance, TrajectoryBalance
from evogfn.benchmark.methods import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MIX,
    DEFAULT_TRAINING_STEPS,
    genetic_gflownet,
    gflownet,
)
from evogfn.benchmark.statistics import compare

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from evogfn.benchmark.methods import Methodology


class Scored(Protocol):
    """The two numbers the rule reads from a stored campaign, and its standing.

    Structural rather than a `RunRecord`, because the rule is arithmetic over a
    couple of fields and nothing else about a record concerns it. That keeps the
    rule testable against hand-built cases -- a selection procedure whose
    behaviour can only be exercised by running campaigns is one whose edge cases
    go unexercised.

    ``exhausted`` is declared here rather than reached for with ``getattr``
    precisely because it must not be optional to consider. A reader that forgets
    it averages a failed campaign into a mean and picks a configuration on the
    strength of the seeds it survived; making it part of the shape means a new
    reader has to look at it.
    """

    @property
    def regret(self) -> float:
        """Distance from what the task was audited to contain."""

    @property
    def diversity(self) -> float:
        """Mean pairwise distance among the top designs."""

    @property
    def exhausted(self) -> bool:
        """Whether the campaign behind this record failed to finish."""


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
    """Mean regret and diversity per arm over the shared seeds.

    An arm that exhausted on any shared seed is made ineligible outright rather
    than scored on the seeds it survived, and that is a different decision from
    the one below it about a missing number. A ``nan`` regret is a quantity this
    campaign does not report; an exhausted seed is a campaign that **could not be
    run to the end**, which is a property of the configuration being selected,
    and the seeds it failed on are the hard ones. Averaging over the rest picks
    the configuration on its own best-case subset -- and does so silently, since
    the mean it produces is a perfectly ordinary number.

    Ineligible, not merely penalised: this rule chooses *our own* configuration,
    and one that cannot complete a campaign is not a configuration to ship
    whatever it scores where it did complete.

    Nothing stored before ``exhausted`` existed can be affected, since it
    defaults to ``False`` -- so no selection already made moves under this.

    Args:
        records: Stored records per arm, keyed by seed.
        seeds: The seeds every arm holds.

    Returns:
        Mean regret and mean diversity per arm, with ``inf``/``-inf`` for an arm
        that is not eligible to be chosen.
    """
    regret: dict[str, float] = {}
    diversity: dict[str, float] = {}
    for name, held in records.items():
        if any(held[s].exhausted for s in seeds):
            regret[name] = float("inf")
            diversity[name] = float("-inf")
            continue
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
        raise ValueError(
            "every arm failed on every shared seed, or exhausted on one of them; "
            "nothing here is eligible to be chosen"
        )

    # Everything a paired comparison cannot separate from the leader is still in
    # the running, including arms with a worse mean: losing by less than the
    # noise is not losing.
    tied = [leader]
    reference = np.array([records[leader][s].regret for s in shared], dtype=np.float64)
    for name in sorted(records):
        if name == leader:
            continue
        # An ineligible arm cannot be tied into the running by the back door.
        # `_means` has already refused it, and a tie is what would otherwise
        # carry it into the diversity tie-break and let it win there.
        if not np.isfinite(regret[name]):
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

#: Objectives that breed, and so read an offline mixing ratio. Refused rather
#: than ignored elsewhere, for the same reason `lam` is.
_TAKES_MIX = frozenset({"genetic-gfn"})


def _build_objective(  # noqa: PLR0913 - an arm is defined by its hyperparameters
    name: str,
    beta: float,
    steps: int = DEFAULT_TRAINING_STEPS,
    *,
    lam: float | None = None,
    mix: float | None = None,
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
        mix: Share of each training batch bred by a genetic teacher rather than
            sampled from the policy, which stands to Genetic-GFN exactly as
            ``lam`` stands to sub-trajectory balance: it is the knob that makes
            the method a family rather than a point, since the arm degenerates
            to an ordinary GFlowNet at one end of it. ``None`` takes
            [genetic_gflownet][evogfn.benchmark.methods.genetic_gflownet]'s own
            default.
        hidden_dim: Width of the policy trunk, passed to every objective because
            capacity is a property of the policy rather than of the loss.

    Returns:
        A methodology.

    Raises:
        KeyError: If the name is not a known objective. Raised rather than
            defaulted, because a typo that silently produced trajectory balance
            would make a beta scan report the wrong objective's curve.
        ValueError: If ``lam`` or ``mix`` is given for an objective that reads
            neither. Accepting and ignoring one is the worse failure: a scan
            over it would then build one identical arm per value, and a column
            of identical rows reads as "this knob does not matter" rather than
            as "this knob was never applied".
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
    if mix is not None and name not in _TAKES_MIX:
        raise ValueError(
            f"objective {name!r} breeds nothing, so mix={mix} would be silently dropped; "
            f"mix applies to {', '.join(sorted(_TAKES_MIX))}"
        )

    if name == "gfn-tb":
        return gflownet(TrajectoryBalance(), beta=beta, steps=steps, hidden_dim=hidden_dim)
    if name == "gfn-contrastive":
        return gflownet(
            ContrastiveBalance(prune_threshold=0.1), beta=beta, steps=steps, hidden_dim=hidden_dim
        )
    if name == "genetic-gfn":
        return genetic_gflownet(
            TrajectoryBalance(),
            beta=beta,
            steps=steps,
            mix=DEFAULT_MIX if mix is None else mix,
            hidden_dim=hidden_dim,
        )
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


#: Gradient steps per round. This is the GFlowNet's *proxy* budget -- steps x
#: batch_size proxy evaluations per round -- and it is a knob we chose rather
#: than anything the architecture dictates, so it has to be measured rather than
#: inherited. Proxy spend is a reported column in the results table, which makes
#: this a number the paper prints.
#:
#: The values cluster where the knee plausibly is rather than spanning orders of
#: magnitude: a grid whose points are an order of magnitude apart cannot locate a
#: knee sitting between two of them. The question is not "does more help" but
#: "how little is enough", and one point above the inherited setting is what
#: keeps the answer from being pinned at the edge of what was offered.
SCREEN_STEPS: tuple[int, ...] = (50, 100, 150, 200, 300, 450)

#: Sub-trajectory balance's weight per unit sub-trajectory length -- the
#: parameter that objective *is*. It interpolates detailed balance as it
#: approaches zero and trajectory balance as it grows, so an objective selected
#: without ever moving it has been selected at one point of a one-parameter
#: family. That is the same fault as comparing objectives at another objective's
#: inherited defaults, which is what this whole phase exists to avoid.
SCREEN_LAMS: tuple[float, ...] = (0.1, 0.2, 0.5, 0.75, 0.9)

#: Genetic-GFN's offline mixing ratio, which stands to it as `SCREEN_LAMS` does
#: to sub-trajectory balance.
#:
#: Both endpoints are excluded on purpose and neither exclusion is a matter of
#: taste. At zero the teacher contributes nothing and the arm *is* plain
#: trajectory balance carrying unused machinery, so that point does not measure
#: this method; at one the policy sees only bred offspring and never its own
#: samples, which the objective comparison already covers. The interior values
#: bracket the inherited ratio from both sides and reach close to each excluded
#: endpoint without standing on it, so a screen that likes an extreme says which
#: extreme rather than stopping at the middle.
SCREEN_MIXES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

#: Width of the policy trunk. Capacity is the one axis where "the default was
#: fine" and "the policy could not represent the target" produce the same flat
#: table, so it is bracketed on both sides of the shipped width rather than
#: assumed.
SCREEN_HIDDEN_DIMS: tuple[int, ...] = (64, 128, 256)

#: Reward exponents carried into the joint screen, deliberately narrower than
#: `SELECTION_BETAS`.
#:
#: The exponent is here at all because the earlier scan measured it at one step
#: count. If the screen prefers a much smaller step count the exponent need not
#: still be right, and that is the one interaction there is prior reason to
#: expect, so a narrow range around the region the scan found flat is what
#: guards it.
#:
#: Narrow in the low direction specifically. Above the flat region the earlier
#: curve degrades monotonically, and it does so on far more seeds per point than
#: a screen can afford -- spending screen configurations there would re-ask, and
#: re-ask worse, a question already answered. The top of the plausible range is
#: kept as a control so the screen can still show an optimum shifting upward,
#: which is the failure the exponent was carried in to catch.
SCREEN_BETAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 1.0)

#: The objectives that get a joint screen of their own.
#:
#: The objective comparison ran every candidate at one shared set of inherited
#: defaults, which was symmetric and therefore fair. Screening only its winner
#: would put a tuned arm against untuned ones and reintroduce exactly the
#: asymmetry this phase exists to remove -- a comparison that measures the
#: tuning. Genetic-GFN is where that bites hardest: its own defining parameter
#: has never been moved, and the premise it rests on -- that directed evolution
#: *is* a genetic algorithm -- is the strongest prior reason any variant here
#: had to work. Either outcome of screening it is reportable, and "tuned on its
#: own axes and still behind" is a far stronger statement than one drawn from a
#: single inherited configuration.
SCREENED_OBJECTIVES: tuple[str, ...] = ("gfn-subtb", "genetic-gfn")

#: Configurations drawn per screen.
#:
#: Random rather than a grid, and that is not a concession. The joint space is
#: far larger than the budget, has conditional structure -- an axis that exists
#: for one objective and not the other -- and few effective dimensions, which is
#: the regime where random search matches or beats a grid at equal cost: a grid
#: spends its budget re-measuring the axes that do not matter, while every
#: random draw is a fresh value on every axis at once.
SCREEN_SIZE = 100

#: Seeds per screened configuration. Deliberately far below what the rule needs
#: to decide anything, because the screen decides nothing: it nominates, and the
#: confirmation measures. Buying resolution here would buy it in the one place
#: where no claim is drawn.
SCREEN_SEEDS = 10

#: How many screened configurations go forward to the confirmation, across all
#: screens. The split between screens is a judgement made when the screens land
#: and is not encoded here; `propose_allocation` puts a proposal on the screen
#: and the finalists are then named explicitly.
CONFIRM_SLOTS = 10

#: The fewest slots any screened objective may be given. A ten-seed screen is
#: not evidence enough to write an objective off entirely, and the floor is what
#: stops it doing so -- which matters most for an objective that has already
#: been placed low once on evidence that later turned out to be confounded.
CONFIRM_FLOOR = 2

#: Seed for the draw. Fixed and stated so the sampled set is reproducible from
#: this number alone: a screen nobody can redraw is a screen whose candidate set
#: cannot be checked, and "we sampled a hundred at random" is then unfalsifiable.
SCREEN_SAMPLING_SEED = 20260803

#: Separates an objective's name from its settings in an arm name. A hyphen
#: cannot do it: objective names contain hyphens, so a hyphen-split would have
#: no unambiguous boundary and recovering a setting would be guesswork about
#: where the name ended.
_SETTINGS = "@"


@dataclass(frozen=True, slots=True)
class Configuration:
    """One point of the joint space, and the arm name that identifies it.

    Attributes:
        objective: The training objective, named as in `OBJECTIVES`.
        beta: Reward exponent.
        steps: Gradient steps per round.
        hidden_dim: Width of the policy trunk.
        lam: Sub-trajectory length weighting, or ``None`` for an objective that
            has none.
        mix: Offline mixing ratio, or ``None`` for an objective that breeds
            nothing.
    """

    objective: str
    beta: float
    steps: int
    hidden_dim: int
    lam: float | None = None
    mix: float | None = None

    @property
    def name(self) -> str:
        """The arm name, which is a store key and therefore an identity.

        Store keys are file names, so a scheme that changes orphans every record
        written under the old one -- the campaigns are still on disk, and nothing
        will ever look for them again. That is the only requirement here that
        cannot be walked back later, so the grammar is fixed: every axis the
        objective reads appears, in one order, with one spelling per value.

        The name is for identity and nothing else. Recovering what an arm ran at
        is `from_record`'s job, which reads the settings the campaign stored
        beside its result; a reader that instead pulls a number out of a name is
        one component away from silently reading a different number.
        """
        parts = [f"b{self.beta:g}", f"s{self.steps}"]
        if self.lam is not None:
            parts.append(f"l{self.lam:g}")
        if self.mix is not None:
            parts.append(f"m{self.mix:g}")
        parts.append(f"h{self.hidden_dim}")
        return f"{self.objective}{_SETTINGS}{'-'.join(parts)}"

    def build(self) -> Methodology:
        """The methodology this configuration names.

        Returns:
            A methodology, carrying its settings for the record.

        Raises:
            KeyError: If the objective is not a known one.
            ValueError: If an axis is set that this objective does not read.
        """
        return _build_objective(
            self.objective,
            self.beta,
            self.steps,
            lam=self.lam,
            mix=self.mix,
            hidden_dim=self.hidden_dim,
        )

    @classmethod
    def parse(cls, name: str) -> Configuration:
        """Read a configuration back out of an arm name.

        This exists to make the naming scheme *checkable* rather than to be the
        way settings are recovered. A scheme whose names round-trip is one where
        two distinct configurations cannot collide on a store key, which is the
        property that keeps records attributable; a test asserting the round-trip
        over the whole space proves it, and nothing else can.

        Args:
            name: An arm name produced by `name`.

        Returns:
            The configuration it identifies.

        Raises:
            ValueError: If the name is not in the scheme, or omits an axis every
                configuration carries. Raised rather than defaulted: a name that
                silently parsed to the wrong point is the failure this scheme was
                designed to make impossible.
        """
        objective, marker, settings = name.partition(_SETTINGS)
        if not marker:
            raise ValueError(f"{name!r} is not a screened arm name; it carries no {_SETTINGS!r}")
        found: dict[str, str] = {}
        for token in settings.split("-"):
            if len(token) < 2 or not token[0].isalpha():  # noqa: PLR2004 - a letter and a value
                raise ValueError(f"{name!r} has an unreadable setting {token!r}")
            if token[0] in found:
                raise ValueError(f"{name!r} sets {token[0]!r} more than once")
            found[token[0]] = token[1:]
        missing = [key for key in ("b", "s", "h") if key not in found]
        if missing:
            raise ValueError(f"{name!r} omits {', '.join(missing)}")
        unknown = sorted(set(found) - set("bslmh"))
        if unknown:
            raise ValueError(f"{name!r} carries unknown setting(s) {', '.join(unknown)}")
        return cls(
            objective=objective,
            beta=float(found["b"]),
            steps=int(found["s"]),
            hidden_dim=int(found["h"]),
            lam=None if "l" not in found else float(found["l"]),
            mix=None if "m" not in found else float(found["m"]),
        )

    @classmethod
    def from_record(cls, name: str, parameters: Mapping[str, str | float | bool]) -> Configuration:
        """What an arm actually ran at, checked against what its name claims.

        The settings come from the stored record rather than from the name,
        because that is where they are true: a record carries the values the
        methodology closed over, while a name carries whatever the scheme said
        at the time it was written. The name is still parsed, and a disagreement
        raises -- an arm whose name and record describe different configurations
        is precisely the bug this guards, and it is silent in every other reading.

        Args:
            name: The arm name, as stored.
            parameters: The record's resolved settings.

        Returns:
            The configuration the campaign ran.

        Raises:
            ValueError: If the record omits a setting the arm needs, or if it
                disagrees with the name.
        """
        claimed = cls.parse(name)
        missing = [key for key in ("beta", "steps", "hidden_dim") if key not in parameters]
        if missing:
            raise ValueError(f"record for {name!r} omits {', '.join(missing)}")
        ran = cls(
            objective=claimed.objective,
            beta=float(parameters["beta"]),
            steps=int(parameters["steps"]),
            hidden_dim=int(parameters["hidden_dim"]),
            lam=None if "lam" not in parameters else float(parameters["lam"]),
            mix=None if "mix" not in parameters else float(parameters["mix"]),
        )
        if ran != claimed:
            raise ValueError(
                f"arm {name!r} names {claimed} but its record says it ran {ran}; "
                f"one of the two is describing a campaign that never happened"
            )
        return ran


def incumbent(objective: str, beta: float) -> Configuration:
    """The configuration already in hand, at the settings it ships with.

    Read from the shipped defaults rather than restated, so that a default moved
    somewhere else moves the incumbent with it instead of leaving this claiming
    to be current while naming a value nothing else uses.

    Args:
        objective: The objective the earlier stages settled on.
        beta: The reward exponent they settled on.

    Returns:
        The incumbent configuration.
    """
    from evogfn.algorithms.gflownet.flow_objectives import DEFAULT_LAMBDA  # noqa: PLC0415

    return Configuration(
        objective=objective,
        beta=beta,
        steps=DEFAULT_TRAINING_STEPS,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        lam=DEFAULT_LAMBDA if objective in _TAKES_LAM else None,
        mix=DEFAULT_MIX if objective in _TAKES_MIX else None,
    )


def screen_space(objective: str) -> tuple[Configuration, ...]:
    """Every configuration one objective's screen could draw.

    The axes an objective does not read are absent rather than fixed at a
    placeholder, which is what keeps two screens comparable in size: a space
    padded with an inert axis would look larger than it is and would be sampled
    at a correspondingly lower density on the axes that do exist.

    Args:
        objective: The objective to enumerate, named as in `OBJECTIVES`.

    Returns:
        The full joint space, in a fixed order so that a draw from it is
        reproducible from the sampling seed alone.
    """
    lams: tuple[float | None, ...] = SCREEN_LAMS if objective in _TAKES_LAM else (None,)
    mixes: tuple[float | None, ...] = SCREEN_MIXES if objective in _TAKES_MIX else (None,)
    return tuple(
        Configuration(
            objective=objective, beta=beta, steps=steps, hidden_dim=width, lam=lam, mix=mix
        )
        for beta, steps, lam, mix, width in product(
            SCREEN_BETAS, SCREEN_STEPS, lams, mixes, SCREEN_HIDDEN_DIMS
        )
    )


def sample_screen(
    objective: str,
    *,
    size: int = SCREEN_SIZE,
    seed: int = SCREEN_SAMPLING_SEED,
) -> tuple[Configuration, ...]:
    """Draw one screen's configurations uniformly, without replacement.

    Without replacement is not a detail. Two draws of the same point share a
    store key, so the duplicate costs a campaign and buys nothing, and the screen
    would then have fewer distinct configurations than it claims to have.

    Args:
        objective: The objective being screened.
        size: How many configurations to draw.
        seed: The sampling seed. The whole draw is a function of this and the
            grids above, so anyone can reproduce the candidate set exactly.

    Returns:
        The drawn configurations, in the space's own order rather than the
        draw's, so that re-running with the same seed yields an identical list
        and a diff of two screens is readable.

    Raises:
        ValueError: If more configurations are asked for than the space holds.
            Silently returning the whole space would report a screen where an
            exhaustive grid was run.
    """
    space = screen_space(objective)
    if size > len(space):
        raise ValueError(
            f"cannot draw {size} distinct configurations from a space of {len(space)}; "
            f"at that size the screen is an exhaustive grid and should say so"
        )
    drawn = np.random.default_rng(seed).choice(len(space), size=size, replace=False)
    return tuple(space[index] for index in sorted(int(i) for i in drawn))


def screen_arms(configurations: Sequence[Configuration]) -> dict[str, Methodology]:
    """Build the arms for a set of configurations.

    Args:
        configurations: What to build.

    Returns:
        Methodologies by arm name.

    Raises:
        ValueError: If two configurations share a name. They would share a store
            key, and the second would be read as a re-run of the first.
    """
    arms = {configuration.name: configuration.build() for configuration in configurations}
    if len(arms) != len(configurations):
        raise ValueError("two configurations share an arm name, so they would share a store key")
    return arms


@dataclass(frozen=True, slots=True)
class Ranked:
    """One screened configuration's standing, for nomination only.

    Attributes:
        name: The arm name.
        regret: Mean regret over the seeds every arm in its screen shares.
        spread: Standard deviation of that regret across the same seeds,
            reported beside the mean rather than left out. A mean alone invites
            a reader to compare two configurations by their decimal places; the
            spread is what says whether the gap between them is anything at all,
            and on a screen the answer is usually that it is not.
        diversity: Mean top-K diversity over the same seeds.
        separated: Whether a paired comparison separates it from its screen's
            leader. ``True`` for the leader itself, since nothing separates it
            from itself and it is not a candidate for being merged into a tie,
            and ``True`` for a configuration that exhausted on a shared seed --
            that one is not unresolved, it is out.
        regrets: The per-seed regrets the mean was taken over, in the screen's
            seed order. Carried rather than summarised away because the only
            comparison left to make -- one screen's leader against another's --
            is a paired one, and a mean cannot be paired against a mean.
    """

    name: str
    regret: float
    spread: float
    diversity: float
    separated: bool
    regrets: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Screen:
    """What one screen produced, and how little of it is resolved.

    Attributes:
        objective: The objective screened.
        ranked: Every configuration, ordered by mean regret and then by
            diversity. **Not a result.** The order exists so that a shortlist can
            be drawn from the top of it; the numbers in it are ten-seed means on
            a coarse discrete objective and are not reported anywhere.
        seeds: The seeds every arm in this screen holds.
    """

    objective: str
    ranked: tuple[Ranked, ...]
    seeds: tuple[int, ...]

    @property
    def leader(self) -> Ranked:
        """The lowest mean regret in the screen.

        Raises:
            ValueError: If the screen holds nothing.
        """
        if not self.ranked:
            raise ValueError(f"screen for {self.objective} holds no configuration")
        return self.ranked[0]

    @property
    def indistinguishable(self) -> tuple[str, ...]:
        """The plausible set: the leader, and everything level with it.

        The honest measure of what a screen resolved, and the leader is counted
        because the question this answers is how large the set of candidates the
        evidence cannot rank actually is. When it covers most of the ranking, the
        ordering below the leader is noise and a shortlist is a *sample* of the
        plausible set rather than the best of it -- which is worth knowing before
        the confirmation is paid for rather than after.
        """
        return (self.leader.name, *(e.name for e in self.ranked[1:] if not e.separated))


def rank_screen(objective: str, records: Mapping[str, Mapping[int, Scored]]) -> Screen:
    """Order one screen's configurations, and say what it failed to resolve.

    Ordered by mean regret, with mean diversity breaking exact ties. That is the
    pre-declared rule's own pair of axes and in its own priority, so the
    shortlist is not drawn on a criterion invented for the screen -- but it is
    applied here to *nominate*, never to rank in a reported sense, and a tie
    broken this way is broken to put a name in a list rather than to declare a
    winner.

    Args:
        objective: The objective screened, for the label.
        records: Stored records per arm, keyed by seed.

    Returns:
        The screen.

    Raises:
        ValueError: If there is nothing to rank, or no seed is shared by every
            arm -- an unpaired mean is not comparable to another unpaired mean,
            and a shortlist drawn from a mixture of seed sets would be a
            shortlist of who ran on the easiest seeds.
    """
    if not records:
        raise ValueError(f"screen for {objective} has no arms")
    shared = sorted(set.intersection(*(set(held) for held in records.values())))
    if not shared:
        raise ValueError(f"screened arms for {objective} share no seed, so nothing can be paired")

    regret, diversity = _means(records, shared)
    order = sorted(records, key=lambda name: (regret[name], -diversity[name], name))
    leader = order[0]
    reference = np.array([records[leader][s].regret for s in shared], dtype=np.float64)
    ranked = []
    for name in order:
        mine = np.array([records[name][s].regret for s in shared], dtype=np.float64)
        # A single seed carries no variance, so nothing can be separated from
        # anything on one -- which is the honest answer rather than an error,
        # since a one-seed run of this stage is how the wiring gets smoke-tested
        # and a crash there would only be reached after the campaigns were paid
        # for.
        usable = len(shared) > 1 and bool(np.isfinite(mine).all() and np.isfinite(reference).all())
        # An ineligible arm counts as separated, which is not a technicality:
        # `indistinguishable` is read as "the evidence cannot rank these against
        # the leader", and an arm that exhausted has been ranked -- it failed.
        # Left unseparated it would swell the plausible set and could be
        # nominated out of it.
        separated = (
            name == leader
            or not np.isfinite(regret[name])
            or (usable and compare(name, mine, reference, higher_is_better=False).significant)
        )
        finite = mine[np.isfinite(mine)]
        ranked.append(
            Ranked(
                name=name,
                regret=regret[name],
                spread=float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                diversity=diversity[name],
                separated=separated,
                regrets=tuple(float(value) for value in mine),
            )
        )
    return Screen(objective=objective, ranked=tuple(ranked), seeds=tuple(shared))


def propose_allocation(
    screens: Sequence[Screen],
    *,
    slots: int = CONFIRM_SLOTS,
    floor: int = CONFIRM_FLOOR,
) -> dict[str, int]:
    """A starting point for how many finalists each screen contributes.

    A *proposal*, printed for a human to accept or override, and deliberately
    not wired to anything that runs. The allocation is a judgement about how much
    of a fixed confirmation budget to spend on an objective whose screen looked
    worse, and a ten-seed screen is not evidence enough to make that judgement
    mechanically.

    The proposal itself is two-valued, which is as much as the evidence supports.
    Where a paired comparison between the screens' leaders cannot separate them,
    the split is even: nothing distinguishes the objectives, so nothing justifies
    favouring one. Where it can, the weaker screen keeps the floor and the rest
    goes to the better one -- and the floor is the whole point, since the
    comparison being made is between two *winners of a search*, which is biased
    toward whichever screen sampled luckier and is not a comparison between the
    objectives themselves.

    Args:
        screens: The screens, each already ranked.
        slots: Finalists to allocate across them. The incumbent is not one of
            them: it is confirmed regardless and is not competing for a slot.
        floor: The fewest any screen may be given.

    Returns:
        Slots by objective name.

    Raises:
        ValueError: If the floors cannot all be met, or if the screens do not
            share a seed set -- an unpaired comparison between leaders would
            decide the split on which screen ran on which seeds.
    """
    if not screens:
        raise ValueError("no screens to allocate between")
    if floor * len(screens) > slots:
        raise ValueError(
            f"{slots} slots cannot give {len(screens)} screens a floor of {floor} each"
        )
    if len({screen.seeds for screen in screens}) != 1:
        raise ValueError("screens do not share a seed set, so their leaders cannot be paired")

    even, remainder = divmod(slots, len(screens))
    proposal = {
        screen.objective: even + (1 if index < remainder else 0)
        for index, screen in enumerate(screens)
    }
    if len(screens) != 2:  # noqa: PLR2004 - the skew below is a two-way comparison
        return proposal

    first, second = screens
    # Paired on the shared seeds, not compared by their means. Two ten-seed
    # means are never exactly equal, so a test on the means alone would skew the
    # split every time it was asked -- and skew it on the one difference in this
    # whole design least entitled to decide anything, since each leader is the
    # winner of its own search and is flattered by however luckily that search
    # sampled.
    if len(first.seeds) < 2 or not _leaders_separated(first, second):  # noqa: PLR2004
        return proposal
    better, worse = (
        (first, second) if first.leader.regret < second.leader.regret else (second, first)
    )
    return {better.objective: slots - floor, worse.objective: floor}


def _leaders_separated(first: Screen, second: Screen) -> bool:
    """Whether a paired comparison separates two screens' leaders."""
    mine = np.array(first.leader.regrets, dtype=np.float64)
    theirs = np.array(second.leader.regrets, dtype=np.float64)
    if not (np.isfinite(mine).all() and np.isfinite(theirs).all()):
        return False
    label = f"{first.objective} vs {second.objective}"
    return compare(label, mine, theirs, higher_is_better=False).significant


def confirmation_set(
    finalists: Sequence[str], standing: Configuration
) -> tuple[Configuration, ...]:
    """The configurations the confirmation measures.

    The incumbent is unioned in whatever the screens thought of it, and that is
    what makes the whole screen safe to run: the confirmation can only ever add a
    better configuration, never lose the current one to ten noisy seeds. A
    shortlist that happened to contain it is not duplicated.

    Args:
        finalists: Arm names nominated from the screens.
        standing: The configuration already in hand.

    Returns:
        The configurations to confirm, the incumbent first so that a truncated
        reading of the table still contains it.

    Raises:
        ValueError: If a finalist's name is not in the naming scheme.
    """
    chosen = [standing]
    for name in finalists:
        configuration = Configuration.parse(name)
        if configuration not in chosen:
            chosen.append(configuration)
    return tuple(chosen)
