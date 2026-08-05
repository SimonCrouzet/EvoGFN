"""Runs arms across seeds and protocols, and reports what actually separated.

The harness exists because the alternative -- a script per experiment -- is how
comparisons drift. Seeds get reused between arms, one arm quietly runs a
different batch size, a method is compared against a number from a previous run
at a different budget. None of those announces itself in the output.

Three properties it enforces
----------------------------

**Every arm sees the same seed.** An arm is a factory taking a seed, so pairing
is structural: on seed 7 the surrogate, the environment and the initial design
are identical across arms, and the only difference is the method. That is what
makes the paired statistics valid rather than merely convenient.

**Every arm runs the same protocol.** Rounds and batch size come from one
[Protocol][evogfn.benchmark.protocol.Protocol], so no arm can be measured
against a different budget than the one it is reported at.

**Actual spend is recorded, not assumed.** A sampler that cannot fill its plate
spends less than its budget, and the difference is reported per arm rather than
being folded into a headline number that claims otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from evogfn.benchmark.statistics import PairedComparison, compare, seeds_needed
from evogfn.loop.campaign import Campaign
from evogfn.metrics.diversity import diversity

if TYPE_CHECKING:
    from evogfn.benchmark.protocol import Protocol
    from evogfn.landscapes.base import FitnessLandscape

#: An arm builds everything a campaign needs, from a seed. Returning the whole
#: campaign rather than just the sampler is deliberate: an arm that varies the
#: acquisition rule or the surrogate is as legitimate as one that varies the
#: sampler, and the harness should not privilege one axis.
ArmFactory = Callable[[int, "Protocol"], Campaign]


@dataclass(frozen=True, slots=True)
class ArmResult:
    """What one method did across every seed.

    A seed on which the campaign exhausted is carried as ``nan`` in ``best`` and
    therefore in ``regret``; see `run_benchmark`. Every reader below has to
    decide what to do with those, and they do not all decide the same thing.

    Attributes:
        name: The arm's label.
        best: Best value found, per seed. ``nan`` where the campaign could not
            finish, which is a different statement from a bad value.
        regret: Distance to the optimum, per seed. Empty when the landscape
            does not know its optimum.
        diversity: Mean pairwise Hamming distance over everything measured.
        spent: Oracle calls actually used, per seed.
        budget: Calls the protocol allowed.
    """

    name: str
    best: np.ndarray
    regret: np.ndarray
    diversity: np.ndarray
    spent: np.ndarray
    budget: int

    @property
    def underspent(self) -> bool:
        """Whether any seed failed to use its budget.

        Worth surfacing: a method that cannot fill its plate is not being
        compared at the budget the table claims.
        """
        return bool((self.spent < self.budget).any())

    @property
    def failed(self) -> int:
        """Seeds whose campaign produced no measurement at all.

        Counted rather than inferred from a ``nan`` in the printed mean: a
        summary line reading ``nan`` says something went wrong somewhere on this
        arm, and this says how often and lets the rest of the line still be read.
        """
        return int((~np.isfinite(self.best)).sum())

    def summary(self) -> str:
        """One line: the metric, its standard error, the spend, and the failures.

        The mean is taken over the seeds that produced a measurement, and the
        seeds that did not are reported as a count beside it. Both halves are
        load-bearing. Averaging ``nan`` in makes every figure on the line ``nan``,
        so one exhausted seed hides fifteen good ones; dropping them without
        saying so reports a mean over a subset while the header claims the full
        seed count, which is the failure this whole line exists to avoid making
        elsewhere.
        """
        metric = self.regret if self.regret.size else -self.best
        label = "regret" if self.regret.size else "-best"
        usable = metric[np.isfinite(metric)]
        error = usable.std(ddof=1) / np.sqrt(usable.size) if usable.size > 1 else 0.0
        mean = usable.mean() if usable.size else float("nan")
        spend = f"{self.spent.mean():.0f}/{self.budget}"
        flag = " (underspent)" if self.underspent else ""
        failed = f" ({self.failed} exhausted)" if self.failed else ""
        return (
            f"{self.name:<24}{label} {mean:>7.3f} +/- {error:<6.3f}"
            f"div {self.diversity.mean():>5.2f}  spent {spend}{flag}{failed}"
        )


@dataclass
class BenchmarkResult:
    """Every arm, plus the comparisons between them.

    Attributes:
        protocol: The design everything was run under.
        arms: Results by arm name, in the order they were declared.
        seeds: Which seeds were used.
    """

    protocol: Protocol
    arms: dict[str, ArmResult] = field(default_factory=dict)
    seeds: tuple[int, ...] = ()

    def against(self, reference: str, *, metric: str = "regret") -> list[PairedComparison]:
        """Compare every arm against one of them, paired across seeds.

        A seed on which either arm exhausted is **not** dropped here, unlike in
        [ArmResult.summary][evogfn.benchmark.harness.ArmResult.summary], and the
        difference is deliberate: this comparison is paired, so dropping a seed
        from one arm would silently unpair the rest and compare two different
        seed sets. The ``nan`` propagates instead -- through the mean and through
        the interval, so the comparison reports itself as unresolvable rather
        than as a win for either side. Reading it as a null would be a mistake,
        which is why the arm's own line carries the exhausted count.

        Args:
            reference: Name of the arm to compare against.
            metric: ``"regret"``, ``"best"`` or ``"diversity"``.

        Returns:
            One comparison per other arm, best first.

        Raises:
            KeyError: If the reference arm was not run.
            ValueError: If the metric is unknown, or regret was requested from
                a landscape that does not know its optimum.
        """
        if reference not in self.arms:
            raise KeyError(f"no arm named {reference!r}; have {sorted(self.arms)}")
        if metric not in {"regret", "best", "diversity"}:
            raise ValueError(f"unknown metric {metric!r}")

        base = getattr(self.arms[reference], metric)
        if base.size == 0:
            raise ValueError(f"{metric} is unavailable; this landscape does not know its optimum")
        # Regret is a loss, so a smaller value is the better result.
        higher_is_better = metric != "regret"
        comparisons = [
            compare(
                f"{name} vs {reference}",
                getattr(arm, metric),
                base,
                higher_is_better=higher_is_better,
            )
            for name, arm in self.arms.items()
            if name != reference
        ]
        return sorted(comparisons, key=lambda c: -c.mean)

    def report(self, *, reference: str | None = None, metric: str = "regret") -> str:
        """A human-readable summary, including what did *not* separate.

        Args:
            reference: Arm to compare the others against. Defaults to the first.
            metric: Which metric to compare on.

        Returns:
            A multi-line report.
        """
        lines = [
            f"{self.protocol!r}  ({len(self.seeds)} seeds)",
            "-" * 72,
            *(arm.summary() for arm in self.arms.values()),
        ]
        if len(self.arms) > 1:
            base = reference or next(iter(self.arms))
            lines += ["", f"paired on {metric}, positive favours the first:"]
            for comparison in self.against(base, metric=metric):
                lines.append(f"  {comparison!r}")
                if not comparison.significant:
                    needed = seeds_needed(comparison)
                    lines.append(
                        f"      inconclusive; ~{needed} seeds would resolve an effect this size"
                        if needed
                        else "      no effect to resolve"
                    )
        return "\n".join(lines)


def run_benchmark(
    landscape: FitnessLandscape,
    arms: Mapping[str, ArmFactory],
    protocol: Protocol,
    *,
    seeds: Sequence[int] = tuple(range(30)),
) -> BenchmarkResult:
    """Run every arm on every seed under one protocol.

    Args:
        landscape: The oracle every arm is scored against.
        arms: Named factories, each building a campaign from a seed and the
            protocol.
        protocol: Rounds and batch size, shared by every arm.
        seeds: Seeds to run. The same seed reaches every arm, which is what
            makes the paired comparison valid.

    Returns:
        Per-arm results and the comparisons between them.

    Raises:
        ValueError: If no arms were given, or fewer than two seeds -- a
            comparison from one seed has no variance to speak of.
    """
    if not arms:
        raise ValueError("no arms to run")
    if len(seeds) < 2:  # noqa: PLR2004 - a variance needs two observations
        raise ValueError(f"need at least 2 seeds, got {len(seeds)}")

    optimum = landscape.optimum
    best_possible = float(np.max(optimum)) if optimum is not None else None

    result = BenchmarkResult(protocol=protocol, seeds=tuple(seeds))
    for name, build in arms.items():
        best, spread, spent = [], [], []
        for seed in seeds:
            campaign = build(seed, protocol)
            try:
                run = campaign.run()
            except RuntimeError:
                # A sampler that cannot fill its plate raises rather than
                # quietly measuring fewer designs, which is right: an arm
                # compared at less than the stated budget is not being compared
                # at the budget the table claims. Recorded here rather than
                # propagated, because one exhausted arm must not discard the
                # seeds every other arm has already spent -- the seed is scored
                # as spending nothing, which `underspent` then surfaces.
                #
                # `nan` rather than a worst-case sentinel: the arm has no best
                # value here, which is a different statement from having an
                # infinitely bad one, and an infinity propagates into the
                # spread as `inf - inf`.
                best.append(np.nan)
                spread.append(0.0)
                spent.append(0)
                continue
            best.append(run.best_value)
            spread.append(diversity(run.sequences) if len(run.sequences) > 1 else 0.0)
            spent.append(run.oracle_calls)
        best_array = np.asarray(best, dtype=np.float64)
        result.arms[name] = ArmResult(
            name=name,
            best=best_array,
            regret=(
                best_possible - best_array
                if best_possible is not None
                else np.zeros(0, dtype=np.float64)
            ),
            diversity=np.asarray(spread, dtype=np.float64),
            spent=np.asarray(spent, dtype=np.int64),
            budget=protocol.budget,
        )
    return result
