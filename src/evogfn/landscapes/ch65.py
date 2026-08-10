r"""The CH65 antibody landscape: three antigens, one Boolean lattice.

Phillips, Maurer, Brooks, Dupic, Schmidt and Desai, *Hierarchical sequence-affinity
landscapes shape the evolution of breadth in an anti-influenza receptor binding site
antibody* (*eLife* 12:e83628, 2023) measured the dissociation constant of the
broadly neutralising human antibody CH65 against three H1 influenza antigens, for
every combination of the **16 somatic mutations** that separate the mature antibody
from its inferred unmutated common ancestor. That is $2^{16} = 65{,}536$ variants,
of which **62,926 (96%) cleared quality control on all three antigens**.

It is the only genuinely multi-objective landscape in this package, and the
trade-off is the paper's own subject: the three objectives are the same antibody
assayed against a 1990 strain, an escape mutant of it, and a 2006 strain, and
affinity maturation towards one costs affinity towards another.

## The space is a Boolean subset lattice

Every one of the 16 sites is **binary**: germline residue or mature residue. The
alphabet has size 2, and a variant is a *subset* of applied somatic mutations.

That is exactly the graph
[MutationEnvironment][evogfn.env.mutation.MutationEnvironment] already models. In
that environment a position may be mutated at most once, which on a binary
alphabet means the reachable set from the germline is the full subset lattice and
nothing is masked away artificially. On GB1 and TrpB the same rule is a modelling
restriction imposed on a 20-letter space; here it is the assay's own structure.

The 16 mutations, in the order the authors index them, are six in the light chain
(`N26D`, `S29R`, `Y35N`, `Y48C`, `D49Y`, `V98I`) and ten in the heavy chain
(`G31D`, `Y33H`, `M34I`, `H35N`, `N52H`, `G57D`, `L83V`, `S84N`, `R85G`, `R87K`).
Two further VH mutations of the real antibody, `Q1E` and `S75A`, were left out of
the library by the authors as having no measurable effect on affinity, so this is
a 16-site library of an 18-mutation antibody.

## Three antigens, three objectives

Affinity was measured by Tite-Seq in biological duplicate against:

| Objective | Antigen | Meaning |
| --- | --- | --- |
| `affinity_MA90` | A/Massachusetts/1/1990 | the strain CH65 was raised against |
| `affinity_MA90_G189E` | MA90 plus G189E | a receptor-binding-site escape variant |
| `affinity_SI06` | A/Solomon Islands/3/2006 | a strain 16 years later, the breadth test |

Each value is the replicate mean of

$$f_a(x) = -\log_{10} K_D(x, a)$$

with $K_D$ in molar, so **higher is better** and the package's maximisation
convention holds without a sign flip. Values run from 6.0 to 10.53, four orders
of magnitude in affinity.

## Left-censoring, stated per objective

Tite-Seq has a detection floor. A variant whose affinity is weaker than
$K_D = 10^{-6}\,\mathrm{M}$ produces no titration curve to fit, and the authors
pin it to the boundary at exactly `6.0`. **That number is a bound, not a
measurement**: two variants both reported at 6.0 may differ by any amount below
the floor.

Of the 62,926 retained variants, the fraction reported at the floor is:

| Objective | Censored | Fraction |
| --- | --- | --- |
| `affinity_MA90` | 0 | **0.00%** |
| `affinity_MA90_G189E` | 9,451 | **15.02%** |
| `affinity_SI06` | 29,847 | **47.43%** |

This class does **not** impute, drop or repair those values. It reports them as
published and exposes
[CH65Landscape.is_censored][evogfn.landscapes.ch65.CH65Landscape.is_censored],
which answers per variant *and per objective*, so an analysis can exclude them,
fit a censored likelihood, or knowingly ignore the issue -- but cannot do the
last by accident.

The per-objective breakdown matters. The sibling CR9114 library from the same lab
is unusable for multi-objective work because its second and third objectives are
~97% censored, which makes them constant rather than informative; a single pooled
figure would hide that. Here MA90 is uncensored and SI06 is censored for nearly
half the library, so any conclusion resting on SI06 ordering within the censored
half is an artefact.

There is no right-censoring: the strongest measured affinities are isolated points
(10.53, 10.33, 9.76) with no pile-up at a ceiling.

## The wild type is the germline ancestor

[CH65Landscape.wild_type][evogfn.landscapes.ch65.CH65Landscape.wild_type] is the
**germline** ancestor, all 16 sites unmutated, not the mature antibody. It is the
sequence evolution started from; starting at the mature antibody would make every
action in the environment a reversion, which is a different experiment.

The germline binds MA90 at 8.55 and is **at the detection floor on both other
antigens**. The mature antibody reaches (10.10, 9.75, 9.35), so the library spans
the entire acquisition of breadth.

The mature antibody is available as
[CH65Landscape.mature][evogfn.landscapes.ch65.CH65Landscape.mature], and it is
**not Pareto-optimal**: 20 library variants form the non-dominated set and CH65
itself is dominated by several of them.

## The optimum here is an ideal point, not a front

[CH65Landscape.optimum][evogfn.landscapes.ch65.CH65Landscape.optimum] returns the
per-objective maxima, `(10.526, 10.331, 9.762)`. **No variant attains it** -- the
three maxima belong to three different sequences. It is the *ideal point* of
Miettinen (1999) §2.2, useful as a normaliser and as a reference for
[Tchebycheff][evogfn.rewards.scalarization.Tchebycheff], and meaningless as a
target to regret against. "How close did this run get" needs the indicators in
[evogfn.metrics.pareto][] -- hypervolume against a fixed reference, or IGD+
against the 20-point front.

## What this loader deliberately does not provide

The per-variant standard errors, the individual replicate values, the expression
measurements and the inferred epistatic coefficients are all in the deposit and
none are loaded here. The two that would change an analysis are the standard
errors (how much variation between neighbouring variants is measurement noise)
and the replicate values (from which a partially censored variant could be
identified). See ``notes/review/08-ch65-provenance.md``.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, Final

import numpy as np

from evogfn.core.types import Alphabet
from evogfn.data.cache import fetch
from evogfn.landscapes.base import FitnessLandscape

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens

#: The authors' own deposit, pinned to the final commit rather than to a branch
#: so the URL cannot start serving different bytes. MIT licensed. This is the
#: file the paper's own figures are drawn from -- ``QCfilt`` is the quality
#: filter, ``REPfilt`` the between-replicate agreement filter -- so no
#: reprocessing stands between the published numbers and these.
CH65_URL = (
    "https://raw.githubusercontent.com/amphilli/CH65-comblib/"
    "cea336dfd05fa31d675f79038428bf7f0d177e78/Kd_Inference/results_CH65/"
    "Kd_processed/20221008_CH65_QCfilt_REPfilt.csv"
)

#: Pinned so that a silently updated remote file fails loudly rather than
#: changing published numbers.
CH65_SHA256 = "d526633e53a0c84f6739109d123438db6cc25f1e1e277012082f23156607811f"

CH65_FILENAME = "ch65_20221008_QCfilt_REPfilt.csv"

#: Germline and mature residue, in that order, so token 0 is always the ancestral
#: state and a decoded variant is byte-for-byte the ``geno`` key of the source
#: file. Keeping that identity means a value can be checked against the deposit
#: without a translation step.
CH65_ALPHABET = "01"

#: Number of somatic mutations varied in the library.
CH65_N_SITES = 16

#: The 16 mutations, indexed as the authors index them: position ``i`` of a
#: variant string is ``CH65_MUTATIONS[i]``. The first six are light chain, the
#: last ten heavy chain.
CH65_MUTATIONS: Final = (
    "N26D",
    "S29R",
    "Y35N",
    "Y48C",
    "D49Y",
    "V98I",
    "G31D",
    "Y33H",
    "M34I",
    "H35N",
    "N52H",
    "G57D",
    "L83V",
    "S84N",
    "R85G",
    "R87K",
)

#: How many of the 16 mutations are in the light chain; the rest are heavy.
CH65_N_LIGHT_CHAIN = 6

#: The germline ancestor: no somatic mutation applied.
CH65_GERMLINE = "0" * CH65_N_SITES

#: The mature antibody: every somatic mutation applied.
CH65_MATURE = "1" * CH65_N_SITES

#: Variants clearing QC on all three antigens, of the 65,536 possible. The
#: publication's own figure, reproduced exactly by this loader.
CH65_N_MEASURED = 62_926

#: Rows in the source file. One combination is absent from it entirely, so this
#: is 65,535 rather than 65,536.
CH65_N_ROWS = 65_535

#: Antigen names, in objective order.
CH65_ANTIGENS: Final = ("MA90", "MA90_G189E", "SI06")

#: Per-antigen retained counts published in the paper. Checked on load: getting
#: the intersection right by luck while mis-parsing an individual column is
#: exactly the failure a single total would not catch.
CH65_N_RETAINED_PER_ANTIGEN: Final = (65_530, 63_840, 64_619)

#: Tite-Seq detection floor, as $-\log_{10} K_D$ with $K_D$ in molar. Values
#: reported at exactly this number are left-censored: the affinity is *at most*
#: $10^{-6}\,\mathrm{M}$, and how much less is unknown.
CH65_DETECTION_FLOOR = 6.0

#: Number of objectives. Named so the shape checks below do not read as magic.
CH65_N_OBJECTIVES = 3

# Columns holding the replicate-mean affinity, in objective order. The file names
# the escape variant by its substitution alone, so the mapping to the antigen
# names above is not the identity and is written out rather than derived.
_MEAN_COLUMNS: Final = ("MA90_mean", "G189E_mean", "SI06_mean")

# Values the source file uses for "this antigen has no usable measurement for
# this variant", as distinct from a measurement pinned to the detection floor.
_MISSING: Final = frozenset({"", "NA", "NaN", "nan"})


class CH65Landscape(FitnessLandscape):
    """Affinity of CH65 variants against three influenza antigens.

    Scoring is a table lookup over a 65,536-row array, so evaluation is free and
    the whole space can be enumerated -- which is what makes an exact Pareto
    front, an exact hypervolume and an exact target distribution available, as
    for the single-objective landscapes.

    Unlike them, [evaluate][evogfn.landscapes.base.FitnessLandscape.evaluate]
    returns `(n, 3)`. Nothing downstream has to branch on that: a
    [ScalarizedReward][evogfn.rewards.scalarization.ScalarizedReward] turns the
    vector into the scalar the sampler and the loss expect, and the objective
    vector survives for the metrics that need it.

    Args:
        unmeasured_value: Value returned on every objective for the 2,610
            combinations that did not clear QC on all three antigens. Defaults to
            `CH65_DETECTION_FLOOR`, **not** to `0.0` as on GB1 and TrpB: zero is
            off this scale entirely -- it would claim an affinity of 1 M -- and
            would manufacture a cliff the assay never measured. The floor instead
            says "indistinguishable from a non-binder", which is the weakest claim
            the data supports. Pass `float("nan")` to make their absence propagate
            visibly instead.
        force_download: Re-download the dataset even if a valid cached copy
            exists.

    Raises:
        ChecksumMismatchError: If the downloaded data does not match its pinned
            checksum.
        ValueError: If the file does not contain the expected rows, per-antigen
            counts, variant count or genotype format, any of which would mean it
            is not the dataset this class was built against.
    """

    def __init__(
        self,
        *,
        unmeasured_value: float = CH65_DETECTION_FLOOR,
        force_download: bool = False,
    ) -> None:
        """Load the landscape into a dense lookup table."""
        self._alphabet = Alphabet.from_string(CH65_ALPHABET)
        self._unmeasured_value = unmeasured_value

        path = fetch(CH65_URL, sha256=CH65_SHA256, filename=CH65_FILENAME, force=force_download)
        variants, values = _read_variants(path)

        size = self._alphabet.size**CH65_N_SITES
        self._table = np.full((size, CH65_N_OBJECTIVES), unmeasured_value, dtype=np.float64)
        self._measured = np.zeros(size, dtype=np.bool_)
        self._censored = np.zeros((size, CH65_N_OBJECTIVES), dtype=np.bool_)

        indices = self._flat_index(self._alphabet.encode_many(variants))
        self._table[indices] = values
        self._measured[indices] = True
        # Only measured rows can be censored. An unmeasured variant is not a
        # variant known to bind weakly, and collapsing the two would let the
        # imputation default silently inflate the censored count.
        self._censored[indices] = values <= CH65_DETECTION_FLOOR

        self._ideal = values.max(axis=0)
        self._best_variants = tuple(variants[int(i)] for i in values.argmax(axis=0))
        self._censored_fraction = tuple(
            float(f) for f in (values <= CH65_DETECTION_FLOOR).mean(axis=0)
        )

    @property
    def alphabet(self) -> Alphabet:
        """Two tokens: `0` germline, `1` mature. Every site is binary."""
        return self._alphabet

    @property
    def sequence_length(self) -> int:
        """Sixteen, one per somatic mutation in the library."""
        return CH65_N_SITES

    @property
    def n_objectives(self) -> int:
        """Three, one per antigen."""
        return CH65_N_OBJECTIVES

    @property
    def objective_names(self) -> tuple[str, ...]:
        """One name per antigen, each the negative log-molar dissociation constant."""
        return tuple(f"affinity_{antigen}" for antigen in CH65_ANTIGENS)

    @property
    def optimum(self) -> Fitness:
        """The **ideal point**: the best measured value on each antigen separately.

        Returns:
            A `(3,)` array, `(10.526, 10.331, 9.762)`.

        Note:
            No variant attains this -- the three maxima belong to three different
            sequences -- so it is not a design and the gap to it is not a regret.
            Use it to normalise objectives or as a
            [Tchebycheff][evogfn.rewards.scalarization.Tchebycheff] reference. For
            "how close did this run get", use the indicators in
            [evogfn.metrics.pareto][] against
            [non_dominated][evogfn.metrics.pareto.non_dominated] of the measured
            set.
        """
        return self._ideal.copy()

    @property
    def reference_point(self) -> Fitness:
        r"""The Tite-Seq detection floor, `(6, 6, 6)` -- where hypervolume starts.

        A reference point is a claim about the assay rather than a parameter of
        an experiment: it is the worst value worth counting on each objective,
        designs failing to beat it contribute no volume, and two hypervolumes
        taken from different points are not comparable. This landscape is the
        only object that holds the assay, so it is the only one entitled to
        state it -- and a campaign picks it up automatically through
        [StatesReferencePoint][evogfn.loop.campaign.StatesReferencePoint].

        `CH65_DETECTION_FLOOR` is the defensible choice and not merely a
        convenient one. It is where the titration stops resolving, so a variant
        at 6.0 on an antigen is indistinguishable from a non-binder and has
        genuinely contributed nothing on that objective. Pushing the point lower
        would not be the safe option: every method would then earn the same
        large constant box, and the differences a comparison is reading would
        shrink into it.

        Returns:
            A `(3,)` array of `CH65_DETECTION_FLOOR` on each antigen.
        """
        return np.full(CH65_N_OBJECTIVES, CH65_DETECTION_FLOOR, dtype=np.float64)

    @property
    def optimal_variants(self) -> tuple[str, ...]:
        """The best measured variant on each antigen, one per objective.

        Returns:
            Three genotype strings, in objective order. They are all distinct,
            which is the multi-objective conflict stated as concretely as it can
            be.
        """
        return self._best_variants

    @property
    def wild_type(self) -> Tokens:
        """The germline ancestor, all sites unmutated -- the sequence evolution started from.

        Deliberately not the mature antibody. This is the parent a campaign
        should build from; starting at the mature sequence would make every
        available action a reversion.
        """
        return self._alphabet.encode(CH65_GERMLINE)

    @property
    def mature(self) -> Tokens:
        """The mature CH65 antibody, all 16 somatic mutations applied.

        Scores `(10.10, 9.75, 9.35)`, and is dominated by other variants in the
        library -- it is not on the Pareto front.
        """
        return self._alphabet.encode(CH65_MATURE)

    @property
    def n_measured(self) -> int:
        """How many of the 65,536 combinations cleared QC on all three antigens."""
        return int(self._measured.sum())

    @property
    def censored_fraction(self) -> tuple[float, ...]:
        """Share of *measured* variants sitting at the detection floor, per objective.

        Returns:
            Three fractions in objective order: `(0.0, 0.1502, 0.4743)`. Reported
            per objective rather than pooled because that is the number which
            decides whether an objective carries information at all -- a pooled
            figure is what makes a degenerate objective look usable.
        """
        return self._censored_fraction

    def is_measured(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        """Report which sequences cleared QC on all three antigens.

        Args:
            sequences: An `(n, 16)` array of token indices.

        Returns:
            An `(n,)` boolean array. `False` means the returned values are the
            imputed `unmeasured_value` on every objective, not measurements.

        Raises:
            ValueError: If the input fails validation.
        """
        checked = self._validate(sequences)
        return self._measured[self._flat_index(checked)]

    def is_censored(self, sequences: Tokens) -> npt.NDArray[np.bool_]:
        r"""Report which measurements are at the detection floor rather than resolved.

        Args:
            sequences: An `(n, 16)` array of token indices.

        Returns:
            An `(n, 3)` boolean array, one column per objective. `True` means the
            reported affinity is an upper bound of $10^{-6}\,\mathrm{M}$ and the
            true value is unknown below it. Unmeasured variants are `False`
            throughout: their values are absent, which is a different failure.

        Raises:
            ValueError: If the input fails validation.
        """
        checked = self._validate(sequences)
        return self._censored[self._flat_index(checked)]

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Look up the three affinities for each sequence."""
        return self._table[self._flat_index(sequences)]

    def _flat_index(self, sequences: Tokens) -> npt.NDArray[np.intp]:
        """Collapse `(n, 16)` token indices to positions in the lookup table."""
        size = self._alphabet.size
        weights = size ** np.arange(CH65_N_SITES - 1, -1, -1)
        return np.asarray(sequences, dtype=np.intp) @ weights


def _read_variants(path: Path) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Extract genotypes and their three replicate-mean affinities.

    A row is kept only when all three antigens have a usable measurement, which
    is what the paper's 62,926 counts and what makes the objective vector
    complete for every variant this landscape reports as measured.

    Args:
        path: Path to the downloaded CSV.

    Returns:
        The 16-character genotype strings and an `(n, 3)` array of affinities in
        objective order.

    Raises:
        ValueError: If a genotype is not 16 binary characters, or if the row
            count, the per-antigen retained counts or the complete-row count
            differ from the published figures -- any of which would mean the file
            is not the dataset this landscape was built against.
    """
    variants: list[str] = []
    values: list[tuple[float, ...]] = []
    rows = 0
    per_antigen = [0, 0, 0]

    with path.open("rt", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            geno = row["geno"]
            if len(geno) != CH65_N_SITES or set(geno) - set(CH65_ALPHABET):
                raise ValueError(
                    f"expected genotypes of {CH65_N_SITES} binary characters, found {geno!r}"
                )
            present = [row[column] not in _MISSING for column in _MEAN_COLUMNS]
            for index, ok in enumerate(present):
                per_antigen[index] += ok
            if not all(present):
                continue
            variants.append(geno)
            values.append(tuple(float(row[column]) for column in _MEAN_COLUMNS))

    if rows != CH65_N_ROWS:
        raise ValueError(
            f"expected {CH65_N_ROWS} rows in the CH65 QC-filtered table, found {rows}; "
            f"the dataset is not the one this landscape was built against"
        )
    if tuple(per_antigen) != CH65_N_RETAINED_PER_ANTIGEN:
        raise ValueError(
            f"expected {CH65_N_RETAINED_PER_ANTIGEN} variants retained for "
            f"{CH65_ANTIGENS} respectively, found {tuple(per_antigen)}; the antigen "
            f"columns are not the ones this landscape was built against"
        )
    if len(variants) != CH65_N_MEASURED:
        raise ValueError(
            f"expected {CH65_N_MEASURED} variants measured on all three antigens, "
            f"found {len(variants)}; the published QC intersection could not be "
            f"reproduced from this file"
        )
    return variants, np.asarray(values, dtype=np.float64)
