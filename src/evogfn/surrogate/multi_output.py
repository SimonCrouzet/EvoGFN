r"""A surrogate that keeps the objective vector alive, and the landscape view of it.

Why this exists is a structural fact rather than a preference. A campaign fits
its surrogate to ``self._acquisition.reduce_objectives(...)``, which for a
[ScalarizedAcquisition][evogfn.acquisition.rules.ScalarizedAcquisition] is
already scalarised -- so the proxy a policy trains against has no objective
vector left in it, and *every* fixed-preference arm is therefore searching a
single-objective landscape. That reduction is why the multi-objective content of
a scalarised suite lives in how many preferences there are rather than inside any
one campaign.

MOGFN-PC breaks the reduction for itself by applying ``ω`` at **reward** time,
which it can only do if the vector is still there. So it needs a surrogate that
predicts per objective and a proxy that presents one as a landscape. Neither
exists: [DeepEnsemble][evogfn.surrogate.ensemble.DeepEnsemble] is single-output
in three separate places, and
[ProxyLandscape][evogfn.surrogate.proxy.ProxyLandscape] hard-codes
``reshape(-1, 1)`` and never overrides ``n_objectives``.

Composed rather than merged, and what that costs
-------------------------------------------------

The statistically better answer is an ``n_outputs`` argument on ``DeepEnsemble``:
one shared trunk per member with ``k`` output units, so the objectives share a
representation and correlated objectives -- CH65's three affinities, which are
correlated by the biology the assay measures -- help each other. That is a
**global** change: ``surrogate/ensemble.py`` is in the dependency closure of both
benchmark suites, and the fingerprint is over raw file bytes, so touching it
invalidates every stored single-objective record. Composing ``k`` ensembles here
changes no existing file.

Three costs, stated rather than left to be found:

* **``k`` times the fitting.** At five members and 150 epochs, three objectives
  is fifteen networks per round instead of five. Measurable, not prohibitive at
  384 assays -- and nobody has timed it, because no multi-objective campaign has
  run.
* **The objectives are modelled as independent.** Exactly the correlation
  structure CH65 is interesting for is thrown away.
* **It is inconsistent with the acquisition layer's own spread heuristic.**
  ``ScalarizedAcquisition`` takes ``s(mean + std) - s(mean)``, which for a
  weighted sum is ``sum_i w_i std_i`` -- the *perfectly correlated* case. Independent
  per-objective spreads and that heuristic cannot both be right. The direction of
  the error is toward overstating uncertainty, which errs toward exploration
  rather than toward confident nonsense, but it is a wart and not a design.

Merging into ``DeepEnsemble`` is the right end state and should be a deliberate,
batched edit once nothing is in flight.

The trap this module cannot protect against
--------------------------------------------

An ``(n, k)`` prediction must **never** reach ``Campaign(surrogate=...)``.
``Campaign._design`` passes the surrogate's mean straight into ``_correlation``,
which builds ``isfinite(predicted) & isfinite(measured)`` against an ``(n,)``
``measured``; the ``(n, k)`` broadcasts and the indexing then raises -- inside the
round record, after a plate has been measured. A campaign that wants this model
either runs with ``surrogate=None`` and lets the sampler rank its own output, or
is handed a thin adapter returning the ``(n,)`` scalarised view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from evogfn.landscapes.base import FitnessLandscape
from evogfn.surrogate.base import Surrogate
from evogfn.surrogate.ensemble import DEFAULT_MEMBERS, DeepEnsemble

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch

    from evogfn.core.types import Alphabet, Fitness, Tokens

# An objective matrix is two-dimensional by definition.
_MATRIX_NDIM = 2


class MultiOutputEnsemble(Surrogate):
    """One deep ensemble per objective, behind the single-surrogate interface.

    See the module docstring for why this composes rather than extends, and what
    modelling the objectives independently costs.

    Args:
        n_tokens: Alphabet size.
        sequence_length: Number of positions.
        n_objectives: How many objectives to predict. Must be at least 1;
            a one-objective instance is legal and is simply a `DeepEnsemble`
            returning a column, which is what makes the shape uniform.
        n_members: Networks per objective.
        hidden_dim: Width of each member.
        n_layers: Hidden layers per member.
        epochs: Training passes per fit, per objective.
        learning_rate: Optimiser rate.
        bootstrap: Draw each member's training set with replacement.
        seed: Seeds every objective's ensemble, mixed so that no two objectives
            share a network.
        device: Where to train.

    Raises:
        ValueError: If any size is not positive.
    """

    def __init__(  # noqa: PLR0913 - the ensemble's shape is its definition
        self,
        *,
        n_tokens: int,
        sequence_length: int,
        n_objectives: int,
        n_members: int = DEFAULT_MEMBERS,
        hidden_dim: int = 128,
        n_layers: int = 2,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        bootstrap: bool = False,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Build one ensemble per objective."""
        if n_objectives < 1:
            raise ValueError(f"n_objectives must be at least 1, got {n_objectives}")
        self._n_objectives = n_objectives
        self._length = sequence_length
        self._members = tuple(
            DeepEnsemble(
                n_tokens=n_tokens,
                sequence_length=sequence_length,
                n_members=n_members,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                epochs=epochs,
                learning_rate=learning_rate,
                bootstrap=bootstrap,
                # Mixed rather than `seed + objective`. `DeepEnsemble` already
                # offsets its own members by `seed + index`, so a plain sum makes
                # objective 0's second member and objective 1's first member the
                # identical network -- and their spreads then correlate for a
                # reason that is not the data.
                seed=int(np.random.SeedSequence([seed, objective]).generate_state(1)[0]),
                device=device,
            )
            for objective in range(n_objectives)
        )

    @property
    def n_objectives(self) -> int:
        """How many objectives this surrogate predicts."""
        return self._n_objectives

    @property
    def members(self) -> tuple[DeepEnsemble, ...]:
        """The per-objective ensembles, in objective order, for inspection."""
        return self._members

    @property
    def is_fitted(self) -> bool:
        """Whether *every* objective has been fitted.

        One flag because the proxy reads one flag. A partially fitted model asked
        to predict would return one trained column beside one initialisation, and
        nothing in the returned array would say which was which.
        """
        return all(member.is_fitted for member in self._members)

    def fit(self, sequences: Tokens, values: Fitness) -> None:
        """Train each objective's ensemble on that objective's column.

        Args:
            sequences: An ``(n, length)`` array of measured sequences.
            values: An ``(n, n_objectives)`` array of measured values. The raw
                objective matrix, never a reduction of it -- fitting this model
                to ``reduce_objectives(...)`` would put the preference back
                upstream of the reward, which is the one thing the arm using it
                exists not to do.

        Raises:
            ValueError: If ``values`` is not an ``(n, n_objectives)`` matrix of
                this surrogate's width, or if some objective has no finite
                measurement to fit. The objective is named, because the
                underlying ensemble's message says only that nothing was finite.
        """
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != _MATRIX_NDIM or matrix.shape[1] != self._n_objectives:
            raise ValueError(
                f"expected values of shape (n, {self._n_objectives}) covering "
                f"{self._n_objectives} objectives, got {matrix.shape}; a single column here "
                f"means the objectives were reduced upstream, and this model exists to be "
                f"fitted before that happens"
            )
        for objective, member in enumerate(self._members):
            column = matrix[:, objective : objective + 1]
            if not np.isfinite(column).any():
                raise ValueError(
                    f"objective {objective} has no finite measurement among "
                    f"{matrix.shape[0]} designs, so there is nothing to fit it to"
                )
            member.fit(sequences, column)

    def predict(
        self, sequences: Tokens
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Predict every objective, with each one's ensemble disagreement.

        Args:
            sequences: An ``(n, length)`` array to score.

        Returns:
            Means ``(n, n_objectives)`` and standard deviations
            ``(n, n_objectives)``, on the original scale of the measured values.

        Raises:
            RuntimeError: If any objective has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("the surrogate has not been fitted; call fit() first")
        predicted = [member.predict(sequences) for member in self._members]
        mean = np.column_stack([values for values, _ in predicted])
        spread = np.column_stack([deviation for _, deviation in predicted])
        return mean, spread

    def __repr__(self) -> str:
        """Name the surrogate and how many objectives it models."""
        return f"MultiOutputEnsemble({self._n_objectives} objectives)"


class MultiObjectiveProxy(FitnessLandscape):
    """Presents a multi-output surrogate as a vector-valued landscape.

    The twin of [ProxyLandscape][evogfn.surrogate.proxy.ProxyLandscape], and a
    separate class rather than a generalisation of it for the reason the module
    docstring gives. The difference that matters is one line: ``ProxyLandscape``
    ends ``combined.reshape(-1, 1)`` and inherits ``n_objectives = 1``, so it is a
    single-objective landscape by declaration as well as by shape.

    Like its twin it holds the surrogate **by reference**, so it is a live view:
    built once at the start of a campaign, it sees each round's refitted model
    without anyone handing it a new one.

    Args:
        surrogate: The model to score with.
        alphabet: The alphabet sequences are written in.
        sequence_length: Length of the sequences being scored.
        optimism: Weight on the surrogate's uncertainty, applied per objective.
            Zero scores the mean, which is the comparison to run first: it
            isolates what preference conditioning contributes from what an
            exploration bonus contributes.

    Raises:
        ValueError: If ``optimism`` is negative.
    """

    def __init__(
        self,
        surrogate: MultiOutputEnsemble,
        *,
        alphabet: Alphabet,
        sequence_length: int,
        optimism: float = 0.0,
    ) -> None:
        """Wrap the surrogate without copying it."""
        if optimism < 0:
            raise ValueError(f"optimism must be non-negative, got {optimism}")
        self._surrogate = surrogate
        self._alphabet = alphabet
        self._length = sequence_length
        self._optimism = optimism

    @property
    def alphabet(self) -> Alphabet:
        """The alphabet sequences are written in."""
        return self._alphabet

    @property
    def sequence_length(self) -> int:
        """Length of every sequence this proxy scores."""
        return self._length

    @property
    def n_objectives(self) -> int:
        """How many objectives the wrapped surrogate predicts.

        Overridden, and this is the whole point of the class. The base class
        checks ``_evaluate``'s returned shape against this, so a proxy inheriting
        the default of 1 would refuse its own output -- the good failure. The bad
        one is declaring the objectives and returning a scalarised column, which
        this cannot do because both come from the same surrogate.
        """
        return self._surrogate.n_objectives

    @property
    def surrogate(self) -> MultiOutputEnsemble:
        """The model being used as the reward, for inspection."""
        return self._surrogate

    @property
    def is_ready(self) -> bool:
        """Whether the underlying surrogate has been fitted.

        A campaign's first round has nothing to fit on, so a sampler checks this
        rather than assuming the proxy can be trained against.
        """
        return self._surrogate.is_fitted

    def _evaluate(self, sequences: Tokens) -> Fitness:
        """Score every objective, optionally made optimistic.

        Raises:
            RuntimeError: If the surrogate has not been fitted. Returning an
                unfitted network's output would look like a reward signal and
                would train the policy toward its initialisation.
        """
        mean, spread = self._surrogate.predict(sequences)
        combined = np.asarray(mean, dtype=np.float64)
        if self._optimism:
            combined = combined + self._optimism * np.asarray(spread, dtype=np.float64)
        return combined

    def __repr__(self) -> str:
        """Name the proxy and its optimism."""
        return f"MultiObjectiveProxy({self._surrogate!r}, optimism={self._optimism})"
