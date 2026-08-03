"""A deep ensemble, which is how the uncertainty is obtained.

Jain et al. (ICML 2022) use an ensemble of five MLPs and take their disagreement
as epistemic uncertainty. It is not the most principled estimator available, but
it is the one the comparable literature uses, it needs no approximate inference,
and its failure mode is legible: where the members agree the model is confident,
where they diverge it is guessing.

Members differ only in initialisation and in the order they see the data. That
is enough. With a few hundred observations in a space of 10^14, the members are
unconstrained almost everywhere and diverge freely off the observed manifold --
which is exactly the signal wanted.

Bootstrapping is opt-in, and it is a different estimator
--------------------------------------------------------

``bootstrap`` draws each member's training set with replacement from the
measurements, so the spread between members reflects which *data* they saw and
not only which initialisation they started from. That is the frequentist reading
of an ensemble, and it is what an arm claiming its ensemble is calibrated needs.

It is off by default, and deliberately so: turning it on changes what the spread
*means*, so an arm that acquired it silently would be ranked on an uncertainty
estimate different from the one every arm it is tabled against was ranked on, and
nothing in the numbers would say which. An arm asks for it by name.

The term is ALDE's, describing the configuration they took to the bench --
one-hot encodings, a five-member DNN ensemble with bootstrapping, and Thompson
sampling. Resampling with replacement is the conventional reading of that word
rather than a mechanism they spell out, and it is the reading implemented here.

One-hot rather than learned embeddings
--------------------------------------

At these dataset sizes an embedding layer is more parameters to fit from data
that cannot support them. ALDE reports that protein-language-model encodings
gave no benefit over one-hot in this regime, so the simplest input is used.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from evogfn.surrogate.base import Surrogate

if TYPE_CHECKING:
    import numpy.typing as npt

    from evogfn.core.types import Fitness, Tokens

#: Ensemble size used by Jain et al.
DEFAULT_MEMBERS = 5


class DeepEnsemble(Surrogate):
    """An ensemble of MLPs whose disagreement estimates uncertainty.

    Args:
        n_tokens: Alphabet size.
        sequence_length: Number of positions.
        n_members: How many independently initialised networks.
        hidden_dim: Width of each member.
        n_layers: Hidden layers per member.
        epochs: Training passes per fit.
        learning_rate: Optimiser rate.
        bootstrap: Draw each member's training set with replacement, making the
            spread a resampling estimate rather than an initialisation one. Off
            by default; see the module docstring for why it is not a free
            improvement to be switched on everywhere.
        seed: Seeds initialisation, batch order and the resampling.
        device: Where to train.

    Raises:
        ValueError: If any size is not positive.
    """

    def __init__(  # noqa: PLR0913 - the ensemble's shape is its definition
        self,
        *,
        n_tokens: int,
        sequence_length: int,
        n_members: int = DEFAULT_MEMBERS,
        hidden_dim: int = 128,
        n_layers: int = 2,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        bootstrap: bool = False,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        """Build the ensemble members."""
        for name, value in [
            ("n_tokens", n_tokens),
            ("sequence_length", sequence_length),
            ("n_members", n_members),
            ("hidden_dim", hidden_dim),
            ("n_layers", n_layers),
            ("epochs", epochs),
        ]:
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")

        self._n_tokens = n_tokens
        self._length = sequence_length
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._bootstrap = bootstrap
        self._device = device
        self._fitted = False
        # Standardisation statistics, set at fit time. Fitness scales differ by
        # orders of magnitude between landscapes, and an unnormalised target
        # makes the learning rate landscape-specific.
        self._mean = 0.0
        self._scale = 1.0

        generator = torch.Generator().manual_seed(seed)
        self._members = nn.ModuleList(
            [self._build(hidden_dim, n_layers, generator) for _ in range(n_members)]
        ).to(device)
        self._seed = seed

    def _build(self, hidden_dim: int, n_layers: int, generator: torch.Generator) -> nn.Module:
        """One member, initialised from the shared generator."""
        layers: list[nn.Module] = []
        width = self._length * self._n_tokens
        for _ in range(n_layers):
            layers += [_seeded_linear(width, hidden_dim, generator), nn.ReLU()]
            width = hidden_dim
        # The output layer needs seeding too. Leaving it to global torch state
        # made two ensembles built from the same seed disagree, which is the
        # kind of irreproducibility that surfaces as unexplained run-to-run
        # variance rather than as an error.
        layers.append(_seeded_linear(width, 1, generator))
        return nn.Sequential(*layers)

    @property
    def is_fitted(self) -> bool:
        """Whether the ensemble has been trained."""
        return self._fitted

    @property
    def n_members(self) -> int:
        """Number of networks in the ensemble."""
        return len(self._members)

    @property
    def bootstraps(self) -> bool:
        """Whether members are fitted to resamples rather than to the same data.

        Worth reading off the object rather than off the arm's name: two
        campaigns whose ensembles differ here are ranked on uncertainties that
        mean different things, and the predictions alone cannot say which.
        """
        return self._bootstrap

    def fit(self, sequences: Tokens, values: Fitness) -> None:
        """Train every member on the accumulated measurements.

        Each member sees the data in a different order, which together with a
        different initialisation is what makes them disagree off-manifold. Under
        ``bootstrap`` each also sees a *different sample* of it, drawn with
        replacement, so the disagreement carries information about the data as
        well as about the initialisation.

        Args:
            sequences: An ``(n, length)`` array of measured sequences.
            values: An ``(n, n_objectives)`` array of measured values.

        Raises:
            ValueError: If there is nothing to fit, or the shapes disagree.
        """
        array = np.asarray(sequences)
        if array.shape[0] == 0:
            raise ValueError("cannot fit a surrogate with no observations")
        features = self._encode(array)
        targets = np.asarray(values, dtype=np.float64).reshape(-1)
        if features.shape[0] != targets.shape[0]:
            raise ValueError(f"got {features.shape[0]} sequences and {targets.shape[0]} values")

        # Infeasible designs carry no information about the response surface;
        # keeping them would drag every prediction toward -inf.
        finite = np.isfinite(targets)
        if not finite.any():
            raise ValueError("cannot fit a surrogate when no observation is finite")
        features, targets = features[finite], targets[finite]

        self._mean = float(targets.mean())
        self._scale = float(targets.std()) or 1.0
        standardised = (targets - self._mean) / self._scale

        x = torch.as_tensor(features, dtype=torch.float32, device=self._device)
        y = torch.as_tensor(standardised, dtype=torch.float32, device=self._device)[:, None]

        for index, member in enumerate(self._members):
            generator = torch.Generator().manual_seed(self._seed + index)
            # Drawn once per fit rather than per epoch: a member's bootstrap
            # sample is *its training set*, and redrawing it every pass would
            # make every member converge on the same full dataset and give back
            # exactly the spread the resampling is there to create.
            member_x, member_y = x, y
            if self._bootstrap:
                picks = torch.randint(0, x.shape[0], (x.shape[0],), generator=generator).to(
                    self._device
                )
                member_x, member_y = x[picks], y[picks]
            optimiser = torch.optim.Adam(member.parameters(), lr=self._learning_rate)
            for _ in range(self._epochs):
                order = torch.randperm(member_x.shape[0], generator=generator).to(self._device)
                predicted = member(member_x[order])
                loss = nn.functional.mse_loss(predicted, member_y[order])
                optimiser.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                optimiser.step()
        self._fitted = True

    def predict(
        self, sequences: Tokens
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Predict the mean and the ensemble's disagreement.

        Args:
            sequences: An ``(n, length)`` array to score.

        Returns:
            Means ``(n,)`` and standard deviations ``(n,)``, on the original
            scale of the measured values.

        Raises:
            RuntimeError: If called before ``fit``. An unfitted ensemble would
                return its initialisation, which looks like a prediction.
        """
        if not self._fitted:
            raise RuntimeError("the surrogate has not been fitted; call fit() first")
        x = torch.as_tensor(self._encode(sequences), dtype=torch.float32, device=self._device)
        with torch.no_grad():
            stacked = torch.stack([member(x).squeeze(-1) for member in self._members])
        predictions = stacked.cpu().numpy() * self._scale + self._mean
        # ddof=1: with five members the biased estimator understates the spread,
        # and understating uncertainty is the error that matters here.
        spread = (
            predictions.std(axis=0, ddof=1)
            if len(self._members) > 1
            else np.zeros(predictions.shape[1])
        )
        return predictions.mean(axis=0), spread

    def _encode(self, sequences: Tokens) -> npt.NDArray[np.float64]:
        """Flatten token indices into a one-hot design matrix."""
        array = np.asarray(sequences)
        if array.ndim != 2 or array.shape[1] != self._length:  # noqa: PLR2004
            raise ValueError(f"expected shape (n, {self._length}), got {tuple(array.shape)}")
        one_hot = np.zeros((array.shape[0], self._length, self._n_tokens), dtype=np.float64)
        rows = np.arange(array.shape[0])[:, None]
        positions = np.arange(self._length)[None, :]
        one_hot[rows, positions, array] = 1.0
        return one_hot.reshape(array.shape[0], -1)


def _seeded_linear(in_features: int, out_features: int, generator: torch.Generator) -> nn.Linear:
    """A linear layer whose parameters come from ``generator``, not global state."""
    linear = nn.Linear(in_features, out_features)
    with torch.no_grad():
        bound = 1.0 / math.sqrt(in_features)
        linear.weight.copy_(
            torch.empty_like(linear.weight).uniform_(-bound, bound, generator=generator)
        )
        linear.bias.copy_(
            torch.empty_like(linear.bias).uniform_(-bound, bound, generator=generator)
        )
    return linear
