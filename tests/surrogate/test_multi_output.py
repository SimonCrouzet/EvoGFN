"""Tests for the multi-output surrogate and the proxy that presents it as a landscape.

The component exists because of a structural fact rather than a preference: the
campaign fits its own surrogate to `acquisition.reduce_objectives(...)`, which is
already scalarised, so the proxy a policy trains against has no objective vector
left in it. A preference-conditioned arm applies omega at reward time and
therefore needs the vector to still be there.

Three failures are under test.

**A prediction of the wrong width.** `(n, k)` where `(n,)` is expected reaches
`Campaign._correlation` and raises deep inside the ledger, *after* a plate has
been measured -- and `(n,)` where `(n, k)` is expected reaches
`ScalarizedReward` and applies a k-entry preference to a scalar.

**Objectives fitted to each other's measurements.** One ensemble per objective
means one column each, and an off-by-one in the slicing produces a model that is
confidently wrong rather than one that raises.

**A proxy that declares one objective while returning several.** The landscape
base class checks the returned shape against `n_objectives`, so a proxy that
inherits the default of 1 refuses its own output -- which is the good failure.
The bad one is declaring `k` and returning a scalarised column.
"""

import numpy as np
import pytest

from evogfn.core.types import Alphabet
from evogfn.surrogate.multi_output import MultiObjectiveProxy, MultiOutputEnsemble

LENGTH = 4
TOKENS = 3
ALPHABET = Alphabet.from_string("ACG")


def observations(n=24, n_objectives=2, seed=0):
    """Sequences and objective values with a different signal in each column."""
    rng = np.random.default_rng(seed)
    sequences = rng.integers(0, TOKENS, size=(n, LENGTH))
    # Objective 0 rewards low tokens, objective 1 rewards high ones, so a model
    # that swapped the columns would be visibly anti-correlated rather than
    # merely worse.
    values = np.column_stack(
        [
            (TOKENS - 1 - sequences).sum(axis=1).astype(np.float64),
            sequences.sum(axis=1).astype(np.float64),
        ][:n_objectives]
        + [sequences[:, 0].astype(np.float64) for _ in range(max(n_objectives - 2, 0))]
    )
    return sequences, values


def build(n_objectives=2, seed=0, epochs=20):
    """A small multi-output ensemble, cheap enough to fit on every commit."""
    return MultiOutputEnsemble(
        n_tokens=TOKENS,
        sequence_length=LENGTH,
        n_objectives=n_objectives,
        n_members=2,
        hidden_dim=16,
        epochs=epochs,
        seed=seed,
    )


class TestTheShapeEverythingDownstreamAssumes:
    def test_predict_returns_one_value_and_one_spread_per_objective(self):
        sequences, values = observations()
        surrogate = build()
        surrogate.fit(sequences, values)
        mean, spread = surrogate.predict(sequences)
        assert mean.shape == (sequences.shape[0], 2)
        assert spread.shape == (sequences.shape[0], 2)
        assert np.isfinite(mean).all()
        assert (spread >= 0).all()

    def test_it_is_unfitted_until_every_objective_has_been_fitted(self):
        # Reported as one flag because the proxy reads one flag. A partially
        # fitted ensemble asked to predict would return one trained column and
        # one initialisation, and nothing in the array would say which.
        surrogate = build()
        assert not surrogate.is_fitted
        sequences, values = observations()
        surrogate.fit(sequences, values)
        assert surrogate.is_fitted

    def test_predicting_before_fitting_raises_rather_than_returning_noise(self):
        sequences, _ = observations()
        with pytest.raises(RuntimeError, match="not been fitted"):
            build().predict(sequences)

    def test_a_value_matrix_of_the_wrong_width_is_refused(self):
        sequences, values = observations(n_objectives=2)
        with pytest.raises(ValueError, match="3 objectives"):
            build(n_objectives=3).fit(sequences, values)

    def test_a_single_column_of_values_is_refused(self):
        # The failure this catches is the campaign's own reduction leaking in: a
        # caller that passed `reduce_objectives(...)` here would be fitting the
        # multi-output model to an already-scalarised target, which is the exact
        # thing this class exists to avoid, and the shapes would otherwise
        # broadcast quietly.
        sequences, values = observations()
        with pytest.raises(ValueError, match="objectives"):
            build().fit(sequences, values[:, :1])


class TestEachObjectiveIsModelledFromItsOwnColumn:
    def test_the_columns_are_not_swapped(self):
        # An off-by-one in the slicing gives a model that predicts objective 1
        # where objective 0 was asked for. It raises nothing, and the campaign's
        # ranking is then confidently backwards on both.
        sequences, values = observations(n=48, seed=1)
        # Enough epochs that a *correctly* wired model clears the bar; the
        # failure being caught is a swap, which shows up as a negative
        # correlation and would fail at any threshold.
        surrogate = build(seed=2, epochs=200)
        surrogate.fit(sequences, values)
        mean, _ = surrogate.predict(sequences)
        for column in range(2):
            assert np.corrcoef(mean[:, column], values[:, column])[0, 1] > 0.5

    def test_the_objectives_get_different_initialisations(self):
        # Deriving every objective's seed as `seed + j` collides with the
        # per-member offset inside each ensemble, so two objectives would share
        # networks and their spreads would be correlated for a reason that is not
        # the data.
        sequences, _ = observations()
        surrogate = build(n_objectives=2, seed=0)
        surrogate.fit(sequences, np.column_stack([np.ones(24), np.ones(24)]))
        left, right = surrogate.members
        assert left is not right

    def test_an_objective_with_nothing_finite_is_named_in_the_refusal(self):
        # On a masked landscape a whole column can be -inf. The underlying
        # ensemble refuses that, and without this the message says only "no
        # observation is finite" with no way to tell which objective.
        sequences, values = observations()
        values[:, 1] = -np.inf
        with pytest.raises(ValueError, match="objective 1"):
            build().fit(sequences, values)


class TestTheProxyPresentsItAsALandscape:
    def test_it_declares_the_objectives_it_returns(self):
        # `FitnessLandscape.evaluate` checks the returned shape against
        # `n_objectives`, which defaults to 1. A proxy that did not override it
        # would refuse its own output -- and one that overrode it while returning
        # a scalarised column would not.
        surrogate = build(n_objectives=2)
        sequences, values = observations()
        surrogate.fit(sequences, values)
        proxy = MultiObjectiveProxy(surrogate, alphabet=ALPHABET, sequence_length=LENGTH)
        assert proxy.n_objectives == 2
        assert proxy.evaluate(sequences).shape == (sequences.shape[0], 2)

    def test_it_says_whether_it_can_be_trained_against_yet(self):
        # A campaign's first round has nothing to fit on, so the sampler asks
        # this rather than assuming. Training against an unfitted ensemble would
        # optimise the policy toward its initialisation.
        surrogate = build()
        proxy = MultiObjectiveProxy(surrogate, alphabet=ALPHABET, sequence_length=LENGTH)
        assert not proxy.is_ready
        sequences, values = observations()
        surrogate.fit(sequences, values)
        assert proxy.is_ready

    def test_it_is_a_live_view_of_the_surrogate_rather_than_a_snapshot(self):
        # The property the sampler relies on: it is built once and sees each
        # round's refitted model without anyone handing it a new one.
        surrogate = build()
        proxy = MultiObjectiveProxy(surrogate, alphabet=ALPHABET, sequence_length=LENGTH)
        sequences, values = observations()
        surrogate.fit(sequences, values)
        first = proxy.evaluate(sequences)
        surrogate.fit(sequences, values * 10.0)
        assert not np.allclose(first, proxy.evaluate(sequences))

    def test_optimism_raises_the_prediction_by_the_spread(self):
        surrogate = build()
        sequences, values = observations()
        surrogate.fit(sequences, values)
        mean, spread = surrogate.predict(sequences)
        proxy = MultiObjectiveProxy(
            surrogate, alphabet=ALPHABET, sequence_length=LENGTH, optimism=2.0
        )
        np.testing.assert_allclose(proxy.evaluate(sequences), mean + 2.0 * spread)

    def test_a_negative_optimism_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            MultiObjectiveProxy(build(), alphabet=ALPHABET, sequence_length=LENGTH, optimism=-1.0)
