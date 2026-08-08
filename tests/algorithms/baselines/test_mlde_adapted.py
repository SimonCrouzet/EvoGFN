"""Tests for our adapted MLDE: the training size, and that it is reached.

The arm this file guards exists because of a measurement: on the suite's
``feasibility`` task the mean feasible share of a random screen is 0.053, so a
384-assay campaign returns about 20 usable measurements against `mlde`'s
training size of 96 and `mlde-over-budget`'s 384. Neither handover happens,
both rows are random screens under a supervised method's name, and
`MLDE.is_fitted` is the only thing that says so.

Two failures are under test, and both are silent by construction.

**A training size that stops being reachable.** `ADAPTED_TRAINING_SIZE` is
derived from that feasible share and the plate budget, and nothing in the code
recomputes it. Raised to the mean of the usable count -- which is the round,
obvious-looking answer -- half the seeds would end unfitted and the arm would
report one row that is two methods; raised further it stops fitting at all and
becomes a third copy of the failure it was added to escape. So the derivation is
re-done here from the constants rather than the value being pinned, and the
assertion is on the property that made it 8.

**The two published arms moving.** They are the finding. An edit that lowered
`DEFAULT_TRAINING_SIZE` to make `mlde` fit on a constrained landscape would
delete the result and leave three arms answering the same question, and no test
downstream of this one would notice.
"""

from math import comb

import numpy as np
import pytest

from evogfn.algorithms.baselines.mlde import (
    _MIN_TRAINING,
    ADAPTED_TRAINING_SIZE,
    CAMPAIGN_PLATES,
    CONSTRAINED_FEASIBLE_FRACTION,
    DEFAULT_TRAINING_SIZE,
    MLDE,
    PUBLISHED_BATCH_SIZE,
    PUBLISHED_TRAINING_SIZE,
)
from evogfn.core import Alphabet
from evogfn.env.mutation import MutationEnvironment

#: Wells a campaign may screen before the last plate, which is the deadline the
#: training size is derived against: the handover is only tested when a plate is
#: proposed, so a size not reached by here designs nothing.
SCREENING_WELLS = PUBLISHED_BATCH_SIZE * (CAMPAIGN_PLATES - 1)

#: One-sided two-sigma. The reliability the derivation asks of the handover, and
#: the level at which "reached on almost every seed" stops being a judgement call.
TWO_SIGMA = 0.9772

#: One well in nineteen returns a number, which is 5.26% and the nearest simple
#: period to the measured 0.053. A period rather than a draw so that a failure
#: here is always the sampler's and never an unlucky seed.
FEASIBLE_PERIOD = 19


def make_env(length=8, symbols="ABCD", max_mutations=6):
    return MutationEnvironment(
        np.zeros(length, dtype=np.int32),
        Alphabet.from_string(symbols),
        max_mutations=max_mutations,
    )


def reached(training_size, wells=SCREENING_WELLS, share=CONSTRAINED_FEASIBLE_FRACTION):
    """Chance a random screen of ``wells`` returns ``training_size`` usable values.

    The usable count is binomial -- each well returns a number or does not -- and
    a campaign draws one sample from it rather than the mean, which is the whole
    reason the training size sits below that mean.
    """
    below = sum(
        comb(wells, k) * share**k * (1.0 - share) ** (wells - k) for k in range(training_size)
    )
    return 1.0 - below


def screen(sampler, plates=CAMPAIGN_PLATES, plate=PUBLISHED_BATCH_SIZE):
    """Run a campaign whose wells return a number one time in `FEASIBLE_PERIOD`.

    The stand-in for a constrained landscape, and the mechanism is the one
    `MLDE.observe` documents: an infeasible assay is charged and discarded, so
    the training set grows at the feasible share of the rate the budget shrinks.

    Returns:
        How many plates had been observed when the model took over, or ``None``
        if it never did.
    """
    well = 0
    fitted_after = None
    for plate_index in range(plates):
        proposals = sampler.propose(plate)
        values = np.full((len(proposals), 1), -np.inf)
        for row in range(len(proposals)):
            if (well + row) % FEASIBLE_PERIOD == 0:
                values[row, 0] = float(row)
        well += len(proposals)
        sampler.observe(proposals, values)
        if fitted_after is None and sampler.is_fitted:
            fitted_after = plate_index
    return fitted_after


class TestTheTrainingSizeIsDerivedAndNotPicked:
    def test_it_is_the_largest_size_a_constrained_screen_reliably_returns(self):
        # The derivation, re-done from the constants. One step either way is a
        # different arm: 9 drops below two-sigma, so a handful of seeds in every
        # hundred would end unfitted and the row would silently mix a fitted
        # campaign with a random screen -- which is the ambiguity this arm was
        # added to remove, reappearing inside the arm itself.
        assert reached(ADAPTED_TRAINING_SIZE) >= TWO_SIGMA
        assert reached(ADAPTED_TRAINING_SIZE + 1) < TWO_SIGMA

    def test_the_mean_of_the_usable_count_would_have_been_a_coin_flip(self):
        # Why the obvious answer is wrong, written down so nobody re-derives it.
        # 0.053 * 288 is 15.3 usable measurements, and a training size set there
        # is reached by about half the seeds -- the failure being avoided is not
        # rare, it is the median outcome.
        mean_usable = round(CONSTRAINED_FEASIBLE_FRACTION * SCREENING_WELLS)

        assert 0.4 < reached(mean_usable) < 0.6
        assert mean_usable > ADAPTED_TRAINING_SIZE

    def test_it_leaves_a_full_designed_plate(self):
        # The point of screening against three plates rather than four: a size
        # reached only in the last round is reached after the last proposal, so
        # the ensemble would fit and never design anything. That campaign spends
        # its whole budget on a random screen and reports `is_fitted` true, which
        # is worse than reporting it false.
        assert SCREENING_WELLS + PUBLISHED_BATCH_SIZE == PUBLISHED_BATCH_SIZE * CAMPAIGN_PLATES
        assert reached(ADAPTED_TRAINING_SIZE, wells=SCREENING_WELLS) >= TWO_SIGMA

    def test_it_is_clear_of_the_floor_where_a_fit_is_only_nominal(self):
        # A ridge on one point predicts that point everywhere, so a size near
        # `_MIN_TRAINING` would buy the appearance of a handover and no ranking
        # at all -- reporting nothing in a second way rather than reporting
        # something small.
        assert ADAPTED_TRAINING_SIZE >= 4 * _MIN_TRAINING

    def test_the_size_is_one_number_and_not_a_family(self):
        # `adapted` takes no feasibility argument on purpose. A training size
        # read off each landscape would be tuned on the task it is scored on,
        # and the family would beat any fixed member of itself by construction.
        env = make_env()
        assert MLDE.adapted(env)._training_size == ADAPTED_TRAINING_SIZE
        assert MLDE.adapted(env, seed=7)._training_size == ADAPTED_TRAINING_SIZE


class TestTheHandoverActuallyHappens:
    def test_the_adapted_sampler_fits_where_the_default_never_does(self):
        # The measurement the arm exists for, at the shape it runs: four plates
        # of 96, one well in nineteen returning a number. The default gathers
        # about 20 usable measurements of the 96 it is waiting for and screens at
        # random to the end; nothing in its spend, its plate count or its regret
        # says so.
        adapted = MLDE.adapted(make_env(), seed=0)
        default = MLDE(make_env(), seed=0)

        fitted_after = screen(adapted)

        assert screen(default) is None
        assert default.training_examples < DEFAULT_TRAINING_SIZE
        assert fitted_after is not None
        assert fitted_after <= CAMPAIGN_PLATES - 2, (
            "the handover has to leave a full plate for the model to design"
        )

    def test_the_fitted_model_ranks_rather_than_merely_existing(self):
        # `is_fitted` is a flag, and a flag is what a nominal fit would also set.
        # Eight points is a weak model and the docstring says so, but it must at
        # least separate the pool it is handed, or the arm answers its own
        # question with a random screen wearing a fitted arm's name.
        env = make_env(length=10, max_mutations=8)
        sampler = MLDE.adapted(env, seed=0)
        train = sampler.propose(ADAPTED_TRAINING_SIZE)
        sampler.observe(train, (np.asarray(train) == 1).sum(axis=1, keepdims=True).astype(float))

        predictions = sampler.predict(sampler.propose(64))

        assert sampler.is_fitted
        assert float(predictions.std()) > 0.0

    def test_it_says_it_is_ours_everywhere_it_names_itself(self):
        # Both strings leave the object: `name` is written into the campaign's
        # provenance as the sampler that produced the designs, and `budget_note`
        # is what a results table prints beside the row. An unmarked "MLDE" on
        # either would put a training size we chose under somebody else's
        # method's name, which no amount of docstring elsewhere undoes.
        sampler = MLDE.adapted(make_env())

        assert "adapted" in sampler.name
        assert "our adaptation" in sampler.budget_note
        assert "handicapped" not in sampler.budget_note
        assert MLDE(make_env()).name == "MLDE"
        assert "handicapped" in MLDE(make_env()).budget_note

    def test_only_the_classmethod_confers_the_label(self):
        # A caller who sets the same training size by hand has built a small
        # MLDE, not this arm, and must not inherit its disclosure -- the label
        # has to track the decision rather than the number, or it stops being
        # evidence of anything.
        by_hand = MLDE(make_env(), training_size=ADAPTED_TRAINING_SIZE)

        assert by_hand.name == "MLDE"
        assert "our adaptation" not in by_hand.budget_note

    def test_the_label_survives_a_moved_anchor(self):
        # `reanchored` rebuilds the sampler through the constructor, which does
        # not confer the label. Dropped there, the arm would report itself as
        # MLDE from the first move onward -- and on a re-anchoring task that is
        # nearly the whole campaign, with nothing in the record to say so.
        moved = MLDE.adapted(make_env(), seed=0).reanchored(make_env())

        assert "adapted" in moved.name

    def test_a_moved_anchor_does_not_restore_the_default_training_size(self):
        # `reanchored` rebuilds the sampler, and a rebuild that dropped the
        # adapted size would put the arm back in the screening stage for a
        # training set it can never gather -- on a re-anchoring task, at every
        # move, with the arm still named for the adaptation.
        sampler = MLDE.adapted(make_env(), seed=0)
        moved = sampler.reanchored(make_env())

        assert moved._training_size == ADAPTED_TRAINING_SIZE


class TestThePublishedArmsAreUntouched:
    def test_the_two_published_training_sizes_are_where_they_were(self):
        # The adaptation is a third configuration, not a repair of the other two.
        # Their failure to fit on a constrained landscape is the finding being
        # reported, and an edit that fixed it here would delete the result.
        assert DEFAULT_TRAINING_SIZE == 96
        assert PUBLISHED_TRAINING_SIZE == 384
        assert MLDE(make_env())._training_size == DEFAULT_TRAINING_SIZE
        assert MLDE.as_published(make_env())._training_size == PUBLISHED_TRAINING_SIZE

    def test_the_adapted_size_is_a_fraction_of_the_one_already_called_a_compression(self):
        # Scale, asserted rather than described: this is not a small deviation
        # from the shipped arm but an order of magnitude below it, and a reader
        # who mistook the row for MLDE would be reading a twelfth of a training
        # set as the published method.
        assert ADAPTED_TRAINING_SIZE * 10 < DEFAULT_TRAINING_SIZE
        assert MLDE.adapted(make_env()).runs_below_published_training_size

    @pytest.mark.parametrize(
        "sampler",
        [MLDE(make_env()), MLDE.as_published(make_env()), MLDE.adapted(make_env())],
    )
    def test_every_configuration_still_reports_its_own_budget(self, sampler):
        # `budget_note` is what a results table prints beside the row, and the
        # adapted arm needs it as much as the others: 2% of Wittmann et al.'s
        # sample is the disclosure that stops the row being read as theirs.
        assert str(PUBLISHED_TRAINING_SIZE) in sampler.budget_note
        assert sampler.required_budget == sampler._training_size + PUBLISHED_BATCH_SIZE
