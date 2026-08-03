"""Tests for the rule that picks the configuration the benchmark reports.

The rule is only worth having if it is mechanical. A selection procedure that
can be nudged is not a procedure, so what these pin is that the rule reaches its
answer from the numbers alone: that a genuine winner is not overturned by a
tie-break it never triggered, that a tie *is* broken by diversity rather than by
arrival order or alphabetical accident, and that the cases where selecting at
all would be dishonest raise instead of returning something confident-looking.

The failure to fear is not a crash. It is a `Selection` naming an arm that the
evidence does not support, since nothing downstream re-checks the choice.

The joint screen adds three more failures worth pinning, and none of them
crashes either: a candidate set nobody can redraw, a store key that two
configurations share, and a confirmation that quietly dropped the configuration
already in hand.
"""

import numpy as np
import pytest

from evogfn.benchmark.methods import DEFAULT_HIDDEN_DIM, DEFAULT_MIX, DEFAULT_TRAINING_STEPS
from evogfn.benchmark.selection import (
    SCREEN_BETAS,
    SCREEN_HIDDEN_DIMS,
    SCREEN_LAMS,
    SCREEN_MIXES,
    SCREEN_STEPS,
    SCREENED_OBJECTIVES,
    Configuration,
    Ranked,
    Screen,
    confirmation_set,
    incumbent,
    propose_allocation,
    rank_screen,
    sample_screen,
    screen_arms,
    screen_space,
    select,
)


class Record:
    """The two fields the rule reads, which is all a stored record needs here."""

    def __init__(self, regret, diversity=1.0):
        self.regret = regret
        self.diversity = diversity


def arm(regrets, diversity=1.0):
    return {i: Record(r, diversity) for i, r in enumerate(regrets)}


class TestAClearWinner:
    def test_a_separated_arm_wins_on_regret(self):
        # Wide, non-overlapping separation, so no tie-break should be consulted.
        chosen = select({"good": arm([0.1] * 20), "bad": arm([0.9] * 20)})
        assert chosen.chosen == "good"
        assert chosen.tied == ("good",)

    def test_its_reason_says_it_was_separated(self):
        chosen = select({"good": arm([0.1] * 20), "bad": arm([0.9] * 20)})
        assert "separated from every other arm" in chosen.reason

    def test_diversity_cannot_overturn_a_separated_arm(self):
        # The rule is regret-first. An arm that is genuinely worse on fitness
        # does not win by being more diverse, or the headline column would be
        # selected against the thing the benchmark is measuring.
        chosen = select(
            {
                "good": arm([0.1] * 20, diversity=1.0),
                "diverse-but-worse": arm([0.9] * 20, diversity=99.0),
            }
        )
        assert chosen.chosen == "good"


class TestATie:
    @pytest.fixture
    def tied(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.2, 30)
        return {
            "narrow": {i: Record(0.5 + n, 2.0) for i, n in enumerate(noise)},
            "wide": {i: Record(0.5 + n, 8.0) for i, n in enumerate(noise)},
        }

    def test_diversity_breaks_it(self, tied):
        assert select(tied).chosen == "wide"

    def test_both_arms_are_named_as_tied(self, tied):
        assert select(tied).tied == ("narrow", "wide")

    def test_the_reason_says_the_choice_rested_on_diversity(self, tied):
        # A caption drawn from this must not imply the arm won on fitness.
        assert "tied on regret" in select(tied).reason

    def test_an_arm_behind_by_less_than_the_noise_is_still_eligible(self):
        # Losing by less than the evidence can resolve is not losing. Were this
        # wrong, the rule would silently become "lowest mean", and the tie-break
        # would never fire on real data.
        rng = np.random.default_rng(1)
        # Independent noise per arm, so the *paired difference* carries variance.
        # Sharing one noise draw would make the difference a constant, which a
        # paired test resolves at any sample size -- a tie that cannot occur.
        records = {
            "leader": {i: Record(0.50 + n, 1.0) for i, n in enumerate(rng.normal(0, 0.3, 30))},
            "behind": {i: Record(0.51 + n, 9.0) for i, n in enumerate(rng.normal(0, 0.3, 30))},
        }
        assert select(records).chosen == "behind"


class TestRefusals:
    def test_nothing_to_choose_between(self):
        with pytest.raises(ValueError, match="no arms"):
            select({})

    def test_arms_sharing_no_seed(self):
        # Unpaired arms would still produce a mean, and a mean here would look
        # exactly like a result.
        with pytest.raises(ValueError, match="share no seed"):
            select({"a": {0: Record(0.1)}, "b": {1: Record(0.9)}})

    def test_every_arm_failing_everywhere(self):
        with pytest.raises(ValueError, match="failed on every"):
            select({"a": arm([np.inf] * 5), "b": arm([np.inf] * 5)})


class TestPartialFailure:
    def test_an_arm_is_scored_on_the_seeds_it_survived(self):
        records = {
            "flaky": {0: Record(0.1), 1: Record(np.inf), 2: Record(0.1)},
            "steady": {0: Record(0.5), 1: Record(0.5), 2: Record(0.5)},
        }
        assert select(records).regret["flaky"] == pytest.approx(0.1)

    def test_an_arm_that_failed_somewhere_does_not_win_on_a_tie_break(self):
        # Its mean is computed over survivors, which flatters it; letting that
        # number into a paired comparison would compare different seed sets.
        records = {
            "flaky": {i: Record(np.inf if i % 2 else 0.01, 99.0) for i in range(20)},
            "steady": {i: Record(0.5, 1.0) for i in range(20)},
        }
        assert select(records).tied == ("flaky",)


def test_only_shared_seeds_are_used():
    records = {
        "a": {0: Record(0.1), 1: Record(0.1), 2: Record(99.0)},
        "b": {0: Record(0.5), 1: Record(0.5)},
    }
    # Seed 2 belongs to one arm only, so it must not reach either mean.
    assert select(records).regret["a"] == pytest.approx(0.1)


class TestTheSpace:
    @pytest.mark.parametrize("objective", SCREENED_OBJECTIVES)
    def test_every_axis_is_crossed(self, objective):
        knob = SCREEN_LAMS if objective == "gfn-subtb" else SCREEN_MIXES
        expected = len(SCREEN_BETAS) * len(SCREEN_STEPS) * len(knob) * len(SCREEN_HIDDEN_DIMS)
        assert len(screen_space(objective)) == expected

    def test_subtb_carries_lam_and_never_mix(self):
        space = screen_space("gfn-subtb")
        assert {c.lam for c in space} == set(SCREEN_LAMS)
        assert {c.mix for c in space} == {None}

    def test_genetic_gfn_carries_mix_and_never_lam(self):
        # lam belongs to sub-trajectory balance. An arm built with it here would
        # be refused; one built ignoring it would be a column of identical rows
        # reading as "the knob does not matter".
        space = screen_space("genetic-gfn")
        assert {c.mix for c in space} == set(SCREEN_MIXES)
        assert {c.lam for c in space} == {None}

    def test_neither_endpoint_of_mix_is_offered(self):
        # Zero is plain trajectory balance carrying unused machinery, so it does
        # not measure this method at all.
        assert 0.0 not in SCREEN_MIXES
        assert 1.0 not in SCREEN_MIXES

    @pytest.mark.parametrize("objective", SCREENED_OBJECTIVES)
    def test_the_incumbent_is_a_point_of_its_own_space(self, objective):
        # Otherwise the screen would be searching somewhere the standing
        # configuration could not have been found, and the two would not be
        # comparable on the same axes.
        assert incumbent(objective, SCREEN_BETAS[2]) in screen_space(objective)


class TestTheDraw:
    def test_the_same_seed_draws_the_same_configurations(self):
        assert sample_screen("gfn-subtb", size=20, seed=7) == sample_screen(
            "gfn-subtb", size=20, seed=7
        )

    def test_another_seed_draws_something_else(self):
        # Were the draw seed-independent, "sampled at random" would describe a
        # fixed list and the screen would have no coverage argument at all.
        assert sample_screen("gfn-subtb", size=20, seed=7) != sample_screen(
            "gfn-subtb", size=20, seed=8
        )

    def test_it_draws_the_number_asked_for_without_repeating_one(self):
        drawn = sample_screen("genetic-gfn", size=100, seed=3)
        assert len(drawn) == 100
        assert len(set(drawn)) == 100

    def test_every_draw_is_in_the_space(self):
        space = set(screen_space("gfn-subtb"))
        assert set(sample_screen("gfn-subtb", size=100, seed=3)) <= space

    def test_asking_for_more_than_exists_refuses(self):
        # Silently returning the whole space would report a screen where an
        # exhaustive grid was run.
        with pytest.raises(ValueError, match="exhaustive grid"):
            sample_screen("gfn-subtb", size=10_000)


class TestArmNames:
    @pytest.mark.parametrize("objective", SCREENED_OBJECTIVES)
    def test_a_name_round_trips_to_its_configuration(self, objective):
        # A name is a store key. One that cannot be read back is a record whose
        # configuration is only knowable by re-deriving the draw.
        for configuration in screen_space(objective):
            assert Configuration.parse(configuration.name) == configuration

    @pytest.mark.parametrize("objective", SCREENED_OBJECTIVES)
    def test_no_two_configurations_share_a_name(self, objective):
        space = screen_space(objective)
        assert len({c.name for c in space}) == len(space)

    def test_the_two_spaces_do_not_collide_with_each_other(self):
        first, second = (set(screen_space(o)) for o in SCREENED_OBJECTIVES)
        assert not {c.name for c in first} & {c.name for c in second}

    def test_a_name_from_another_scheme_is_refused(self):
        with pytest.raises(ValueError, match="not a screened arm name"):
            Configuration.parse("gfn-subtb-beta-0.3")

    def test_a_name_missing_an_axis_is_refused(self):
        with pytest.raises(ValueError, match="omits"):
            Configuration.parse("gfn-subtb@b0.3-s300")

    def test_duplicate_configurations_cannot_share_a_store_key(self):
        one = Configuration("gfn-subtb", beta=0.3, steps=300, hidden_dim=128, lam=0.9)
        with pytest.raises(ValueError, match="share an arm name"):
            screen_arms([one, one])


class TestRecoveringWhatRan:
    @pytest.fixture
    def configuration(self):
        return Configuration("genetic-gfn", beta=0.3, steps=150, hidden_dim=64, mix=0.25)

    def test_the_record_is_what_the_configuration_is_read_from(self, configuration):
        recovered = Configuration.from_record(
            configuration.name,
            {"beta": 0.3, "steps": 150, "hidden_dim": 64, "mix": 0.25, "family": "x"},
        )
        assert recovered == configuration

    def test_a_record_disagreeing_with_its_name_refuses(self, configuration):
        # The failure this guards is silent everywhere else: an arm named for
        # one configuration whose campaign ran another.
        with pytest.raises(ValueError, match="never happened"):
            Configuration.from_record(
                configuration.name,
                {"beta": 0.3, "steps": 300, "hidden_dim": 64, "mix": 0.25},
            )

    def test_a_record_missing_a_setting_refuses(self, configuration):
        with pytest.raises(ValueError, match="omits"):
            Configuration.from_record(configuration.name, {"beta": 0.3, "steps": 150})


class TestTheIncumbent:
    @pytest.fixture
    def standing(self):
        return incumbent("gfn-subtb", 0.3)

    def test_it_is_the_shipped_configuration(self, standing):
        assert standing.steps == DEFAULT_TRAINING_STEPS
        assert standing.hidden_dim == DEFAULT_HIDDEN_DIM

    def test_a_breeding_objective_gets_the_shipped_mixing_ratio(self):
        assert incumbent("genetic-gfn", 0.3).mix == DEFAULT_MIX

    def test_it_is_confirmed_even_when_no_screen_nominated_it(self, standing):
        # The whole safety argument. Without this the screen could displace the
        # standing configuration on ten noisy seeds.
        other = Configuration("gfn-subtb", beta=0.1, steps=50, hidden_dim=64, lam=0.1)
        assert standing in confirmation_set([other.name], standing)

    def test_it_is_not_confirmed_twice_when_a_screen_did_nominate_it(self, standing):
        chosen = confirmation_set([standing.name], standing)
        assert chosen == (standing,)

    def test_it_comes_first(self, standing):
        other = Configuration("gfn-subtb", beta=0.1, steps=50, hidden_dim=64, lam=0.1)
        assert confirmation_set([other.name], standing)[0] == standing


class TestNomination:
    def test_configurations_are_ordered_by_regret(self):
        screen = rank_screen(
            "gfn-subtb", {"a": arm([0.5] * 10), "b": arm([0.1] * 10), "c": arm([0.9] * 10)}
        )
        assert [entry.name for entry in screen.ranked] == ["b", "a", "c"]

    def test_an_exact_tie_on_regret_goes_to_diversity(self):
        # The pre-declared rule's own second axis, so a shortlist is not drawn
        # on a criterion invented for the screen -- and never on arrival order.
        screen = rank_screen(
            "gfn-subtb",
            {"narrow": arm([0.5] * 10, diversity=2.0), "wide": arm([0.5] * 10, diversity=8.0)},
        )
        assert screen.leader.name == "wide"

    def test_it_says_what_it_could_not_resolve(self):
        rng = np.random.default_rng(0)
        records = {
            name: {i: Record(0.5 + n) for i, n in enumerate(rng.normal(0, 0.3, 10))}
            for name in ("a", "b", "c")
        }
        # Every arm is the same distribution, so a ten-seed screen must report
        # the whole set as plausible rather than presenting an order.
        assert len(rank_screen("gfn-subtb", records).indistinguishable) == 3

    def test_a_separated_configuration_is_not_in_the_plausible_set(self):
        screen = rank_screen("gfn-subtb", {"good": arm([0.1] * 10), "bad": arm([0.9] * 10)})
        assert screen.indistinguishable == ("good",)

    def test_unpaired_arms_refuse(self):
        with pytest.raises(ValueError, match="share no seed"):
            rank_screen("gfn-subtb", {"a": {0: Record(0.1)}, "b": {1: Record(0.9)}})


class TestTheAllocationProposal:
    def _screen(self, objective, regrets):
        # Built from per-seed values rather than a mean, because the split turns
        # on a paired comparison between the two leaders: a helper that took a
        # mean could only ever exercise a rule that compared means.
        values = tuple(float(r) for r in regrets)
        return Screen(
            objective=objective,
            ranked=(
                Ranked(
                    f"{objective}@b0.3-s300-h128",
                    float(np.mean(values)),
                    float(np.std(values, ddof=1)),
                    1.0,
                    True,
                    values,
                ),
            ),
            seeds=tuple(range(len(values))),
        )

    def _noisy(self, objective, centre, seed):
        rng = np.random.default_rng(seed)
        return self._screen(objective, centre + rng.normal(0, 0.3, 10))

    def test_an_even_split_when_the_leaders_are_level(self):
        screens = [self._screen("gfn-subtb", [0.4] * 10), self._screen("genetic-gfn", [0.4] * 10)]
        assert propose_allocation(screens, slots=10, floor=2) == {
            "gfn-subtb": 5,
            "genetic-gfn": 5,
        }

    def test_a_lead_smaller_than_the_noise_does_not_skew_the_split(self):
        # The failure this guards is silent and would fire on every real screen:
        # two ten-seed means are never exactly equal, so a rule that compared
        # means would hand eight of ten slots to whichever screen sampled
        # luckier and would never once report the two as level.
        screens = [self._noisy("gfn-subtb", 0.40, 0), self._noisy("genetic-gfn", 0.42, 1)]
        assert screens[0].leader.regret != screens[1].leader.regret
        assert propose_allocation(screens, slots=10, floor=2) == {
            "gfn-subtb": 5,
            "genetic-gfn": 5,
        }

    def test_the_weaker_screen_keeps_the_floor_rather_than_nothing(self):
        # A ten-seed screen is not evidence enough to write an objective off,
        # which matters most where one has already been placed low once.
        screens = [self._screen("gfn-subtb", [0.2] * 10), self._screen("genetic-gfn", [0.6] * 10)]
        assert propose_allocation(screens, slots=10, floor=2) == {
            "gfn-subtb": 8,
            "genetic-gfn": 2,
        }

    def test_the_skew_follows_the_better_screen_either_way_round(self):
        screens = [self._screen("gfn-subtb", [0.6] * 10), self._screen("genetic-gfn", [0.2] * 10)]
        assert propose_allocation(screens, slots=10, floor=2)["genetic-gfn"] == 8

    def test_a_floor_that_cannot_be_met_refuses(self):
        screens = [self._screen("gfn-subtb", [0.2] * 10), self._screen("genetic-gfn", [0.6] * 10)]
        with pytest.raises(ValueError, match="floor"):
            propose_allocation(screens, slots=3, floor=2)

    def test_screens_on_different_seeds_refuse(self):
        first = self._screen("gfn-subtb", [0.2] * 10)
        second = Screen("genetic-gfn", first.ranked, tuple(range(5)))
        with pytest.raises(ValueError, match="share a seed set"):
            propose_allocation([first, second], slots=10, floor=2)
