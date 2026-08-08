"""The saturation experiment's grids, and the twin that stands in for an arm.

Two things here can be silently wrong and neither raises when it is.

**A grid the shipped configuration is not on.** The whole point of the ladder is
to say whether the budget the paper prints is the right one, and a grid missing
that budget prints a curve the reported arm is not on. So the shipped budget is
never a literal here -- a literal drifts with the grid, silently, which is the
one failure these checks exist to catch.

That splits in two, because the two halves are knowable in different places.
*Which budget ships* is answered by `shipped_base` in any checkout: with a
selection recorded it is the selected step count, and without one it is the
untuned default the headline table genuinely falls back to. So "the grid
contains and brackets the budget that ships" is checked everywhere, CI included.
*Which arm ships* is answered only by ``results/selected.json``, which is
gitignored and therefore absent in a fresh checkout, so "the ladder's rung at
that budget is the very store cell the confirmation paid for" is checked only
where the selection has run.

**A twin that has drifted from the arm it claims to be.** ``genetic+search``'s
inner-loop budget is not exposed by `classical`, so the ladder runs a
script-local rebuild of it. A rebuild that has drifted still runs, still
produces plausible regrets, and reports them under a name claiming to be the
published ablation. The script's own precondition closes this against the store
at the shipped budget; this closes it at CI speed, end to end on a toy task, so
the drift is caught by the edit that introduces it rather than by an hour of
campaigns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from evogfn.algorithms.inner_loop import DEFAULT_GENERATIONS, ProxyOptimising
from evogfn.benchmark.methods import (
    BASELINES,
    DEFAULT_POOL,
    SELECTED_CONFIGURATION,
    shipped_base,
)
from evogfn.benchmark.saturation import saturation
from evogfn.benchmark.selection import Configuration
from evogfn.benchmark.store import ResultStore
from tests.benchmark.test_methods import toy_task

#: Gates the one check that needs the *name* of the arm that ships. The recorded
#: selection lives under ``results/``, which is gitignored, so it is present on a
#: machine that has run the selection phase and absent everywhere else, CI
#: included. Kept as narrow as it can be: everything the shipped *budget* alone
#: can answer is checked without it, through `shipped_steps` below.
needs_selection = pytest.mark.skipif(
    not SELECTED_CONFIGURATION.exists(),
    reason="results/selected.json is gitignored; present only after the selection phase has run",
)


def _load():
    """Import the experiment script, which is outside the installed package."""
    path = Path(__file__).resolve().parents[2] / "experiments" / "proxy_saturation.py"
    spec = importlib.util.spec_from_file_location("proxy_saturation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = _load()


def shipped_steps():
    """Gradient steps per round for the GFlowNet arm this checkout would run.

    Resolved through `shipped_base` -- the same call ``methods_for`` resolves the
    headline table's GFlowNet arm through -- rather than read out of
    ``results/selected.json`` here or restated as a number. Two things follow,
    and both are the point.

    It is never a literal, so a grid edited away from the budget that ships fails
    rather than agreeing with itself. And it answers in a fresh checkout too,
    where it returns the untuned default, which is not a stand-in for the
    selection but genuinely what the headline table falls back to running there.
    The property the grid owes -- that the ladder covers the budget being
    reported -- is owed in both states, so it is checked in both.
    """
    return shipped_base().steps


def proxy_spend(campaign):
    """What a campaign's sampler reports having spent on the proxy."""
    return campaign.sampler.proxy_calls


def population_of(campaign):
    """The inner loop's population, which is held while `generations` moves."""
    return campaign.sampler._population


def settings(arm):
    """What an arm recorded about its own configuration."""
    return arm.parameters


def flat(budgets):
    """A perfectly flat ladder over `budgets`, so only the spacing is under test."""
    return {budget: dict.fromkeys(range(8), 0.5) for budget in budgets}


class TestTheGrids:
    def test_both_grids_are_ladders_the_rule_will_accept(self):
        # The rule refuses anything that is not a contiguous doubling ladder, so
        # a grid edited to a spacing the rule cannot read fails here rather than
        # after the campaigns have been paid for.
        saturation(flat(experiment.STEPS), axis="steps")
        saturation(flat(experiment.GENERATIONS), axis="generations")

    def test_the_shipped_step_count_is_on_the_grid(self):
        assert shipped_steps() in experiment.STEPS

    def test_the_shipped_generation_count_is_on_the_grid(self):
        assert DEFAULT_GENERATIONS in experiment.GENERATIONS
        assert experiment.SHIPPED_GENERATIONS == DEFAULT_GENERATIONS

    def test_each_grid_brackets_its_shipped_value_from_both_sides(self):
        # A grid whose best value sits at its own edge cannot distinguish "this
        # budget is right" from "this budget is the largest one offered", and
        # the question here is how *little* is enough as much as how much.
        assert min(experiment.STEPS) < shipped_steps() < max(experiment.STEPS)
        assert min(experiment.GENERATIONS) < DEFAULT_GENERATIONS < max(experiment.GENERATIONS)


class TestTheStepsLadder:
    @needs_selection
    def test_the_shipped_rung_is_the_arm_the_confirmation_already_paid_for(self):
        # The half of the grid question that only a recorded selection can
        # answer. `TestTheGrids` establishes that the shipped *budget* is on the
        # ladder; this establishes that the rung standing at it is the same store
        # *cell* the confirmation already bought its seeds under, which turns on
        # `LADDER`'s pinned beta, lambda and width agreeing with the selected
        # ones -- an agreement no amount of grid arithmetic implies.
        #
        # Skipped rather than propped up with a literal arm name: with no
        # selection recorded the shipped base is the untuned `gfn-tb`, which is
        # not a screened arm name and has no entry in `LADDER`, so there is no
        # cell to be identical to. A pinned name would make this pass in CI while
        # testing that the literal equals itself.
        shipped = shipped_base()
        chosen = Configuration.parse(shipped.name)
        names = {rung.steps: rung.name for rung in experiment.steps_rungs(chosen.objective)}
        assert names[chosen.steps] == shipped.name

    def test_only_the_step_count_moves_along_a_ladder(self):
        for objective in experiment.LADDER:
            rungs = experiment.steps_rungs(objective)
            varied = {
                (rung.objective, rung.beta, rung.hidden_dim, rung.lam, rung.mix) for rung in rungs
            }
            assert len(varied) == 1
            assert tuple(rung.steps for rung in rungs) == experiment.STEPS

    def test_every_rung_names_a_distinct_store_key_that_round_trips(self):
        for objective in experiment.LADDER:
            arms = experiment.steps_arms(objective)
            assert len(arms) == len(experiment.STEPS)
            for name in arms:
                assert Configuration.parse(name).name == name

    def test_the_ladder_needs_no_source_change_to_build(self):
        # Everything in it comes from the existing `Configuration`/`screen_arms`
        # API, so nothing under `src/` moves and no stored record goes stale.
        for objective in experiment.LADDER:
            for arm in experiment.steps_arms(objective).values():
                assert callable(arm)


class TestTheTwinIsTheShippedArm:
    def test_it_runs_the_same_campaign_end_to_end(self):
        task = toy_task()
        theirs, mine = (
            BASELINES["genetic+search"](task, 0),
            experiment.genetic_search(DEFAULT_GENERATIONS)(task, 0),
        )
        shipped, twin = theirs.run(), mine.run()
        assert twin.best_value == shipped.best_value
        assert proxy_spend(mine) == proxy_spend(theirs)
        assert twin.oracle_calls == shipped.oracle_calls
        # Every design, in evaluation order: the campaigns are deterministic
        # given their seed, so anything short of identity means the two
        # pipelines diverged somewhere and a tolerance would only decide how
        # much divergence to accept.
        assert twin.sequences.tolist() == shipped.sequences.tolist()
        assert twin.values.tolist() == shipped.values.tolist()

    def test_a_different_budget_changes_the_proxy_spend_proportionally(self):
        task = toy_task()
        at_50 = experiment.genetic_search(50)(task, 0)
        at_100 = experiment.genetic_search(100)(task, 0)
        at_50.run()
        at_100.run()
        assert proxy_spend(at_100) == 2 * proxy_spend(at_50)

    def test_it_is_wired_the_way_the_shipped_arm_is(self):
        campaign = experiment.genetic_search(7)(toy_task(), 0)
        sampler = campaign.sampler
        assert isinstance(sampler, ProxyOptimising)
        assert sampler._generations == 7
        # `population` is held at the wrapper's own default, exactly as the
        # shipped arm holds it: varying both halves of `generations x
        # population` would make the curve a curve in total spend rather than in
        # the knob we chose.
        reference = BASELINES["genetic+search"](toy_task(), 0)
        assert population_of(campaign) == population_of(reference)

    def test_its_record_says_what_it_ran_at(self):
        # The one field the shipped arm's records cannot carry, and the reason a
        # rung is legible from its record rather than only from its name.
        parameters = settings(experiment.genetic_search(200))
        assert parameters["generations"] == 200
        shipped = settings(BASELINES["genetic+search"])
        assert {k: v for k, v in parameters.items() if k != "generations"} == shipped
        assert parameters["pool_size"] == DEFAULT_POOL

    def test_a_rung_can_never_be_read_as_the_shipped_arm(self):
        names = {experiment.twin_name(g) for g in experiment.GENERATIONS}
        assert len(names) == len(experiment.GENERATIONS)
        assert "genetic+search" not in names

    @pytest.mark.parametrize("generations", [0, -1])
    def test_a_budget_that_buys_nothing_is_refused(self, generations):
        with pytest.raises(ValueError, match="at least 1"):
            experiment.genetic_search(generations)


def stocked(root, arms, levels, *, seeds=40, exhausted=()):
    """A store holding one synthetic rung per budget, at a fixed regret level."""
    store = ResultStore(root)
    for budget, name in arms.items():
        for seed in range(seeds):
            store.append(
                store.stamp(
                    task="objectives",
                    method=name,
                    seed=seed,
                    best=1.0 - levels[budget],
                    regret=levels[budget],
                    diversity=10.0,
                    proxy_calls=budget * 64,
                    exhausted=seed in exhausted and budget in exhausted,
                    protocol="4x96=384",
                    feasible_fraction=1.0,
                    oracle_calls=384,
                    proposals=384,
                )
            )
    return store


class TestTheReport:
    def test_it_prints_every_rung_every_doubling_and_the_verdict(self, tmp_path):
        arms = {b: f"arm-s{b}" for b in (75, 150, 300, 600)}
        levels = {75: 0.6, 150: 0.5, 300: 0.5, 600: 0.5}
        block = experiment.describe(
            "steps",
            "toy",
            stocked(tmp_path, arms, levels),
            "objectives",
            arms,
            ceiling=600,
            planned=40,
        )
        for budget in arms:
            assert f"arm-s{budget}" in block
        assert block.count("->") == len(arms) - 1
        assert "RULE: steps=150" in block
        # The seed line reads the spread and never the effect, and it is what
        # the pre-declared top-up trigger is decided from.
        assert "pooled paired spread" in block

    def test_a_ladder_that_never_flattened_says_so_rather_than_naming_a_knee(self, tmp_path):
        arms = {b: f"arm-s{b}" for b in (75, 150, 300, 600)}
        levels = {75: 0.8, 150: 0.6, 300: 0.4, 600: 0.2}
        block = experiment.describe(
            "steps", "toy", stocked(tmp_path, arms, levels), "objectives", arms, ceiling=600
        )
        assert "no saturation point measured" in block
        assert "cost ceiling we chose" in block

    def test_a_rung_that_was_never_run_is_named_rather_than_skipped(self, tmp_path):
        arms = {b: f"arm-s{b}" for b in (75, 150, 300)}
        stored = {75: 0.6, 150: 0.5}
        store = stocked(tmp_path, {b: arms[b] for b in stored}, stored)
        block = experiment.describe("steps", "toy", store, "objectives", arms, ceiling=300)
        assert "arm-s300" in block
        assert "not run" in block

    def test_a_dropped_seed_is_counted_rather_than_swallowed(self, tmp_path):
        arms = {b: f"arm-s{b}" for b in (75, 150)}
        levels = {75: 0.6, 150: 0.5}
        store = stocked(tmp_path, arms, levels, exhausted=(0, 1, 75, 150))
        regret, dropped = experiment.measured(store, "objectives", arms)
        assert dropped == {75: 2, 150: 2}
        assert set(regret[75]) == set(range(2, 40))
