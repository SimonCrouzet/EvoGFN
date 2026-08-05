"""Tests for the transfer probe.

Every failure pinned here is silent. None of them raises on its own, none shows
up as a crash, and each produces a number that looks exactly like a result.

**A "frozen" proposal that trains.** The surrogate carried to anchor B is fitted,
so `GFlowNetSampler.propose` would run 300 gradient steps before sampling and the
probe would be measuring a policy that had already adapted to B. The state-dict
digest is what catches it, and it is a digest rather than an inspection because
the parameter a hand-written check forgets is the one that moved.

**A stale anchor.** `AnchorConditionedPolicy._anchor` is a *persistent* buffer, so
it travels in the state dict: any "carry the weights" written as `load_state_dict`
restores anchor A over anchor B. The policy object is also shared, so the last
sampler constructed owns it. Either way the conditioning is simply wrong, the loss
stays finite, and nothing downstream complains.

**A screen that is not the same screen.** The whole probe rests on every arm at B
being screened identically; an arm reading its pool through a different model is
measuring surrogate quality alongside policy transfer. Object identity is asserted
directly, and the stored digest is asserted to be the same on every record so the
property is checkable from the store and not only from this file.

**A plate that was never screened.** `Campaign._design` returns the first 96
proposals unranked in round zero, which is why `measure_round` exists at all. A
round that silently took a prefix would discard the surrogate and nothing in the
output would say so.

**An anchor pair that is not what it claims.** A near anchor outside one round's
budget, or a far anchor inside it, collapses the distance axis criterion 2 is
measured on -- and the walk that builds them is the only thing standing between
the probe and a pair drawn at whatever distance the constructor happened to give.

**A probe that invalidates the store.** The module exists as a new file precisely
because roughly sixteen thousand records name the closure of
`suite.RESULT_DEPENDENCIES`. The moment anything in that closure imports it, an
edit to the probe makes every one of those records stale.

The end-to-end run is a twelve-position toy at one seed, which is enough to
exercise every seam -- both anchors, all ten arms, the retrain, the store -- and
cheap enough to run on every commit.
"""

import numpy as np
import pytest
import torch

from evogfn.algorithms.gflownet.sampler import GFlowNetSampler
from evogfn.algorithms.gflownet.training import TrainingConfig
from evogfn.benchmark import methods
from evogfn.benchmark import transfer as T  # noqa: N812 - the module under test, used on every line
from evogfn.benchmark.attainable import attainable_optimum
from evogfn.benchmark.protocol import Protocol
from evogfn.benchmark.selection import Configuration
from evogfn.benchmark.store import ResultStore, dependency_closure
from evogfn.benchmark.suite import DIAGNOSTIC_MUTATIONS, RESULT_DEPENDENCIES
from evogfn.benchmark.tasks import Task
from evogfn.env.mutation import MutationEnvironment
from evogfn.landscapes.ehrlich import EhrlichLandscape
from evogfn.models.policy import AnchorConditionedPolicy
from evogfn.rewards.base import TemperedReward
from evogfn.surrogate.base import Surrogate
from evogfn.surrogate.ensemble import DeepEnsemble
from evogfn.surrogate.proxy import ProxyLandscape

# A twelve-position, four-letter instance. Small enough that a whole probe run is
# seconds, constrained enough that the masks actually refuse moves -- which is
# what the anchor walk has to survive.
TOY = dict(  # noqa: C408 - keyword form mirrors EhrlichLandscape's own signature
    sequence_length=12,
    vocab_size=4,
    n_motifs=2,
    motif_length=2,
    transition_density=0.5,
    seed=3,
)

# Distances the toy landscape can hold. The real probe's 4 and 16 do not fit in
# twelve positions, and the *meaning* of the two levels -- inside one round's
# budget, and beyond it -- is what the tests check, not the magnitudes.
TOY_BUDGET = T.ProbeBudget(
    rounds=2,
    batch_size=4,
    pool_size=16,
    steps=2,
    near_distance=2,
    far_distance=6,
    beam_width=8,
)


@pytest.fixture(scope="module")
def toy():
    """The toy landscape."""
    return EhrlichLandscape(**TOY)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def toy_env(toy):
    """An environment at the toy landscape's own feasible sequence."""
    return T.training_environment(toy, toy.feasible_sequence(0))


@pytest.fixture(scope="module")
def probe_store(tmp_path_factory, toy):
    """One whole probe run at one seed, for the tests that read the store."""
    store = ResultStore(tmp_path_factory.mktemp("transfer"))
    ran = T.run_transfer_probe(store, [0], landscape=toy, budget=TOY_BUDGET, report=lambda _: None)
    return store, ran


def _fitted(landscape, env, seed=0):
    """A surrogate fitted on a handful of designs, for the screening tests."""
    designs = np.stack([env.parent for _ in range(6)])
    designs[np.arange(6), np.arange(6)] = (designs[np.arange(6), np.arange(6)] + 1) % 4
    ensemble = DeepEnsemble(
        n_tokens=landscape.alphabet.size,
        sequence_length=landscape.sequence_length,
        epochs=5,
        seed=seed,
    )
    ensemble.fit(designs, landscape.evaluate(designs))
    return ensemble


class TestTheFrozenPath:
    """A proposal at anchor B must change nothing about the policy."""

    def test_a_frozen_proposal_moves_no_weight(self, toy, toy_env):  # noqa: ARG002 - fixture ordering
        # The failure this catches is the whole reason `propose_frozen` exists:
        # `GFlowNetSampler.propose` trains first whenever its proxy is ready, and
        # the proxy carried from A always is. An arm that trained at B would look
        # like a policy that transferred.
        policy = T._policy(toy_env, hidden_dim=8, seed=1, anchor_conditioned=False)
        before = T.state_digest(policy)
        T.propose_frozen(toy_env, policy, 4, generator=torch.Generator().manual_seed(0))
        assert T.state_digest(policy) == before

    def test_a_frozen_proposal_leaves_no_gradient(self, toy_env):
        # A rollout that built a graph would not raise; it would leak memory and
        # make the next `.backward()` anywhere touch these parameters.
        policy = T._policy(toy_env, hidden_dim=8, seed=1, anchor_conditioned=False)
        T.propose_frozen(toy_env, policy, 4, generator=torch.Generator().manual_seed(0))
        assert all(p.grad is None for p in policy.parameters())

    def test_frozen_is_not_inert(self, toy_env):
        # "Frozen" must mean the weights do not move, not that the sampler
        # repeats itself. A generator that failed to advance would hand every
        # round the same pool and the plate would stop being a sample at all.
        policy = T._policy(toy_env, hidden_dim=8, seed=1, anchor_conditioned=False)
        generator = torch.Generator().manual_seed(0)
        first = T.propose_frozen(toy_env, policy, 8, generator=generator)
        digest = T.state_digest(policy)
        second = T.propose_frozen(toy_env, policy, 8, generator=generator)
        assert not np.array_equal(first, second)
        assert T.state_digest(policy) == digest

    def test_log_z_is_covered_by_the_digest(self, toy_env):
        # `log_z` carries a learning rate an order of magnitude above the
        # policy's, so it is the parameter an accidental optimiser step moves
        # first. Pinned separately because a digest that silently skipped it
        # would still pass every other test here.
        policy = T._policy(toy_env, hidden_dim=8, seed=1, anchor_conditioned=False)
        before = T.state_digest(policy)
        with torch.no_grad():
            policy.log_z += 1.0  # type: ignore[misc]
        assert T.state_digest(policy) != before

    def test_the_digest_covers_the_anchor_buffer(self, toy_env):
        # The anchor is a *buffer*, not a parameter. A digest over
        # `parameters()` alone would call a policy frozen while its conditioning
        # moved underneath it -- which is exactly the silent failure this module
        # is built around.
        policy = T._policy(toy_env, hidden_dim=8, seed=1, anchor_conditioned=True)
        before = T.state_digest(policy)
        moved = np.asarray(toy_env.parent).copy()
        moved[0] = (moved[0] + 1) % 4
        policy.set_anchor(moved)  # type: ignore[operator]
        assert T.state_digest(policy) != before


class TestTheStaleAnchorGuard:
    """The one failure the specification says produces publishable-looking numbers."""

    def test_a_state_dict_carry_restores_the_old_anchor_and_is_refused(self, toy, toy_env):
        # `_anchor` is registered with the default `persistent=True`, so
        # `load_state_dict` writes anchor A back over anchor B without a word.
        # This is the most natural way to write "weights carried", so the guard
        # has to catch it rather than the convention having to avoid it.
        moved_env = toy_env.reanchored(_walked(toy, toy_env, 2))
        at_a = T._policy(toy_env, hidden_dim=8, seed=1, anchor_conditioned=True)
        at_b = T._policy(moved_env, hidden_dim=8, seed=2, anchor_conditioned=True)
        at_b.load_state_dict(at_a.state_dict())
        with pytest.raises(T.StaleAnchorError, match="conditioned on a parent"):
            T.propose_frozen(moved_env, at_b, 2, generator=torch.Generator().manual_seed(0))

    def test_a_policy_the_last_sampler_took_is_refused(self, toy, toy_env):
        # `GFlowNetSampler.__init__` rebinds the anchor, and the policy is shared
        # rather than copied -- so a policy belongs to whichever sampler was
        # constructed last. The probe holds an A-context and a B-context in one
        # process by design, so this is live rather than theoretical.
        moved_env = toy_env.reanchored(_walked(toy, toy_env, 2))
        policy = T._policy(toy_env, hidden_dim=8, seed=1, anchor_conditioned=True)
        parts = _sampler_parts(toy, toy_env)
        GFlowNetSampler(toy_env, policy, **parts)
        GFlowNetSampler(moved_env, policy, **parts)
        with pytest.raises(T.StaleAnchorError):
            T.propose_frozen(toy_env, policy, 2, generator=torch.Generator().manual_seed(0))

    def test_an_unconditioned_policy_is_left_alone(self, toy, toy_env):
        # The guard must not turn into a constraint on the arms that hold no
        # anchor: `plain-transferred` is the arm whose whole point is that it
        # cannot see one.
        moved_env = toy_env.reanchored(_walked(toy, toy_env, 2))
        policy = T._policy(toy_env, hidden_dim=8, seed=1, anchor_conditioned=False)
        T.assert_anchor_bound(policy, moved_env)

    def test_binding_happens_outside_the_measured_window(self, toy, toy_env):
        # If the arm re-bound inside its own proposal, the anchor buffer would
        # move mid-proposal and the frozen check would have to be weakened to
        # tolerate it -- losing the assertion that catches an optimiser step.
        moved_env = toy_env.reanchored(_walked(toy, toy_env, 2))
        policy = T._policy(toy_env, hidden_dim=8, seed=1, anchor_conditioned=True)
        propose = T._frozen_arm(moved_env, policy, seed=0)
        assert np.array_equal(
            policy.anchor.numpy(),  # type: ignore[operator]
            np.asarray(moved_env.parent),
        )
        before = T.state_digest(policy)
        propose(4)
        assert T.state_digest(policy) == before


class TestAnchorPairs:
    """B has to be exactly as far from A as the level says, and buildable."""

    @pytest.mark.parametrize("level", ["near", "far"])
    def test_the_anchor_moves_the_nominal_distance(self, toy, level):
        pair = T.anchor_pair(
            toy,
            seed=0,
            level=level,
            near_distance=TOY_BUDGET.near_distance,
            far_distance=TOY_BUDGET.far_distance,
        )
        expected = TOY_BUDGET.near_distance if level == "near" else TOY_BUDGET.far_distance
        assert pair.achieved_distance == expected
        assert int(np.count_nonzero(pair.training != pair.moved)) == expected

    @pytest.mark.parametrize("level", ["near", "far"])
    def test_both_anchors_are_feasible(self, toy, level):
        # An infeasible anchor is refused by `MutationEnvironment.reanchored`
        # mid-run, after the training campaign has already been paid for.
        pair = T.anchor_pair(
            toy,
            seed=0,
            level=level,
            near_distance=TOY_BUDGET.near_distance,
            far_distance=TOY_BUDGET.far_distance,
        )
        assert bool(toy.is_feasible(pair.training[None, :])[0])
        assert bool(toy.is_feasible(pair.moved[None, :])[0])

    def test_the_far_anchor_is_beyond_one_round(self, toy):
        # This is the operative content of "no trajectory from the training
        # anchor arrives in this neighbourhood". Without it the two levels do not
        # bracket anything and criterion 2 has no axis to be measured on.
        pair = T.anchor_pair(
            toy,
            seed=0,
            level="far",
            near_distance=TOY_BUDGET.near_distance,
            far_distance=TOY_BUDGET.far_distance,
        )
        env = MutationEnvironment(
            pair.training,
            toy.alphabet,
            max_mutations=DIAGNOSTIC_MUTATIONS,
            transitions=np.asarray(toy.transition_matrix),
        )
        assert not bool(env.is_reachable(pair.moved[None, :])[0])

    def test_the_near_anchor_is_inside_one_round(self, toy):
        pair = T.anchor_pair(
            toy,
            seed=0,
            level="near",
            near_distance=TOY_BUDGET.near_distance,
            far_distance=TOY_BUDGET.far_distance,
        )
        env = MutationEnvironment(
            pair.training,
            toy.alphabet,
            max_mutations=DIAGNOSTIC_MUTATIONS,
            transitions=np.asarray(toy.transition_matrix),
        )
        assert bool(env.is_constructible(pair.moved[None, :])[0])

    def test_the_two_levels_do_not_share_a_walk(self, toy):
        # A shared random stream would put the near anchor on the path to the far
        # one, so the two levels would differ by less than their distances claim
        # and the interaction criterion 2 measures would be attenuated.
        kwargs = {
            "near_distance": TOY_BUDGET.near_distance,
            "far_distance": TOY_BUDGET.far_distance,
        }
        near = T.anchor_pair(toy, seed=0, level="near", **kwargs)
        far = T.anchor_pair(toy, seed=0, level="far", **kwargs)
        assert np.array_equal(near.training, far.training)
        near_sites = set(np.nonzero(near.training != near.moved)[0].tolist())
        far_sites = set(np.nonzero(far.training != far.moved)[0].tolist())
        assert not near_sites <= far_sites

    def test_a_distance_the_landscape_cannot_hold_is_refused(self, toy):
        with pytest.raises(ValueError, match="must lie in"):
            T.anchor_pair(toy, seed=0, level="far", far_distance=99)


class TestPerAnchorAttainability:
    """The audited quantity, re-parameterised and not reimplemented."""

    def test_it_reproduces_the_audit_at_the_wild_type(self):
        # The whole claim of `anchor_attainable` is that it is
        # `attainable_optimum` with the parent taken as an argument instead of
        # read off a task. A drift between the two would put the probe's regret
        # column on a different scale from every other regret in the store, and
        # nothing in either number would say so.
        landscape = T.diagnostic_landscape()
        task = Task(
            name="probe-check",
            purpose="pin the re-parameterisation",
            build=T.diagnostic_landscape,
            protocol=Protocol(rounds=1, batch_size=4, max_mutations=DIAGNOSTIC_MUTATIONS),
            max_mutations=DIAGNOSTIC_MUTATIONS,
            parent_seed=0,
        )
        theirs = attainable_optimum(task, budget=DIAGNOSTIC_MUTATIONS, beam_width=8)
        mine = T.anchor_attainable(
            landscape,
            task.parent(landscape),
            budget=DIAGNOSTIC_MUTATIONS,
            label="probe-check",
            beam_width=8,
        )
        assert mine.lower == theirs.lower
        assert mine.upper == theirs.upper
        assert mine.exact == theirs.exact

    def test_it_audits_the_moved_anchor_rather_than_the_training_one(self):
        # Attainability depends on the anchor, which is the entire reason the
        # audit is per anchor. An implementation that ignored the argument -- and
        # scored the wild type, or the task's parent -- would put every arm's
        # regret against the wrong target while every number stayed plausible.
        # Run on the real diagnostic instance: the toy landscape is solved from
        # everywhere at this budget, so it cannot tell the two apart.
        landscape = T.diagnostic_landscape()
        pairs = [T.anchor_pair(landscape, seed=seed, level="far") for seed in range(3)]

        def audited(anchors):
            return [
                T.anchor_attainable(
                    landscape,
                    anchor,
                    budget=DIAGNOSTIC_MUTATIONS,
                    label="probe-check",
                    beam_width=8,
                ).upper
                for anchor in anchors
            ]

        assert audited([p.moved for p in pairs]) != audited([p.training for p in pairs])


class TestTheSharedScreen:
    """One surrogate, no refits, and a plate that was actually ranked."""

    def test_a_refit_is_refused(self, toy, toy_env):
        # Refitting at B would confound transfer of the policy with transfer of
        # the surrogate, and the training routines this module calls would
        # happily do it if they could.
        frozen = T.FrozenSurrogate(_fitted(toy, toy_env))
        with pytest.raises(RuntimeError, match="must not be refitted"):
            frozen.fit(np.stack([toy_env.parent]), toy.evaluate(toy_env.parent[None, :]))
        assert frozen.refits_attempted == 1

    def test_an_unfitted_surrogate_is_refused_at_construction(self, toy):
        # Wrapping an unfitted model would leave every arm's plate unranked --
        # the exact failure the shared screen exists to prevent -- and nothing in
        # a record would say so, because the plate would still be full.
        ensemble = DeepEnsemble(
            n_tokens=toy.alphabet.size, sequence_length=toy.sequence_length, epochs=1
        )
        with pytest.raises(ValueError, match="has not been fitted"):
            T.FrozenSurrogate(ensemble)

    def test_the_plate_is_screened_and_not_a_prefix(self, toy, toy_env):
        # `Campaign._design` returns `pool[:size]` unscored in round zero, which
        # is why this module reimplements the round. A prefix would silently
        # discard the screen and the plate would still look full and plausible.
        pool = T.propose_frozen(
            toy_env,
            T._policy(toy_env, hidden_dim=8, seed=0, anchor_conditioned=False),
            12,
            generator=torch.Generator().manual_seed(0),
        )
        measured = T.measure_round(
            toy, pool, T.FrozenSurrogate(_Reversing(len(pool))), batch_size=3
        )
        assert np.array_equal(measured.batch, pool[::-1][:3])

    def test_the_round_never_refits(self, toy, toy_env):
        pool = T.propose_frozen(
            toy_env,
            T._policy(toy_env, hidden_dim=8, seed=0, anchor_conditioned=False),
            12,
            generator=torch.Generator().manual_seed(0),
        )
        frozen = T.FrozenSurrogate(_fitted(toy, toy_env))
        measured = T.measure_round(toy, pool, frozen, batch_size=3)
        assert measured.surrogate_refits == 0
        assert frozen.predictions == 1

    def test_a_pool_smaller_than_the_plate_is_refused(self, toy, toy_env):
        frozen = T.FrozenSurrogate(_fitted(toy, toy_env))
        pool = np.stack([toy_env.parent, toy_env.parent])
        with pytest.raises(ValueError, match="nothing to screen"):
            T.measure_round(toy, pool, frozen, batch_size=4)

    def test_the_dataset_is_generated_by_no_proposer(self, toy):
        # The declared deviation: the screen is fitted on random mutagenesis at
        # anchor A, so it favours no arm. A run under any other dataset must be
        # distinguishable from this one in the store.
        fitted = T.fit_shared_surrogate(
            toy, toy.feasible_sequence(0), seed=0, rounds=2, batch_size=4
        )
        assert fitted.dataset == T.INDEPENDENT_DATASET
        assert fitted.oracle_calls == 8
        assert fitted.surrogate.is_fitted

    def test_every_arm_reads_the_same_object(self, toy):
        # The ruling is about object identity, not about two models that happen
        # to agree. Asserting identity is what makes "screening is provably
        # identical across arms" a fact rather than a hope.
        prep = T.prepare_seed(toy, 0, TOY_BUDGET)
        pair = T.anchor_pair(
            toy,
            seed=0,
            level="near",
            near_distance=TOY_BUDGET.near_distance,
            far_distance=TOY_BUDGET.far_distance,
        )
        frozen = T.FrozenSurrogate(prep.surrogate.surrogate)
        arms = T.arms_at_anchor(toy, prep, pair, frozen, TOY_BUDGET)
        assert set(arms) == set(T.ARM_NAMES)
        assert frozen.inner is prep.surrogate.surrogate


class TestWhatTheStoreEndsUpHolding:
    """The probe's numbers have to be fingerprinted and re-runnable like the rest."""

    def test_every_arm_is_stored_at_every_level(self, probe_store):
        store, ran = probe_store
        assert ran == len(T.ARM_NAMES) * len(T.LEVELS)
        for level in T.LEVELS:
            assert sorted(store.methods(T.task_name(level))) == sorted(T.ARM_NAMES)

    def test_a_second_run_runs_nothing(self, probe_store, toy):
        # Without this the probe is not resumable, and a run killed at hour six
        # would repeat every campaign it had already paid for.
        store, _ = probe_store
        again = T.run_transfer_probe(
            store, [0], landscape=toy, budget=TOY_BUDGET, report=lambda _: None
        )
        assert again == 0

    def test_the_records_are_current_against_the_probe(self, probe_store):
        # A record stamped through `suite.RESULT_DEPENDENCIES` could not notice an
        # edit to the arm that produced it, because that closure does not reach
        # this module. These declare their own.
        store, _ = probe_store
        for level in T.LEVELS:
            for name in T.ARM_NAMES:
                record = store.load(T.task_name(level), name)[0]
                assert "evogfn.benchmark.transfer" in record.source
                assert not store.stale(T.task_name(level), name)

    def test_one_screen_is_visible_in_the_store(self, probe_store):
        # The shared-screen ruling, checkable years later from the records alone
        # rather than only from the identity assertion above.
        store, _ = probe_store
        for level in T.LEVELS:
            digests = {
                store.load(T.task_name(level), name)[0].parameters["surrogate_dataset_digest"]
                for name in T.ARM_NAMES
            }
            predictions = {
                store.load(T.task_name(level), name)[0].parameters["surrogate_prediction_digest"]
                for name in T.ARM_NAMES
            }
            datasets = {
                store.load(T.task_name(level), name)[0].parameters["surrogate_dataset"]
                for name in T.ARM_NAMES
            }
            assert len(digests) == 1
            assert len(predictions) == 1
            assert datasets == {T.INDEPENDENT_DATASET}

    def test_the_achieved_distance_is_recorded(self, probe_store):
        # The specification asks for achieved distances rather than nominal ones,
        # and a reader cannot recover them from anything else in the record.
        store, _ = probe_store
        for level, expected in zip(
            T.LEVELS, [TOY_BUDGET.near_distance, TOY_BUDGET.far_distance], strict=True
        ):
            record = store.load(T.task_name(level), "plain-transferred")[0]
            assert record.parameters["achieved_distance"] == float(expected)

    def test_the_conditioned_arm_records_where_its_buffer_sat(self, probe_store):
        # This is the post-hoc tell for the stale anchor: a run whose buffer never
        # moved reports a distance of zero here while every other column looks
        # entirely normal.
        store, _ = probe_store
        for level, expected in zip(
            T.LEVELS, [TOY_BUDGET.near_distance, TOY_BUDGET.far_distance], strict=True
        ):
            record = store.load(T.task_name(level), "conditioned-transferred")[0]
            assert record.parameters["policy_anchor_distance"] == float(expected)

    def test_arms_without_a_policy_record_no_anchor_distance(self, probe_store):
        # An absent key is not a zero. Storing zero for an arm that holds no
        # anchor would make the stale-anchor tell unreadable, since zero is
        # exactly what a stale run reports.
        store, _ = probe_store
        record = store.load(T.task_name("near"), "genetic-rebuilt")[0]
        assert "policy_anchor_distance" not in record.parameters

    def test_only_the_retrained_arm_trains_at_b(self, probe_store):
        store, _ = probe_store
        for name in T.ARM_NAMES:
            record = store.load(T.task_name("near"), name)[0]
            steps = record.parameters["gradient_steps_at_b"]
            assert (steps > 0) == (name == "plain-retrained")
            assert record.parameters["frozen"] == (name != "plain-retrained")

    def test_the_probe_is_marked_diagnostic(self, probe_store):
        # One landscape, one instance: these rows license no comparative claim
        # about method ranking, and the provenance string is what stops a
        # downstream table promoting them into one.
        store, _ = probe_store
        record = store.load(T.task_name("near"), "plain-transferred")[0]
        assert "diagnostic" in record.protocol


class TestWhatTheProbeMustNotDisturb:
    """The reason this is a new module and not an edit."""

    def test_no_stored_campaign_can_reach_the_probe(self):
        # The load-bearing invariant. `dependency_closure` only reaches a module
        # something imports, so the probe is fingerprint-inert exactly as long as
        # nothing in the suite's entry points imports it -- and the day one does,
        # every edit to the probe makes roughly sixteen thousand records stale.
        assert "evogfn.benchmark.transfer" not in dependency_closure(RESULT_DEPENDENCIES)

    def test_the_probes_policy_matches_the_suites(self, toy_env):
        # `_policy` is a deliberate mirror of the private `methods._policy`. A
        # drift in sizing would make the probe's policies a different network
        # from the suite's while both kept the same arm name and the same
        # configuration string.
        mine = T._policy(toy_env, hidden_dim=8, seed=5, anchor_conditioned=False)
        theirs = methods._policy(toy_env, hidden_dim=8, learn_flow=True, seed=5)
        assert set(mine.state_dict()) == set(theirs.state_dict())
        for key, value in mine.state_dict().items():
            assert torch.equal(value, theirs.state_dict()[key])

    def test_the_conditioned_mirror_matches_too(self, toy_env):
        mine = T._policy(toy_env, hidden_dim=8, seed=5, anchor_conditioned=True)
        theirs = methods._policy(
            toy_env, hidden_dim=8, learn_flow=True, seed=5, anchor_conditioned=True
        )
        assert isinstance(mine, AnchorConditionedPolicy)
        for key, value in mine.state_dict().items():
            assert torch.equal(value, theirs.state_dict()[key])


class TestWhatTheArmsAreConfiguredAs:
    """Settings that a report will quote, checked against what actually runs."""

    def test_the_configuration_string_names_the_settings_that_run(self):
        # The probe runs the configuration the selection phase chose. Parsing the
        # name here rather than at run time keeps a single place to be wrong, and
        # this is the test that catches it being wrong.
        parsed = Configuration.parse(T.TRANSFER_CONFIGURATION)
        assert parsed.beta == T.TRANSFER_BETA
        assert parsed.steps == T.TRANSFER_STEPS
        assert parsed.lam == T.TRANSFER_LAMBDA
        assert parsed.hidden_dim == T.TRANSFER_HIDDEN_DIM

    def test_the_capacity_control_over_resources_the_arm_it_controls_for(self):
        # No integer width matches the conditioned policy exactly, so the honest
        # control carries *more* parameters and criterion 3 is tested against a
        # better-resourced comparator. A report describing the two as
        # parameter-matched would contradict this number, which is stored on
        # every record of that arm.
        landscape = T.diagnostic_landscape()
        env = T.training_environment(landscape, landscape.feasible_sequence(0))
        conditioned = T._policy(
            env, hidden_dim=T.TRANSFER_HIDDEN_DIM, seed=0, anchor_conditioned=True
        )
        wide = T._policy(env, hidden_dim=T.CAPACITY_HIDDEN_DIM, seed=0, anchor_conditioned=False)
        counted = sum(p.numel() for p in wide.parameters()) - sum(
            p.numel() for p in conditioned.parameters()
        )
        assert counted == T.CAPACITY_RESIDUAL
        assert counted > 0

    def test_no_plain_width_matches_exactly(self):
        # States the arithmetic the deviation rests on: 101 is the closest
        # integer width and 100 undershoots, so there is no width that would have
        # made the control exact.
        landscape = T.diagnostic_landscape()
        env = T.training_environment(landscape, landscape.feasible_sequence(0))
        target = sum(
            p.numel()
            for p in T._policy(
                env, hidden_dim=T.TRANSFER_HIDDEN_DIM, seed=0, anchor_conditioned=True
            ).parameters()
        )
        widths = {
            h: sum(
                p.numel()
                for p in T._policy(env, hidden_dim=h, seed=0, anchor_conditioned=False).parameters()
            )
            for h in (100, T.CAPACITY_HIDDEN_DIM)
        }
        assert widths[100] < target < widths[T.CAPACITY_HIDDEN_DIM]

    def test_the_retrain_gets_a_whole_campaigns_worth_of_steps(self):
        # The ceiling arm has to match the training budget it is a ceiling for.
        # Derived rather than written as a literal, so shrinking the campaign
        # shrinks the arm with it.
        assert T.ProbeBudget().retraining_steps == T.TRANSFER_ROUNDS * T.TRANSFER_STEPS
        assert TOY_BUDGET.retraining_steps == TOY_BUDGET.rounds * TOY_BUDGET.steps

    def test_the_defaults_are_the_probe_as_specified(self):
        # A budget exists so the path can be exercised at toy scale; a default
        # that had drifted from the specification would mean the thing that runs
        # is not the thing that was declared.
        budget = T.ProbeBudget()
        assert budget.near_distance == T.NEAR_DISTANCE
        assert budget.far_distance == T.FAR_DISTANCE
        assert budget.pool_size == T.TRANSFER_POOL
        assert budget.rounds == T.TRANSFER_ROUNDS


class _Reversing(Surrogate):
    """Scores a pool in reverse index order, so the screen is visible in the plate.

    A real surrogate's ranking is not predictable from the outside, so a test
    asserting "the plate was screened" against one could only ever assert that
    the plate was not the prefix -- which passes by luck whenever the surrogate
    happens to disagree with the pool order. This makes the expected plate exact.
    """

    def __init__(self, n):
        self._n = n

    @property
    def is_fitted(self):
        return True

    def fit(self, sequences, values):  # noqa: ARG002 - the signature is the point; it must never run
        raise AssertionError("nothing should fit this")

    def predict(self, sequences):
        # Ascending, so the *last* candidate scores highest and a screened plate
        # is the reversed tail. A prefix and a screen would coincide under the
        # opposite order, which is the failure this helper exists to separate.
        scores = np.arange(len(sequences), dtype=np.float64)
        return scores, np.zeros_like(scores)


def _walked(landscape, env, distance):
    """A feasible design a legal walk of ``distance`` steps reaches from ``env``."""
    walk = MutationEnvironment(
        env.parent,
        landscape.alphabet,
        max_mutations=distance,
        transitions=np.asarray(landscape.transition_matrix),
    )
    return T._legal_walk(walk, np.random.default_rng(0), distance)


def _sampler_parts(landscape, env):
    """The keyword arguments a `GFlowNetSampler` needs, for the sharing test."""
    ensemble = _fitted(landscape, env)
    return dict(  # noqa: C408 - keyword form mirrors the sampler's own signature
        proxy=ProxyLandscape(ensemble, alphabet=env.alphabet, sequence_length=env.sequence_length),
        reward=TemperedReward(beta=T.TRANSFER_BETA),
        config=TrainingConfig(steps=1, batch_size=2, seed=0),
    )
