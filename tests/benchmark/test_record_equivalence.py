"""A stored record must survive a refactor of the code that produced it.

This test exists for one operation: the fingerprinted closure is 48 modules over
37,762 stored records, so *any* edit to `env/mutation.py`, `loop/campaign.py`,
`benchmark/methods.py` or `benchmark/tasks.py` marks every one of them stale.
The escape hatch is [bless][evogfn.benchmark.store.ResultStore.bless], which
restamps a record as current for a named module -- and a bless is only honest if
somebody has checked that the change could not have moved the number.

That check is what this file automates. It re-runs a handful of committed cells
through the *production* path -- `run_task` into a throwaway store, not a
hand-rolled campaign -- and asserts the fresh record agrees with the committed
one on every field that is not a clock or a fingerprint. Run it before a closure
edit to confirm the cells reproduce, and again after, to confirm the edit was
behaviour-preserving. A field that moves is a bless refused.

The roster is chosen for coverage of the axes a feasibility refactor could break
rather than for size: a constrained re-anchoring task, a constrained fixed-anchor
task, and an unconstrained empirical one; a learned arm and a classical arm on
each. Marked `slow` because a GFlowNet campaign is a couple of hundred seconds
and this is an operation run twice per refactor, not on every commit.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import fields
from math import isnan
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from run_suite import methods_for, tiers  # type: ignore[import-not-found]

from evogfn.benchmark.determinism import configure_determinism
from evogfn.benchmark.store import ResultStore
from evogfn.benchmark.suite import run_task

#: Where the committed records live.
RESULTS = Path(__file__).resolve().parents[2] / "results"

#: Fields that legitimately differ between two runs of the same campaign: two
#: clocks, the fingerprint of the code that produced it, and the bless ledger.
#: Everything else is the campaign's output and must be identical.
#:
#: ``blessed`` is here because it is *provenance about* the record rather than
#: anything the campaign measured -- a committed record carries the modules a
#: past bless waved through, and a record made fresh from current code has waved
#: nothing through and carries an empty list. Comparing it would fail every cell
#: that has ever been blessed, which is the majority, and would fail them for a
#: reason that has nothing to do with whether the number moved.
NOT_COMPARED = frozenset({"cpu_seconds", "wall_seconds", "source", "depends_on", "blessed"})

#: The cells re-run, and why each is here. One learned arm and one classical arm
#: on each of the three shapes a feasibility change could act on differently:
#: constrained + re-anchored, constrained + fixed anchor, unconstrained.
CELLS = [
    ("protocol-alde", "gfn-subtb@b0.1-s300-l0.9-h64", 0),
    ("protocol-alde", "genetic", 0),
    # Re-anchoring *and* filtering on feasibility. Added after a refactor broke
    # exactly that combination -- a re-anchored environment carried the
    # predicate but not the transition matrix, so a reachability test written
    # against the matrix stopped constraining anything from round two onward --
    # and every other cell in this roster passed anyway. A cell that moves its
    # anchor and rejects on feasibility is the one that notices.
    ("protocol-alde", "genetic-feasible", 0),
    ("feasibility", "gfn-subtb@b0.1-s300-l0.9-h64", 0),
    ("feasibility", "genetic-feasible", 0),
    ("gb1-anchor", "gfn-subtb@b0.1-s300-l0.9-h64", 0),
    ("gb1-anchor", "cmaes", 0),
]


def _cell(task_name: str, arm: str):
    """The task object and arm builder a tier resolves for this cell.

    Read off the tier declarations rather than rebuilt, so this test pins what
    the suite actually runs. A cell naming a task or arm no tier resolves fails
    here instead of silently testing nothing.

    Args:
        task_name: The task to find.
        arm: The methodology name wanted on it.

    Returns:
        A ``(task, methods)`` pair, where ``methods`` holds only ``arm``.
    """
    for tier in tiers(main_seeds=1, diagnostic_seeds=1):
        for task in tier.tasks:
            if task.name != task_name:
                continue
            arms = methods_for(tier)
            if arm in arms:
                return task, {arm: arms[arm]}
    raise AssertionError(f"no tier resolves {task_name}/{arm}")


def _same(committed: object, fresh: object) -> bool:
    """Whether a committed value and a freshly-computed one agree.

    Two departures from ``==``, and both are about what a *behaviour* change is
    rather than about float tolerance -- there is no tolerance here, because a
    pinned run reproduces exactly or it does not reproduce.

    ``nan != nan``, so a campaign that measured nothing would otherwise fail
    against its own re-run. And a committed record predating a *schema*
    extension carries fewer keys than a fresh one: `protocol-alde/genetic` has
    no ``duplicates`` on its round rows and no ``acquisition`` in its
    parameters, because those columns were added after it was stored and it was
    blessed forward. A key the old record never had cannot have moved, so the
    comparison runs over the keys they share and an added key is not a
    difference. A key that *disappeared* is caught, because it is then missing
    from the fresh side of a shared name.

    Args:
        committed: The stored value.
        fresh: The value the re-run produced.

    Returns:
        Whether they agree on everything the committed record actually claims.
    """
    if isinstance(committed, float) and isinstance(fresh, float):
        return committed == fresh or (isnan(committed) and isnan(fresh))
    if isinstance(committed, Mapping) and isinstance(fresh, Mapping):
        return all(k in fresh and _same(v, fresh[k]) for k, v in committed.items())
    if isinstance(committed, (list, tuple)) and isinstance(fresh, (list, tuple)):
        return len(committed) == len(fresh) and all(
            _same(a, b) for a, b in zip(committed, fresh, strict=True)
        )
    return bool(committed == fresh)


@pytest.mark.slow
@pytest.mark.parametrize(("task_name", "arm", "seed"), CELLS, ids=str)
def test_a_rerun_reproduces_the_committed_record(task_name, arm, seed, tmp_path):
    # `load`, not `usable`: after a closure edit every stored record is stale by
    # definition, and `usable` filters stale records out -- so reading through it
    # would make this test skip at exactly the moment it is supposed to run.
    committed = ResultStore(RESULTS).load(task_name, arm).get(seed)
    if committed is None:
        pytest.skip(f"{task_name}/{arm} seed {seed} is not committed, so there is nothing to pin")

    configure_determinism()
    task, methods = _cell(task_name, arm)
    fresh_store = ResultStore(tmp_path)
    run_task(task, methods, fresh_store, [seed], report=lambda _: None)

    fresh = fresh_store.usable(task_name, arm).get(seed)
    assert fresh is not None, "the re-run stored nothing"

    # `fields` rather than `vars`: the record is a slotted dataclass, so it
    # carries no instance dict to iterate.
    differences = {
        f.name: (getattr(committed, f.name), getattr(fresh, f.name))
        for f in fields(committed)
        if f.name not in NOT_COMPARED
        and not _same(getattr(committed, f.name), getattr(fresh, f.name))
    }
    assert not differences, f"{task_name}/{arm} seed {seed} moved: {differences}"
