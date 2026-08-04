"""Tests for the result store, and above all for when it refuses a cached run.

The store's job is to be right about staleness in both directions. Handing back
a result produced by code that has since changed silently mixes versions inside
one table; re-running a result that could not have changed costs hours. Both
failures are cheap to write and expensive to notice, so the import-graph walk
that decides between them is tested against a synthetic package where the
answer is known by construction.
"""

import json
import multiprocessing
from pathlib import Path
from typing import Any

import pytest

from evogfn.benchmark.store import (
    ResultStore,
    RunRecord,
    dependency_closure,
    fingerprint,
    package_fingerprint,
)

# A package whose import graph is small enough to reason about by eye. Each
# module exists to pin one behaviour of the walk.
SOURCES = {
    "__init__.py": "",
    "alpha.py": "from pkg import beta\nfrom pkg.deep import gamma\n",
    "beta.py": "import pkg.leaf\n",
    "leaf.py": "import json\n",
    "lonely.py": "VALUE = 1\n",
    "deep/__init__.py": "",
    "deep/gamma.py": "from . import sibling\nfrom ..beta import THING\n",
    "deep/sibling.py": "",
    "cyclic_a.py": "import pkg.cyclic_b\n",
    "cyclic_b.py": "import pkg.cyclic_a\n",
    "typed.py": (
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from pkg import lonely\n"
    ),
    "ambiguous.py": "from pkg.deep import gamma, Thing\n",
    "algorithms/__init__.py": "",
    "algorithms/genetic.py": "from pkg import leaf\n",
}

# Splatted into RunRecord and ResultStore.stamp, which is why the values are
# Any: inferred as `object` the splat fails every field's declared type.
RECORD_FIELDS: dict[str, Any] = {
    "task": "t",
    "method": "m",
    "seed": 0,
    "protocol": "P",
    "best": 1.0,
    "regret": 0.0,
    "diversity": 0.5,
    "feasible_fraction": 1.0,
    "oracle_calls": 10,
    "proposals": 20,
}


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """A synthetic package, installed as the thing the store fingerprints."""
    root = tmp_path / "pkg"
    for name, body in SOURCES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    monkeypatch.setattr("evogfn.benchmark.store._package_root", lambda: root)
    return root


@pytest.fixture
def store(pkg, tmp_path):
    """A store over the synthetic package."""
    del pkg
    return ResultStore(tmp_path / "results")


def test_fingerprint_is_one_entry_per_module(pkg):
    del pkg
    digests = fingerprint()
    assert set(digests) == {
        "pkg",
        "pkg.algorithms",
        "pkg.algorithms.genetic",
        "pkg.alpha",
        "pkg.ambiguous",
        "pkg.beta",
        "pkg.cyclic_a",
        "pkg.cyclic_b",
        "pkg.deep",
        "pkg.deep.gamma",
        "pkg.deep.sibling",
        "pkg.leaf",
        "pkg.lonely",
        "pkg.typed",
    }
    assert len(digests) == len(SOURCES)


def test_fingerprint_is_stable_and_content_addressed(pkg):
    before = fingerprint()
    assert fingerprint() == before
    (pkg / "lonely.py").write_text("VALUE = 2\n")
    after = fingerprint()
    assert after["pkg.lonely"] != before["pkg.lonely"]
    assert {k: v for k, v in after.items() if k != "pkg.lonely"} == {
        k: v for k, v in before.items() if k != "pkg.lonely"
    }


def test_closure_is_transitive(pkg):
    del pkg
    assert dependency_closure(["pkg.alpha"]) == (
        "pkg.alpha",
        "pkg.beta",
        "pkg.deep.gamma",
        "pkg.deep.sibling",
        "pkg.leaf",
    )


def test_closure_excludes_unreachable_modules(pkg):
    del pkg
    assert "pkg.lonely" not in dependency_closure(["pkg.alpha"])


def test_closure_ignores_external_imports(pkg):
    del pkg
    assert dependency_closure(["pkg.leaf"]) == ("pkg.leaf",)


def test_relative_imports_resolve(pkg):
    del pkg
    # `from . import sibling` and `from ..beta import THING`, resolved against
    # the package containing pkg/deep/gamma.py.
    closure = dependency_closure(["pkg.deep.gamma"])
    assert "pkg.deep.sibling" in closure
    assert "pkg.beta" in closure


def test_type_checking_imports_are_included(pkg):
    del pkg
    assert "pkg.lonely" in dependency_closure(["pkg.typed"])


def test_cycles_terminate(pkg):
    del pkg
    assert dependency_closure(["pkg.cyclic_a"]) == ("pkg.cyclic_a", "pkg.cyclic_b")


def test_from_import_prefers_the_submodule_and_falls_back_to_the_package(pkg):
    del pkg
    # `from pkg.deep import gamma, Thing`: gamma is a module, Thing is not, so
    # the first resolves to the submodule and the second to pkg.deep itself.
    closure = dependency_closure(["pkg.ambiguous"])
    assert "pkg.deep.gamma" in closure
    assert "pkg.deep" in closure


def test_unknown_entry_point_is_an_error(pkg):
    del pkg
    with pytest.raises(ValueError, match="no such module"):
        dependency_closure(["pkg.nonexistent"])


def test_stamp_stores_only_the_declared_closure(store):
    record = store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS)
    assert set(record.source) == set(dependency_closure(["pkg.alpha"]))


def test_unrelated_change_leaves_a_record_usable(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    (pkg / "lonely.py").write_text("VALUE = 99\n")

    reopened = ResultStore(store.root)
    assert set(reopened.usable("t", "m")) == {0}
    assert reopened.missing("t", "m", [0]) == []
    assert reopened.stale("t", "m") == {}


def test_change_to_a_depended_module_marks_a_record_stale(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    # Two hops from the entry point: alpha imports beta imports leaf.
    (pkg / "leaf.py").write_text("import json\n\nVALUE = 2\n")

    reopened = ResultStore(store.root)
    assert reopened.usable("t", "m") == {}
    assert reopened.missing("t", "m", [0]) == [0]
    assert reopened.stale("t", "m") == {0: ("pkg.leaf",)}


def test_a_deleted_dependency_marks_a_record_stale(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    (pkg / "leaf.py").unlink()

    assert ResultStore(store.root).stale("t", "m") == {0: ("pkg.leaf",)}


def test_stamp_without_depends_on_falls_back_to_everything(pkg, store):
    record = store.stamp(**RECORD_FIELDS)
    assert set(record.source) == set(fingerprint())
    # The root package hashes under a bare name, which must not be mistaken
    # for the bare package names the old scheme used.
    assert "pkg" in record.source
    assert not record.per_package

    store.append(record)
    (pkg / "lonely.py").write_text("VALUE = 99\n")
    assert ResultStore(store.root).stale("t", "m") == {0: ("pkg.lonely",)}


def test_a_record_without_a_fingerprint_is_treated_as_current(store):
    store.append(RunRecord(**RECORD_FIELDS))
    assert set(store.usable("t", "m")) == {0}


def test_cost_and_duplicate_fields_survive_a_round_trip(store):
    """Guards a cost column that reads back as zero for every arm.

    `asdict` is what `append` serialises through and `RunRecord(**payload)` is
    what `load` rebuilds with, so a field the dataclass gained but the JSON
    round trip mangles would leave the whole compute comparison silently flat.

    The arm's settings ride the same path, and they carry mixed types -- a
    name, a count, a flag -- because that is what a resolved arm is. A record
    that lost them would put the suite back to reading a reward exponent out of
    an arm's name, which is the parse that already went wrong once.
    """
    store.append(
        store.stamp(
            depends_on=["pkg.alpha"],
            **RECORD_FIELDS,
            cpu_seconds=612.5,
            wall_seconds=1840.25,
            duplicate_fraction=0.125,
            parameters={
                "family": "gflownet",
                "objective": "TrajectoryBalance",
                "steps": 300,
                "beta": 3.0,
                "learn_flow": False,
            },
        )
    )

    record = ResultStore(store.root).load("t", "m")[0]
    assert record.cpu_seconds == 612.5
    assert record.wall_seconds == 1840.25
    assert record.duplicate_fraction == 0.125
    assert record.parameters == {
        "family": "gflownet",
        "objective": "TrajectoryBalance",
        "steps": 300,
        "beta": 3.0,
        "learn_flow": False,
    }
    # A flag has to come back a flag. JSON keeps `false` distinct from `0`, and
    # a reader asking whether an arm had a flow head must not be handed a count.
    assert record.parameters["learn_flow"] is False


def test_a_record_written_before_the_cost_fields_still_loads(store):
    """Guards the silent loss of every campaign already on disk.

    `load` treats a `TypeError` from the constructor as a partial line and
    skips it, so a new field that is not defaulted turns 8,000 stored campaigns
    into 8,000 "corrupt" lines -- with no error, just an empty store and a
    suite that reruns for days.
    """
    path = store.root / "t"
    path.mkdir(parents=True, exist_ok=True)
    (path / "m.jsonl").write_text(json.dumps(RECORD_FIELDS) + "\n")

    record = store.load("t", "m")[0]
    assert record.cpu_seconds == 0.0
    assert record.wall_seconds == 0.0
    assert record.duplicate_fraction == 0.0
    # Empty, and empty means "this record does not say". It must not be read as
    # an arm that had no settings, and the default is what keeps every record
    # written before the field loadable at all.
    assert record.parameters == {}


def test_a_record_carrying_an_unknown_field_is_dropped(store):
    """Pins why a stored field may be added but never renamed.

    An absent key takes the dataclass default; an *unexpected* one raises
    `TypeError` and is swallowed as a partial line. So renaming a field does
    not migrate the store, it deletes the half of it written under the old
    name, and it does so without saying a word. This is that behaviour stated
    out loud rather than left to be rediscovered.
    """
    path = store.root / "t"
    path.mkdir(parents=True, exist_ok=True)
    (path / "m.jsonl").write_text(json.dumps({**RECORD_FIELDS, "cpu_time": 1.0}) + "\n")

    assert store.load("t", "m") == {}


def write_legacy(store, source):
    """Append a record in the superseded per-package format."""
    path = store.root / "t"
    path.mkdir(parents=True, exist_ok=True)
    (path / "m.jsonl").write_text(json.dumps({**RECORD_FIELDS, "source": source}) + "\n")


def test_old_format_records_still_load_and_are_judged_current(pkg, store):
    del pkg
    write_legacy(store, package_fingerprint())

    reopened = ResultStore(store.root)
    assert reopened.load("t", "m")[0].per_package
    assert set(reopened.usable("t", "m")) == {0}


def test_old_format_records_go_stale_on_a_package_change(pkg, store):
    write_legacy(store, package_fingerprint())
    (pkg / "algorithms" / "genetic.py").write_text("from pkg import leaf\n\nVALUE = 2\n")

    assert ResultStore(store.root).stale("t", "m") == {0: ("algorithms",)}


def test_old_format_records_do_not_match_module_hashes(pkg, store):
    del pkg, store
    # The two schemes share no keys, so a legacy record compared against the
    # per-module fingerprint would trivially look current. It must not.
    record = RunRecord(**RECORD_FIELDS, source={"algorithms": "0000000000000000"})
    assert record.per_package
    assert record.stale_against(fingerprint()) == ("algorithms",)


def test_bless_restores_a_stale_record_without_widening_it(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    declared = set(store.load("t", "m")[0].source)
    (pkg / "leaf.py").write_text("import json\n\nVALUE = 2\n")

    reopened = ResultStore(store.root)
    assert reopened.bless("t", "m", modules=["pkg.leaf"]) == 1
    assert set(reopened.usable("t", "m")) == {0}
    assert set(reopened.load("t", "m")[0].source) == declared


def test_bless_keeps_an_old_format_record_in_its_own_format(pkg, store):
    write_legacy(store, package_fingerprint())
    (pkg / "algorithms" / "genetic.py").write_text("VALUE = 2\n")

    reopened = ResultStore(store.root)
    assert reopened.bless("t", "m", modules=["algorithms"]) == 1
    assert reopened.stale("t", "m") == {}
    assert set(reopened.load("t", "m")[0].source) == {"algorithms"}


def test_bless_leaves_unnamed_modules_stale(pkg, store):
    # The failure this exists for: a caller means to wave through one edit and
    # restamps the whole closure, so an unrelated change that did alter the run
    # comes back as current. Nine arms of 900 campaigns kept a `proxy_calls` of
    # zero that way, against code that measures 57,600.
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    (pkg / "leaf.py").write_text("import json\n\nVALUE = 2\n")
    (pkg / "deep" / "sibling.py").write_text("VALUE = 2\n")

    reopened = ResultStore(store.root)
    assert reopened.bless("t", "m", modules=["pkg.leaf"]) == 1
    assert reopened.stale("t", "m") == {0: ("pkg.deep.sibling",)}
    assert reopened.usable("t", "m") == {}


def test_bless_records_what_it_waved_through(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    (pkg / "leaf.py").write_text("import json\n\nVALUE = 2\n")
    ResultStore(store.root).bless("t", "m", modules=["pkg.leaf"])

    # A record that is current because somebody said so has to be readable as
    # such afterwards, or a suspect column can only be traced by file
    # timestamps -- which is how this was found.
    reopened = ResultStore(store.root)
    assert reopened.blessed("t", "m") == {0: ("pkg.leaf",)}
    assert "1 blessed: ['pkg.leaf']" in reopened.summarise()


def test_bless_accumulates_across_calls(pkg, store):
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))
    (pkg / "leaf.py").write_text("import json\n\nVALUE = 2\n")
    ResultStore(store.root).bless("t", "m", modules=["pkg.leaf"])
    (pkg / "deep" / "sibling.py").write_text("VALUE = 2\n")
    ResultStore(store.root).bless("t", "m", modules=["pkg.deep.sibling"])

    assert ResultStore(store.root).blessed("t", "m") == {0: ("pkg.deep.sibling", "pkg.leaf")}


def test_bless_reports_nothing_when_the_named_module_did_not_change(pkg, store):
    del pkg
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))

    # Nothing was overridden, so nothing is claimed: a record that was already
    # current must not come back carrying an assertion nobody made.
    reopened = ResultStore(store.root)
    assert reopened.bless("t", "m", modules=["pkg.leaf"]) == 0
    assert reopened.blessed("t", "m") == {}


def test_bless_refuses_to_name_nothing(pkg, store):
    del pkg
    store.append(store.stamp(depends_on=["pkg.alpha"], **RECORD_FIELDS))

    with pytest.raises(ValueError, match="requires the modules"):
        store.bless("t", "m", modules=[])


def test_bless_leaves_an_unfingerprinted_record_alone(store):
    path = store.root / "t"
    path.mkdir(parents=True, exist_ok=True)
    (path / "m.jsonl").write_text(json.dumps(RECORD_FIELDS) + "\n")

    # No fingerprint means the record predates the mechanism and is already read
    # as current. Stamping it here would newly pin it to a tree it never ran
    # under, and record an assertion about code nobody claimed anything about.
    reopened = ResultStore(store.root)
    assert reopened.bless("t", "m", modules=["pkg.leaf"]) == 0
    assert reopened.load("t", "m")[0].source == {}
    assert reopened.blessed("t", "m") == {}


def _hammer(args):
    """One writer process, appending oversized records to a shared arm file."""
    root, tag, count = args
    store = ResultStore(Path(root))
    for seed in range(count):
        store.append(
            RunRecord(
                task="race",
                method="arm",
                seed=seed * 100 + tag,
                protocol="p",
                best=1.0,
                regret=0.0,
                diversity=1.0,
                feasible_fraction=1.0,
                oracle_calls=1,
                proposals=1,
                # Deliberately far past the size at which the kernel makes an
                # append atomic. A small record would pass this test whether or
                # not the lock existed, which is how the bug reached the store
                # in the first place.
                top_sequences=[[i % 7 for i in range(600)] for _ in range(10)],
            )
        )
    return tag


def test_concurrent_writers_never_tear_a_record(tmp_path):
    """Guards the failure that is silent in both directions.

    Sharding a suite by seed puts several processes on one arm's file. A record
    here runs to tens of kilobytes against a four-kilobyte atomicity limit, so
    without a lock two appends interleave and leave a line that parses as
    nothing -- and `load` skips what it cannot parse, so the campaign does not
    come back damaged, it simply disappears and the seed looks unrun.
    """
    writers, each = 4, 15
    with multiprocessing.Pool(writers) as pool:
        pool.map(_hammer, [(str(tmp_path), tag, each) for tag in range(writers)])

    lines = [
        line for line in (tmp_path / "race" / "arm.jsonl").read_text().splitlines() if line.strip()
    ]
    for line in lines:
        json.loads(line)  # raises if any record was torn
    assert len(lines) == writers * each
