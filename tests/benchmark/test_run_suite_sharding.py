"""Sharding a tier must not change what the tier is.

Parallelism here has to be by process -- determinism pins each one to a single
thread -- so a sweep is cut into seed ranges and run as subprocesses. That is a
*scheduling* decision, and the property that makes it one is that every campaign
is seeded from its own seed rather than from process order: a sharded run and a
serial one write identical records.

What can still go wrong is the cut itself. A seed dropped between two ranges is
a campaign nobody runs and nobody misses, and a seed in two ranges is a campaign
run twice whose second write silently replaces the first. Both are invisible
downstream -- the store keys by seed, so neither leaves a mark -- which is why
the arithmetic is pinned here rather than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))

from run_suite import _shard_bounds  # type: ignore[import-not-found]


@pytest.mark.parametrize("count", [0, 1, 3, 7, 30, 100])
@pytest.mark.parametrize("workers", [1, 2, 3, 4, 11, 64])
def test_the_shards_cover_every_seed_exactly_once(count, workers):
    bounds = _shard_bounds(count, workers)
    covered = [seed for low, high in bounds for seed in range(low, high)]
    assert covered == list(range(count))


@pytest.mark.parametrize(("count", "workers"), [(3, 8), (1, 4), (0, 4)])
def test_more_workers_than_seeds_yields_no_empty_shard_but_one(count, workers):
    # An empty shard is a process that pays interpreter start-up to do nothing.
    # The single degenerate case is a tier with no seeds at all, which has to
    # return something for the caller to iterate.
    bounds = _shard_bounds(count, workers)
    assert len(bounds) == max(1, count)


def test_the_shards_are_contiguous_ranges_not_a_stride():
    # Contiguous so a shard's log reads as a seed range, and so a shard that
    # dies leaves an obvious hole rather than a comb nobody notices.
    bounds = _shard_bounds(100, 4)
    assert bounds == [(0, 25), (25, 50), (50, 75), (75, 100)]


def test_the_sizes_differ_by_at_most_one():
    # The remainder is spread rather than dumped on the last shard, which would
    # otherwise run long and hold the whole sweep open.
    sizes = [high - low for low, high in _shard_bounds(100, 7)]
    assert max(sizes) - min(sizes) <= 1
