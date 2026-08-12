from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from marconi.mcp.streams import ensure_cf32
from marconi.mcp.workspace import (
    conversion_cache_budget,
    conversion_cache_dir,
    evict_conversion_cache,
    touch_cache_entry,
)


@pytest.fixture(autouse=True)
def _workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))


def _entry(name: str, size: int, age_s: float) -> Path:
    p = conversion_cache_dir() / name
    p.write_bytes(b"\0" * size)
    stamp = time.time() - age_s
    os.utime(p, (stamp, stamp))
    return p


def test_budget_reads_the_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARCONI_CONVERSION_CACHE_BYTES", "4096")
    assert conversion_cache_budget() == 4096


def test_unparseable_budget_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARCONI_CONVERSION_CACHE_BYTES", "lots")
    assert conversion_cache_budget() > 0


def test_eviction_drops_least_recently_used_first() -> None:
    old = _entry("old.cf32", 1000, age_s=10_000)
    mid = _entry("mid.cf32", 1000, age_s=5_000)
    new = _entry("new.cf32", 1000, age_s=1_000)

    freed = evict_conversion_cache(budget=1500)

    assert freed == 2000
    assert not old.exists() and not mid.exists()
    assert new.exists()


def test_a_fresh_entry_is_never_evicted_out_from_under_a_starting_run() -> None:
    _entry("aged.cf32", 1000, age_s=10_000)
    fresh = conversion_cache_dir() / "fresh.cf32"
    fresh.write_bytes(b"\0" * 10_000)

    evict_conversion_cache(budget=0)

    assert fresh.exists(), "an entry inside the grace window must survive"


def test_a_cache_hit_restamps_recency() -> None:
    stale = _entry("stale.cf32", 1000, age_s=10_000)
    keep = _entry("keep.cf32", 1000, age_s=9_000)
    touch_cache_entry(stale)

    evict_conversion_cache(budget=1500)

    assert stale.exists(), "the entry just used must outrank the untouched one"
    assert not keep.exists()


def test_conversion_reuses_its_cache_entry(tmp_path: Path) -> None:
    src = tmp_path / "cap.ci16"
    np.arange(64, dtype=np.int16).tofile(src)

    first = ensure_cf32(src, "ci16")
    second = ensure_cf32(src, "ci16")

    assert first.path == second.path
    assert len(list(conversion_cache_dir().glob("*.cf32"))) == 1


def test_conversion_bounds_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aged = _entry("aged.cf32", 1_000_000, age_s=10_000)
    monkeypatch.setenv("MARCONI_CONVERSION_CACHE_BYTES", "1024")
    src = tmp_path / "cap.ci16"
    np.arange(64, dtype=np.int16).tofile(src)

    out = ensure_cf32(src, "ci16")

    assert not aged.exists(), "the over-budget entry must be evicted before writing"
    assert out.path.is_file()
