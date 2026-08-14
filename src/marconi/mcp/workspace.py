from __future__ import annotations

import os
import shutil
import stat
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# The cf32 conversion cache trades disk for the re-run loop: iterating specs
# against one capture must not re-convert it. Unbounded, that trade is a leak —
# every distinct (offset, samples) pair writes another full copy of the slice,
# and an integer capture quadruples in size on the way in. Entries are evicted
# least-recently-used down to this budget; _EVICT_GRACE_S keeps eviction from
# pulling a file out from under an in-flight decode that has been handed the
# path but not yet opened it.
_CACHE_BUDGET_BYTES = 8 * 1024**3
_EVICT_GRACE_S = 300.0


def workspace_root() -> Path:
    """MARCONI_WORKSPACE, else a per-user cache directory - never the CWD.
    As a Claude Code plugin the server's CWD is the USER'S OWN PROJECT, and
    the "." default silently accumulated run artifacts there (measured: 18 GB
    across 711 run dirs in one checkout) that only this repo's .gitignore
    knew to ignore - the user's `git add -A` would offer all of it."""
    env = os.environ.get("MARCONI_WORKSPACE")
    if env:
        return Path(env).resolve()
    return Path.home() / ".cache" / "marconi"


def new_run_dir(label: str) -> Path:
    d = workspace_root() / "marconi-runs" / f"{label}-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=False)
    return d


@contextmanager
def discarded_if_unused(run_dir: Path) -> Iterator[Path]:
    """Remove the run directory on the way out if nothing was written into it.
    A run that failed before producing anything leaves no trace; one that wrote
    even a partial stream keeps it, because partial output is evidence. Without
    this every failing call left an empty directory behind — 29 of them in one
    working checkout.

    "Nothing written" means no FILE at any depth, not an empty top level: a
    trace=True run mkdirs run_dir/trace before the graph starts, and a plain
    rmdir then failed on the very case this exists for."""
    try:
        yield run_dir
    finally:
        try:
            if not any(p.is_file() for p in run_dir.rglob("*")):
                shutil.rmtree(run_dir)
        except OSError:
            pass


def conversion_cache_dir() -> Path:
    d = workspace_root() / "marconi-runs" / "conversions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def conversion_cache_budget() -> int:
    raw = os.environ.get("MARCONI_CONVERSION_CACHE_BYTES")
    if raw is None:
        return _CACHE_BUDGET_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return _CACHE_BUDGET_BYTES


def touch_cache_entry(path: Path) -> None:
    """Re-stamp a cache hit so recency reflects USE, not creation — otherwise
    the entry a caller keeps re-running is the first one evicted."""
    try:
        os.utime(path)
    except OSError:
        pass


def evict_conversion_cache(budget: int | None = None, incoming: int = 0) -> int:
    """Delete least-recently-used cache entries until the directory fits its
    budget, leaving room for an `incoming` entry about to be written. Returns
    the bytes freed.

    `incoming` is what makes "the budget bounds the peak" true rather than
    aspirational: evicting to the plain limit and then writing left the cache
    at limit + one slice, and a slice can be gigabytes.

    Two passes. The first spares entries inside the grace window, which is what
    keeps a concurrent run's input alive. The second runs only if the cache is
    STILL over budget and takes them too, oldest first: young entries counted
    toward the total but could never be evicted, so a burst of large slices
    inside one grace window left the budget unenforceable — the one case it
    exists for."""
    limit = max(0, (conversion_cache_budget() if budget is None else budget) - incoming)
    cutoff = time.time() - _EVICT_GRACE_S
    entries: list[tuple[float, int, Path]] = []
    total = 0
    for p in conversion_cache_dir().iterdir():
        try:
            st = p.stat()
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        total += st.st_size
        entries.append((st.st_mtime, st.st_size, p))
    entries.sort()
    freed = 0
    for young in (False, True):
        for mtime, size, p in entries:
            if total - freed <= limit:
                return freed
            if (mtime >= cutoff) is not young:
                continue
            try:
                p.unlink()
            except OSError:
                continue
            freed += size
    return freed
