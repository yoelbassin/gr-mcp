from __future__ import annotations

import os
import stat
import time
import uuid
from pathlib import Path

# The cf32 conversion cache trades disk for the re-run loop: iterating specs
# against one capture must not re-convert it. Unbounded, that trade is a leak —
# every distinct (offset, samples) pair writes another full copy of the slice,
# and an integer capture quadruples in size on the way in. Entries are evicted
# least-recently-used down to this budget; a run's own input is protected by
# _EVICT_GRACE_S so eviction can never pull a file out from under an in-flight
# decode that has been handed the path but not yet opened it.
_CACHE_BUDGET_BYTES = 8 * 1024**3
_EVICT_GRACE_S = 300.0


def workspace_root() -> Path:
    return Path(os.environ.get("MARCONI_WORKSPACE", ".")).resolve()


def new_run_dir(label: str) -> Path:
    d = workspace_root() / "marconi-runs" / f"{label}-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=False)
    return d


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


def evict_conversion_cache(budget: int | None = None) -> int:
    """Delete least-recently-used cache entries until the directory fits its
    budget. Returns the bytes freed. Entries younger than the grace window are
    never candidates: they belong to runs that may still be starting."""
    limit = conversion_cache_budget() if budget is None else budget
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
        if st.st_mtime < cutoff:
            entries.append((st.st_mtime, st.st_size, p))
    freed = 0
    for _, size, p in sorted(entries):
        if total - freed <= limit:
            break
        try:
            p.unlink()
        except OSError:
            continue
        freed += size
    return freed
