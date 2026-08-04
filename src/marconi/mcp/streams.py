from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from marconi.mcp.workspace import conversion_cache_dir

_MAX_INLINE_BITS = 1_048_576
_MAX_PAGE_ITEMS = 65536
_CHUNK_ITEMS = 1 << 20

_STATS_SAMPLE_ITEMS = 65536
_STATS_SAMPLE_CHUNKS = 16
_STATS_MAX_CLUSTERS = 16
_STATS_MAX_BINS = 200
_STATS_KMEANS_ITERS = 100

# "l" (int64) is a paging-only type for run sidecars (windows/marks offsets),
# never an engine wire type
_SUFFIX_TYPES = {".u8": "b", ".i16": "s", ".f32": "f", ".i64": "l"}
_ITEM_DTYPES: dict[str, type] = {
    "b": np.uint8,
    "s": np.int16,
    "f": np.float32,
    "l": np.int64,
}
_RAW_SCALES: dict[str, tuple[type, float, float]] = {
    "ci16": (np.int16, 32768.0, 0.0),
    "ci8": (np.int8, 128.0, 0.0),
    "cu8": (np.uint8, 127.5, 127.5),
}


def parse_bits(text: str) -> np.ndarray:
    if not text or len(text) > _MAX_INLINE_BITS:
        raise ValueError(
            f"bits must be 1..{_MAX_INLINE_BITS} characters of '0'/'1', "
            f"got {len(text)}"
        )
    if set(text) - {"0", "1"}:
        raise ValueError("bits must contain only '0' and '1'")
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8) - ord("0")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"stream output not found at {path}; marconi-runs/ outputs are "
            f"not auto-cleaned — re-run the spec if the run was removed"
        )


def _resolve_item_type(path: Path, item_type: str | None) -> str:
    if item_type is not None:
        if item_type not in _ITEM_DTYPES:
            raise ValueError(f"item_type must be one of {sorted(_ITEM_DTYPES)}")
        return item_type
    inferred = _SUFFIX_TYPES.get(path.suffix)
    if inferred is None:
        raise ValueError(
            f"cannot infer item_type from suffix {path.suffix!r}; pass "
            f"item_type explicitly (suffixes {sorted(_SUFFIX_TYPES)} infer)"
        )
    return inferred


def render_page(
    path: Path, *, offset: int, count: int, item_type: str | None
) -> dict[str, object]:
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    _require_file(path)
    kind = _resolve_item_type(path, item_type)
    dtype = np.dtype(_ITEM_DTYPES[kind])
    total = path.stat().st_size // dtype.itemsize
    count = max(0, min(count, _MAX_PAGE_ITEMS, total - offset))
    with path.open("rb") as f:
        f.seek(offset * dtype.itemsize)
        items = np.fromfile(f, dtype=dtype, count=count)
    page: dict[str, object] = {
        "item_type": kind,
        "offset": offset,
        "count": int(items.size),
        "total_items": int(total),
    }
    if kind == "b":
        bits_array = items.astype(np.uint8)
        page["bits"] = "".join(chr(ord("0") + int(v & np.uint8(1))) for v in bits_array)
    elif kind == "s":
        page["symbols"] = [int(v) for v in items]
    elif kind == "l":
        page["values"] = [int(v) for v in items]
    else:
        page["values"] = [round(float(v), 4) for v in items]
    return page


def ensure_cf32(
    src: Path, dtype: str, *, offset: int = 0, samples: int = 0
) -> tuple[Path, int, int]:
    """Resolve a capture to (cf32 path, source offset items, source length
    items). cf32 passes through untouched - the GR file source does the
    slicing while streaming. Raw integer formats convert only the requested
    slice, into a content-keyed cache shared across runs: iterating specs
    against one capture must not write a fresh multi-GB copy per run."""
    if dtype == "cf32":
        return src, offset, samples
    if dtype not in _RAW_SCALES:
        raise ValueError(
            f"capture_dtype must be one of {sorted(('cf32', *_RAW_SCALES))}"
        )
    st = src.stat()
    # mtime_ns keying can serve stale on filesystems with coarse timestamps
    # if a capture is overwritten same-size within one tick; acceptable
    key = f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{dtype}|{offset}|{samples}"
    dest = conversion_cache_dir() / (
        hashlib.sha1(key.encode()).hexdigest()[:24] + ".cf32"
    )
    if dest.is_file():
        return dest, 0, 0
    raw_dtype, scale, bias = _RAW_SCALES[dtype]
    raw_size = np.dtype(raw_dtype).itemsize
    remaining = samples * 2 if samples > 0 else None
    with src.open("rb") as fin:
        fin.seek(offset * 2 * raw_size)
        # mkstemp: concurrent converters of the same capture (FastMCP runs
        # tools on worker threads) must never share a tmp path - the loser
        # would publish interleaved garbage into the immortal cache
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".cf32.tmp")
        try:
            with os.fdopen(fd, "wb") as fout:
                while True:
                    want = (
                        _CHUNK_ITEMS * 2
                        if remaining is None
                        else min(_CHUNK_ITEMS * 2, remaining)
                    )
                    if want == 0:
                        break
                    block: np.ndarray = np.fromfile(fin, dtype=raw_dtype, count=want)
                    if block.size == 0:
                        break
                    if remaining is not None:
                        remaining -= int(block.size)
                    pairs = block[: block.size - (block.size % 2)].astype(np.float32)
                    pairs = (pairs - bias) / scale
                    (pairs[0::2] + 1j * pairs[1::2]).astype(np.complex64).tofile(fout)
            os.replace(tmp_name, dest)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    return dest, 0, 0


def _sample_stream(path: Path, dtype: np.dtype) -> tuple[np.ndarray, int, bool]:
    total = path.stat().st_size // dtype.itemsize
    with path.open("rb") as f:
        if total <= _STATS_SAMPLE_ITEMS:
            return np.fromfile(f, dtype=dtype), total, False
        per = _STATS_SAMPLE_ITEMS // _STATS_SAMPLE_CHUNKS
        starts = np.linspace(0, total - per, _STATS_SAMPLE_CHUNKS).astype(np.int64)
        parts = []
        for s in starts:
            f.seek(int(s) * dtype.itemsize)
            parts.append(np.fromfile(f, dtype=dtype, count=per))
    return np.concatenate(parts), total, True


def _kmeans_1d(x: np.ndarray, k: int) -> np.ndarray:
    lo, hi = (float(v) for v in np.percentile(x, [1.0, 99.0]))
    if hi <= lo:
        hi = lo + 1.0
    xc = np.clip(x, lo, hi)
    seeds = np.percentile(xc, np.linspace(0.0, 1.0, k + 2)[1:-1] * 100.0)
    centers = np.unique(seeds)
    if centers.size < k:
        centers = np.linspace(lo, hi, k)
    for _ in range(_STATS_KMEANS_ITERS):
        labels = _nearest_labels(xc, centers)
        moved = np.array(
            [
                xc[labels == j].mean() if np.any(labels == j) else centers[j]
                for j in range(k)
            ]
        )
        if np.allclose(moved, centers):
            break
        centers = moved
    return np.sort(centers)


def _nearest_labels(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.abs(values[:, None] - centers[None, :]).argmin(1)


def stream_stats(
    path: Path, *, item_type: str | None, clusters: int, bins: int
) -> dict[str, object]:
    _require_file(path)
    kind = _resolve_item_type(path, item_type)
    if not 1 <= bins <= _STATS_MAX_BINS:
        raise ValueError(f"bins must be 1..{_STATS_MAX_BINS}, got {bins}")
    if not 0 <= clusters <= _STATS_MAX_CLUSTERS:
        raise ValueError(f"clusters must be 0..{_STATS_MAX_CLUSTERS}, got {clusters}")
    dtype = np.dtype(_ITEM_DTYPES[kind])
    sample, total, sampled = _sample_stream(path, dtype)
    out: dict[str, object] = {
        "item_type": kind,
        "total_items": int(total),
        "sampled_items": int(sample.size),
        "sampled": sampled,
    }
    if kind == "b":
        ones = float((sample & np.uint8(1)).mean()) if sample.size else 0.0
        out["ones_fraction"] = round(ones, 6)
        return out
    x = sample.astype(np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("stream has no finite items to summarize")
    out["sampled_items"] = int(x.size)
    out.update(
        min=round(float(x.min()), 6),
        max=round(float(x.max()), 6),
        mean=round(float(x.mean()), 6),
        std=round(float(x.std()), 6),
    )
    lo, hi = (float(v) for v in np.percentile(x, [0.5, 99.5]))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(np.clip(x, lo, hi), bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2.0
    out["histogram"] = [
        {"center": round(float(c), 6), "count": int(n)} for c, n in zip(mids, counts)
    ]
    if clusters >= 1:
        centers = _kmeans_1d(x, clusters)
        labels = _nearest_labels(x, centers)
        cluster_counts = np.array([int((labels == j).sum()) for j in range(clusters)])
        keep = cluster_counts > 0
        centers = centers[keep]
        cluster_counts = cluster_counts[keep]
        rounded = [round(float(v), 6) for v in centers]
        out["centers"] = rounded
        out["cluster_counts"] = [int(n) for n in cluster_counts]
        out["levels"] = list(rounded)
    return out
