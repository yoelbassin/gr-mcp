from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

import numpy as np
import numpy.typing as npt

from marconi.engine.deadline import check_deadline
from marconi.engine.io.source import SourceSlice
from marconi.levels import kmeans_1d
from marconi.mcp.workspace import conversion_cache_dir

_MAX_INLINE_BITS = 1_048_576
_MAX_PAGE_ITEMS = 65536
_CHUNK_ITEMS = 1 << 20

# default page sizes hold a default call near a few KB of JSON regardless of
# how many chars one rendered item costs; an explicit count still pages more
_DEFAULT_PAGE_ITEMS = {"b": 4096, "s": 2048, "f": 1024, "l": 1024, "c": 256}

_STATS_SAMPLE_ITEMS = 65536
_STATS_SAMPLE_CHUNKS = 16
_STATS_MAX_CLUSTERS = 16
_STATS_MAX_BINS = 200
_STATS_KMEANS_ITERS = 100

# "l" (int64) is a paging-only type for run sidecars (windows/marks offsets),
# never an engine wire type
_SUFFIX_TYPES = {".u8": "b", ".i16": "s", ".f32": "f", ".i64": "l", ".cf32": "c"}
_ITEM_DTYPES: dict[str, type] = {
    "b": np.uint8,
    "s": np.int16,
    "f": np.float32,
    "l": np.int64,
    "c": np.complex64,
}
_RAW_SCALES: dict[str, tuple[type, float, float]] = {
    "ci16": (np.int16, 32768.0, 0.0),
    "ci8": (np.int8, 128.0, 0.0),
    "cu8": (np.uint8, 127.5, 127.5),
}


def parse_bits(text: str) -> npt.NDArray[np.uint8]:
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
    path: Path, *, offset: int, count: int | None, item_type: str | None
) -> dict[str, object]:
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    _require_file(path)
    kind = _resolve_item_type(path, item_type)
    if count is None:
        count = _DEFAULT_PAGE_ITEMS[kind]
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
    page.update(_PAGE_FORMATTERS[kind](items))
    return page


def _format_bits(items: npt.NDArray[Any]) -> dict[str, object]:
    bits_array = items.astype(np.uint8)
    return {"bits": "".join(chr(ord("0") + int(v & np.uint8(1))) for v in bits_array)}


def _format_ints(key: str) -> Callable[[npt.NDArray[Any]], dict[str, object]]:
    def _format(items: npt.NDArray[Any]) -> dict[str, object]:
        return {key: [int(v) for v in items]}

    return _format


def _format_complex(items: npt.NDArray[Any]) -> dict[str, object]:
    return {
        "real": [round(float(v.real), 6) for v in items],
        "imag": [round(float(v.imag), 6) for v in items],
    }


def _format_floats(items: npt.NDArray[Any]) -> dict[str, object]:
    return {"values": [round(float(v), 4) for v in items]}


_PAGE_FORMATTERS: dict[str, Callable[[npt.NDArray[Any]], dict[str, object]]] = {
    "b": _format_bits,
    "s": _format_ints("symbols"),
    "l": _format_ints("values"),
    "c": _format_complex,
    "f": _format_floats,
}


def ensure_cf32(
    src: Path, dtype: str, *, offset: int = 0, samples: int = 0
) -> SourceSlice:
    """Resolve a capture to the cf32 SourceSlice a GR file source can stream.
    cf32 passes through untouched - the GR file source does the slicing while
    streaming. Raw integer formats convert only the requested slice, into a
    content-keyed cache shared across runs: iterating specs against one
    capture must not write a fresh multi-GB copy per run."""
    if dtype == "cf32":
        return SourceSlice(path=src, offset=offset, length=samples)
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
        return SourceSlice(path=dest)
    with src.open("rb") as fin:
        raw_size = np.dtype(_RAW_SCALES[dtype][0]).itemsize
        fin.seek(offset * 2 * raw_size)
        # mkstemp: concurrent converters of the same capture (FastMCP runs
        # tools on worker threads) must never share a tmp path - the loser
        # would publish interleaved garbage into the immortal cache
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".cf32.tmp")
        try:
            with os.fdopen(fd, "wb") as fout:
                _convert_slice(fin, fout, dtype, samples)
            os.replace(tmp_name, dest)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    return SourceSlice(path=dest)


def _convert_slice(fin: BinaryIO, fout: BinaryIO, dtype: str, samples: int) -> None:
    raw_dtype, scale, bias = _RAW_SCALES[dtype]
    remaining = samples * 2 if samples > 0 else None
    while True:
        check_deadline()
        want = (
            _CHUNK_ITEMS * 2 if remaining is None else min(_CHUNK_ITEMS * 2, remaining)
        )
        if want == 0:
            return
        block: npt.NDArray[np.int16] | npt.NDArray[np.int8] | npt.NDArray[np.uint8] = (
            np.fromfile(fin, dtype=raw_dtype, count=want)
        )
        if block.size == 0:
            return
        if remaining is not None:
            remaining -= int(block.size)
        pairs = block[: block.size - (block.size % 2)].astype(np.float32)
        pairs = (pairs - bias) / scale
        (pairs[0::2] + 1j * pairs[1::2]).astype(np.complex64).tofile(fout)


_S = TypeVar("_S", bound=np.generic)


def _sample_stream(
    path: Path, dtype: np.dtype[_S]
) -> tuple[npt.NDArray[_S], int, bool]:
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


def _hist(x: npt.NDArray[np.float64], bins: int) -> dict[str, object]:
    lo, hi = (float(v) for v in np.percentile(x, [0.5, 99.5]))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(np.clip(x, lo, hi), bins=edges)
    # bin i's center is start + i*step: the axis is derivable, never shipped
    return {
        "start": round(float((edges[0] + edges[1]) / 2.0), 6),
        "step": round(float(edges[1] - edges[0]), 6),
        "counts": [int(n) for n in counts],
    }


_N = TypeVar("_N", np.float64, np.complex128)


def _nearest_labels(
    values: npt.NDArray[_N], centers: npt.NDArray[_N]
) -> npt.NDArray[np.intp]:
    return np.abs(values[:, None] - centers[None, :]).argmin(1)


def _farthest_point_seeds(
    points: npt.NDArray[np.complex128], k: int
) -> npt.NDArray[np.complex128]:
    # deterministic greedy k-means++ (no RNG): each seed is the point farthest
    # from every seed chosen so far. Data-driven, not a fixed angular grid, so
    # it has no unlucky-rotation basin (a grid seeded exactly between a
    # constellation's lobes can converge with two lobes merged into one seed).
    seeds = np.empty(k, dtype=np.complex128)
    seeds[0] = points[np.argmax(np.abs(points - points.mean()))]
    dist = np.abs(points - seeds[0])
    for i in range(1, k):
        seeds[i] = points[np.argmax(dist)]
        dist = np.minimum(dist, np.abs(points - seeds[i]))
    return seeds


def _kmeans_2d(
    points: npt.NDArray[np.complex128], k: int
) -> npt.NDArray[np.complex128]:
    centers = _farthest_point_seeds(points, k)
    for _ in range(_STATS_KMEANS_ITERS):
        labels = _nearest_labels(points, centers)
        moved = np.array(
            [
                points[labels == j].mean() if np.any(labels == j) else centers[j]
                for j in range(k)
            ]
        )
        if np.allclose(moved, centers):
            break
        centers = moved
    return centers


def stream_stats(
    path: Path, *, item_type: str | None, clusters: int, bins: int = 41
) -> dict[str, object]:
    _require_file(path)
    kind = _resolve_item_type(path, item_type)
    if not 1 <= bins <= _STATS_MAX_BINS:
        raise ValueError(f"bins must be 1..{_STATS_MAX_BINS}, got {bins}")
    if not 0 <= clusters <= _STATS_MAX_CLUSTERS:
        raise ValueError(f"clusters must be 0..{_STATS_MAX_CLUSTERS}, got {clusters}")
    dtype: np.dtype[Any] = np.dtype(_ITEM_DTYPES[kind])
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
    if kind == "c":
        return _constellation_stats(out, sample, clusters, bins)
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
    out["histogram"] = _hist(x, bins)
    if clusters >= 1:
        centers = kmeans_1d(x, clusters)
        labels = _nearest_labels(x, centers)
        cluster_counts = np.array(
            [int((labels == j).sum()) for j in range(centers.size)]
        )
        keep = cluster_counts > 0
        centers = centers[keep]
        cluster_counts = cluster_counts[keep]
        rounded = [round(float(v), 6) for v in centers]
        out["centers"] = rounded
        out["cluster_counts"] = [int(n) for n in cluster_counts]
        out["levels"] = list(rounded)
    return out


def _constellation_stats(
    out: dict[str, object], sample: npt.NDArray[np.complex64], clusters: int, bins: int
) -> dict[str, object]:
    z = sample.astype(np.complex128)
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError("stream has no finite items to summarize")
    out["sampled_items"] = int(z.size)
    mag = np.abs(z)
    mean_mag = float(mag.mean())
    out.update(
        mean_magnitude=round(mean_mag, 6),
        std_magnitude=round(float(mag.std()), 6),
        constant_modulus_ratio=(
            round(float(mag.std() / mean_mag), 6) if mean_mag > 0 else None
        ),
    )
    out["magnitude_histogram"] = _hist(mag, bins)
    out["phase_histogram"] = _hist(np.angle(z), bins)
    if clusters >= 1:
        centers = _kmeans_2d(z, clusters)
        labels = _nearest_labels(z, centers)
        counts = np.array([int((labels == j).sum()) for j in range(clusters)])
        # evm before the keep/order reindex below: dropping empty clusters or
        # resorting by phase never changes any point's nearest center, so this
        # is exact either way (verified) - but `labels` is only valid as an
        # index into `centers` in its current, pre-reindex form
        assigned = centers[labels]
        ref = float(np.sqrt((np.abs(assigned) ** 2).mean()))
        err = float(np.sqrt((np.abs(z - assigned) ** 2).mean()))
        out["evm"] = round(err / ref, 6) if ref > 0 else None
        keep = counts > 0
        centers, counts = centers[keep], counts[keep]
        order = np.argsort(np.angle(centers))
        centers, counts = centers[order], counts[order]
        out["clusters"] = [
            {
                "real": round(float(c.real), 6),
                "imag": round(float(c.imag), 6),
                "magnitude": round(float(abs(c)), 6),
                "phase_deg": round(float(np.degrees(np.angle(c))), 6),
                "count": int(n),
            }
            for c, n in zip(centers, counts)
        ]
    return out
