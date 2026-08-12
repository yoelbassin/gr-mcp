from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from marconi.deadline import check_deadline
from marconi.engine.io.source import SourceSlice
from marconi.engine.types.enums import CaptureDtype, ItemType
from marconi.levelfit import kmeans_1d, percentile_span
from marconi.mcp.workspace import (
    conversion_cache_dir,
    evict_conversion_cache,
    touch_cache_entry,
)
from marconi.wire import Histogram, Payload

_MAX_INLINE_BITS = 1_048_576
# Per-type ceilings chosen from what an item COSTS as JSON, not from a round
# item count: a bit is ~2 bytes, an int64 mark ~11, a complex pair ~21. A flat
# count therefore priced pages between 82 and 184 KB (21k-47k tokens) — one
# read_stream call could take a quarter of an agent's context, and the worst
# of them was the int64 sidecar run_rx explicitly tells the agent to page.
# Every type lands under _MAX_PAGE_BYTES; test_streams pins that budget
# directly, so a future retune is measured against the cost, not the number.
_MAX_PAGE_BYTES = 48 * 1024
_CHUNK_ITEMS = 1 << 20

_STATS_SAMPLE_ITEMS = 65536
_STATS_SAMPLE_CHUNKS = 16
_STATS_MAX_CLUSTERS = 16
_STATS_MAX_BINS = 200
_STATS_KMEANS_ITERS = 100


class PageType(StrEnum):
    """Every item type read_stream/stream_stats can page: the engine's wire
    types, plus L for the int64 sidecar lists a run writes (windows/marks
    offsets), which is never an engine wire type. _PAGE_SPECS is checked
    against both this enum and ItemType at import, so a new wire type cannot
    reach the paging surface without its page budget and renderer."""

    C = "c"
    F = "f"
    B = "b"
    S = "s"
    L = "l"


@dataclass(frozen=True)
class PageSpec:
    """Everything paging needs to know about one item type. One row per type,
    so the dtype, the suffix it infers from, its two page budgets and its
    renderer cannot drift apart."""

    dtype: np.dtype[Any]
    suffix: str
    # a default call stays near a few KB of JSON regardless of how many chars
    # one rendered item costs; an explicit count still pages more
    default_items: int
    max_items: int
    render: Callable[[npt.NDArray[Any]], dict[str, object]]


class StreamPage(Payload):
    """read_stream's wire shape. Exactly one of the rendered-item fields is
    set, chosen by the page's item type."""

    item_type: str
    offset: int
    count: int
    total_items: int
    capped_at: int | None = None
    bits: str | None = None
    symbols: list[int] | None = None
    values: list[float] | list[int] | None = None
    real: list[float] | None = None
    imag: list[float] | None = None


class ConstellationCluster(Payload):
    real: float
    imag: float
    magnitude: float
    phase_deg: float
    count: int


class StreamStats(Payload):
    """stream_stats' wire shape across all item types: the common header, then
    the branch a type populates. A field absent from the response is one this
    item type has no such measure for."""

    item_type: str
    total_items: int
    sampled_items: int
    sampled: bool
    # bits
    ones_fraction: float | None = None
    # real-valued streams
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    histogram: Histogram | None = None
    centers: list[float] | None = None
    cluster_counts: list[int] | None = None
    # complex constellation streams
    mean_magnitude: float | None = None
    std_magnitude: float | None = None
    constant_modulus_ratio: float | None = None
    magnitude_histogram: Histogram | None = None
    phase_histogram: Histogram | None = None
    evm: float | None = None
    clusters: list[ConstellationCluster] | None = None


@dataclass(frozen=True)
class RawFormat:
    """How an interleaved integer capture maps onto unit-scale cf32."""

    np_dtype: type[np.signedinteger[Any] | np.unsignedinteger[Any]]
    scale: float
    bias: float


# Keyed by the enum and checked for completeness at import: CF32 is the native
# wire and needs no conversion, every other member must name how it maps.
_RAW_FORMATS: dict[CaptureDtype, RawFormat] = {
    CaptureDtype.CI16: RawFormat(np.int16, 32768.0, 0.0),
    CaptureDtype.CI8: RawFormat(np.int8, 128.0, 0.0),
    CaptureDtype.CU8: RawFormat(np.uint8, 127.5, 127.5),
}

_unconvertible = sorted(
    d.value
    for d in CaptureDtype
    if d is not CaptureDtype.CF32 and d not in _RAW_FORMATS
)
if _unconvertible:
    raise RuntimeError(f"CaptureDtype members with no RawFormat: {_unconvertible}")


def resolve_capture_dtype(dtype: str) -> CaptureDtype:
    try:
        return CaptureDtype(dtype)
    except ValueError:
        raise ValueError(
            f"capture_dtype must be one of {sorted(d.value for d in CaptureDtype)}"
        ) from None


def parse_bits(text: str) -> npt.NDArray[np.uint8]:
    if not text or len(text) > _MAX_INLINE_BITS:
        raise ValueError(
            f"bits must be 1..{_MAX_INLINE_BITS} characters of '0'/'1', "
            f"got {len(text)}"
        )
    if set(text) - {"0", "1"}:
        raise ValueError("bits must contain only '0' and '1'")
    return np.frombuffer(text.encode("ascii"), dtype=np.uint8) - ord("0")


def require_file(path: Path, what: str = "stream output") -> None:
    """The remedy depends on who owns the path. A missing run output really
    was produced by a spec and re-running is the fix; a capture or bits file
    the caller named was never produced by anything here, and telling them to
    re-run a spec sends them to edit the one thing that is not wrong."""
    if path.is_file():
        return
    if _under_workspace(path):
        hint = (
            "marconi-runs/ outputs are not auto-cleaned — re-run the spec if "
            "the run was removed"
        )
    elif path.is_dir():
        hint = "that path is a directory; name the file itself"
    else:
        hint = (
            "check the path as given — a relative path resolves against the "
            "server's working directory, not the capture's"
        )
    raise FileNotFoundError(f"{what} not found at {path}; {hint}")


def _under_workspace(path: Path) -> bool:
    try:
        return conversion_cache_dir().parent in path.resolve().parents
    except OSError:
        return False


def _render_bits(items: npt.NDArray[Any]) -> dict[str, object]:
    ascii_bits = (items.astype(np.uint8) & np.uint8(1)) + np.uint8(ord("0"))
    return {"bits": ascii_bits.tobytes().decode("ascii")}


def _render_ints(key: str) -> Callable[[npt.NDArray[Any]], dict[str, object]]:
    def render(items: npt.NDArray[Any]) -> dict[str, object]:
        return {key: [int(v) for v in items]}

    return render


def _render_complex(items: npt.NDArray[Any]) -> dict[str, object]:
    return {
        "real": [round(float(v.real), 6) for v in items],
        "imag": [round(float(v.imag), 6) for v in items],
    }


def _render_floats(items: npt.NDArray[Any]) -> dict[str, object]:
    return {"values": [round(float(v), 4) for v in items]}


_PAGE_SPECS: dict[PageType, PageSpec] = {
    PageType.B: PageSpec(ItemType.B.np_dtype, ".u8", 4096, 16384, _render_bits),
    PageType.S: PageSpec(
        ItemType.S.np_dtype, ".i16", 2048, 8192, _render_ints("symbols")
    ),
    PageType.F: PageSpec(ItemType.F.np_dtype, ".f32", 1024, 4096, _render_floats),
    PageType.C: PageSpec(ItemType.C.np_dtype, ".cf32", 256, 2048, _render_complex),
    PageType.L: PageSpec(
        np.dtype(np.int64), ".i64", 1024, 2048, _render_ints("values")
    ),
}


def _check_page_specs() -> None:
    missing = sorted(t.value for t in PageType if t not in _PAGE_SPECS)
    if missing:
        raise RuntimeError(f"PageType members with no PageSpec: {missing}")
    unpageable = sorted(t.value for t in ItemType if t.value not in set(PageType))
    if unpageable:
        raise RuntimeError(
            f"ItemType members the paging surface cannot render: {unpageable}; "
            "add a PageType member and its PageSpec"
        )
    mismatched = sorted(
        t.value for t in ItemType if _PAGE_SPECS[PageType(t.value)].suffix != t.suffix
    )
    if mismatched:
        raise RuntimeError(f"PageSpec suffix disagrees with ItemType: {mismatched}")


_check_page_specs()

_SUFFIX_TYPES: dict[str, PageType] = {s.suffix: t for t, s in _PAGE_SPECS.items()}


def resolve_page_type(path: Path, item_type: str | None) -> PageType:
    if item_type is not None:
        try:
            return PageType(item_type)
        except ValueError:
            raise ValueError(
                f"item_type must be one of {sorted(t.value for t in PageType)}"
            ) from None
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
    require_file(path)
    kind = resolve_page_type(path, item_type)
    spec = _PAGE_SPECS[kind]
    requested = spec.default_items if count is None else count
    total = path.stat().st_size // spec.dtype.itemsize
    count = max(0, min(requested, spec.max_items, total - offset))
    with path.open("rb") as f:
        f.seek(offset * spec.dtype.itemsize)
        items = np.fromfile(f, dtype=spec.dtype, count=count)
    fields: dict[str, object] = {
        "item_type": kind.value,
        "offset": offset,
        "count": int(items.size),
        "total_items": int(total),
    }
    if requested > spec.max_items:
        fields["capped_at"] = spec.max_items
    fields.update(spec.render(items))
    return StreamPage.model_validate(fields).as_payload()


def ensure_cf32(
    src: Path, dtype: str, *, offset: int = 0, samples: int = 0
) -> SourceSlice:
    """Resolve a capture to the cf32 SourceSlice a GR file source can stream.
    cf32 passes through untouched - the GR file source does the slicing while
    streaming. Raw integer formats convert only the requested slice, into a
    content-keyed LRU cache shared across runs: iterating specs against one
    capture must not write a fresh multi-GB copy per run, and the cache must
    not grow without bound as the requested slices vary."""
    kind = resolve_capture_dtype(dtype)
    if kind is CaptureDtype.CF32:
        return SourceSlice(path=src, offset=offset, length=samples)
    st = src.stat()
    # mtime_ns keying can serve stale on filesystems with coarse timestamps
    # if a capture is overwritten same-size within one tick; acceptable
    key = f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{kind}|{offset}|{samples}"
    dest = conversion_cache_dir() / (
        hashlib.sha1(key.encode()).hexdigest()[:24] + ".cf32"
    )
    if dest.is_file():
        touch_cache_entry(dest)
        return SourceSlice(path=dest)
    # make room BEFORE writing, so the budget bounds the peak and the entry
    # about to be published is never itself a candidate
    evict_conversion_cache()
    with src.open("rb") as fin:
        raw_size = np.dtype(_RAW_FORMATS[kind].np_dtype).itemsize
        fin.seek(offset * 2 * raw_size)
        # mkstemp: concurrent converters of the same capture (FastMCP runs
        # tools on worker threads) must never share a tmp path - the loser
        # would publish interleaved garbage into the cache
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".cf32.tmp")
        try:
            with os.fdopen(fd, "wb") as fout:
                _convert_slice(fin, fout, kind, samples)
            os.replace(tmp_name, dest)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    return SourceSlice(path=dest)


def _convert_slice(
    fin: BinaryIO, fout: BinaryIO, dtype: CaptureDtype, samples: int
) -> None:
    fmt = _RAW_FORMATS[dtype]
    remaining = samples * 2 if samples > 0 else None
    while True:
        check_deadline()
        want = (
            _CHUNK_ITEMS * 2 if remaining is None else min(_CHUNK_ITEMS * 2, remaining)
        )
        if want == 0:
            return
        block: npt.NDArray[np.int16] | npt.NDArray[np.int8] | npt.NDArray[np.uint8] = (
            np.fromfile(fin, dtype=fmt.np_dtype, count=want)
        )
        if block.size == 0:
            return
        if remaining is not None:
            remaining -= int(block.size)
        pairs = block[: block.size - (block.size % 2)].astype(np.float32)
        pairs = (pairs - fmt.bias) / fmt.scale
        (pairs[0::2] + 1j * pairs[1::2]).astype(np.complex64).tofile(fout)


_S = TypeVar("_S", bound=np.generic)


@dataclass(frozen=True)
class StreamSample(Generic[_S]):
    items: npt.NDArray[_S]
    total_items: int
    strided: bool


def _sample_stream(path: Path, dtype: np.dtype[_S]) -> StreamSample[_S]:
    total = path.stat().st_size // dtype.itemsize
    with path.open("rb") as f:
        if total <= _STATS_SAMPLE_ITEMS:
            return StreamSample(np.fromfile(f, dtype=dtype), total, strided=False)
        per = _STATS_SAMPLE_ITEMS // _STATS_SAMPLE_CHUNKS
        starts = np.linspace(0, total - per, _STATS_SAMPLE_CHUNKS).astype(np.int64)
        parts = []
        for s in starts:
            f.seek(int(s) * dtype.itemsize)
            parts.append(np.fromfile(f, dtype=dtype, count=per))
    return StreamSample(np.concatenate(parts), total, strided=True)


def _hist(x: npt.NDArray[np.float64], bins: int) -> Histogram:
    lo, hi = percentile_span(x)
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(np.clip(x, lo, hi), bins=edges)
    return Histogram(
        start=round(float((edges[0] + edges[1]) / 2.0), 6),
        step=round(float(edges[1] - edges[0]), 6),
        counts=[int(n) for n in counts],
    )


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
    return stream_stats_payload(
        path, item_type=item_type, clusters=clusters, bins=bins
    ).as_payload()


def stream_stats_payload(
    path: Path, *, item_type: str | None, clusters: int, bins: int = 41
) -> StreamStats:
    require_file(path)
    kind = resolve_page_type(path, item_type)
    if not 1 <= bins <= _STATS_MAX_BINS:
        raise ValueError(f"bins must be 1..{_STATS_MAX_BINS}, got {bins}")
    if not 0 <= clusters <= _STATS_MAX_CLUSTERS:
        raise ValueError(f"clusters must be 0..{_STATS_MAX_CLUSTERS}, got {clusters}")
    sampled = _sample_stream(path, _PAGE_SPECS[kind].dtype)
    sample, total = sampled.items, sampled.total_items
    if sample.size == 0:
        raise ValueError("stream has no finite items to summarize")
    # one construction buffer, validated in one step against the declared
    # wire model: extra="forbid" rejects a typo'd key and the field types are
    # checked, neither of which a hand-assembled response dict could do
    fields: dict[str, object] = {
        "item_type": kind.value,
        "total_items": int(total),
        "sampled_items": int(sample.size),
        "sampled": sampled.strided,
    }
    if kind is PageType.B:
        fields["ones_fraction"] = round(float((sample & np.uint8(1)).mean()), 6)
    elif kind is PageType.C:
        fields.update(_constellation_fields(sample, clusters, bins))
    else:
        fields.update(_real_fields(sample, clusters, bins))
    return StreamStats.model_validate(fields)


def _real_fields(
    sample: npt.NDArray[Any], clusters: int, bins: int
) -> dict[str, object]:
    x = sample.astype(np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("stream has no finite items to summarize")
    fields: dict[str, object] = {
        "sampled_items": int(x.size),
        "min": round(float(x.min()), 6),
        "max": round(float(x.max()), 6),
        "mean": round(float(x.mean()), 6),
        "std": round(float(x.std()), 6),
        "histogram": _hist(x, bins),
    }
    if clusters >= 1:
        centers = kmeans_1d(x, clusters)
        labels = _nearest_labels(x, centers)
        cluster_counts = np.array(
            [int((labels == j).sum()) for j in range(centers.size)]
        )
        keep = cluster_counts > 0
        # "centers" was also shipped verbatim as "levels": two names, one array,
        # twice the tokens. centers IS the paste-ready list.
        fields["centers"] = [round(float(v), 6) for v in centers[keep]]
        fields["cluster_counts"] = [int(n) for n in cluster_counts[keep]]
    return fields


def _constellation_fields(
    sample: npt.NDArray[Any], clusters: int, bins: int
) -> dict[str, object]:
    z = sample.astype(np.complex128)
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError("stream has no finite items to summarize")
    mag = np.abs(z)
    mean_mag = float(mag.mean())
    fields: dict[str, object] = {
        "sampled_items": int(z.size),
        "mean_magnitude": round(mean_mag, 6),
        "std_magnitude": round(float(mag.std()), 6),
        # explicitly null when undefined, never omitted: the tool docstring
        # sends the agent to read this FIRST, and a missing key reads as
        # "this stream has no such measure"
        "constant_modulus_ratio": (
            round(float(mag.std() / mean_mag), 6) if mean_mag > 0 else None
        ),
        "magnitude_histogram": _hist(mag, bins),
        "phase_histogram": _hist(np.angle(z), bins),
    }
    if clusters >= 1:
        centers = _kmeans_2d(z, clusters)
        labels = _nearest_labels(z, centers)
        counts = np.array([int((labels == j).sum()) for j in range(clusters)])
        # `labels` indexes `centers` only in its current, pre-reindex form
        assigned = centers[labels]
        ref = float(np.sqrt((np.abs(assigned) ** 2).mean()))
        err = float(np.sqrt((np.abs(z - assigned) ** 2).mean()))
        fields["evm"] = round(err / ref, 6) if ref > 0 else None
        keep = counts > 0
        centers, counts = centers[keep], counts[keep]
        order = np.argsort(np.angle(centers))
        centers, counts = centers[order], counts[order]
        fields["clusters"] = [
            ConstellationCluster(
                real=round(float(c.real), 6),
                imag=round(float(c.imag), 6),
                magnitude=round(float(abs(c)), 6),
                phase_deg=round(float(np.degrees(np.angle(c))), 6),
                count=int(n),
            )
            for c, n in zip(centers, counts)
        ]
    return fields
