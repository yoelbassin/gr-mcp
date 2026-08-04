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
