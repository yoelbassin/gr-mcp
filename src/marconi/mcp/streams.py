from __future__ import annotations

from pathlib import Path

import numpy as np

_MAX_INLINE_BITS = 1_048_576
_MAX_PAGE_ITEMS = 65536
_CHUNK_ITEMS = 1 << 20

_SUFFIX_TYPES = {".u8": "b", ".i16": "s", ".f32": "f"}
_ITEM_DTYPES: dict[str, type] = {"b": np.uint8, "s": np.int16, "f": np.float32}
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
    else:
        page["llrs"] = [round(float(v), 4) for v in items]
    return page


def ensure_cf32(src: Path, dtype: str, dest_dir: Path) -> Path:
    if dtype == "cf32":
        return src
    if dtype not in _RAW_SCALES:
        raise ValueError(
            f"capture_dtype must be one of {sorted(('cf32', *_RAW_SCALES))}"
        )
    raw_dtype, scale, bias = _RAW_SCALES[dtype]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (src.stem + ".cf32")
    with src.open("rb") as fin, dest.open("wb") as fout:
        while True:
            block: np.ndarray = np.fromfile(
                fin, dtype=raw_dtype, count=_CHUNK_ITEMS * 2
            )
            if block.size == 0:
                break
            pairs = block[: block.size - (block.size % 2)].astype(np.float32)
            pairs = (pairs - bias) / scale
            (pairs[0::2] + 1j * pairs[1::2]).astype(np.complex64).tofile(fout)
    return dest
