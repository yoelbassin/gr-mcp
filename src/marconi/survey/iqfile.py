from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from marconi.errors import register_error

_SURVEY_SAMPLE_ITEMS = 1 << 20
_SURVEY_SAMPLE_CHUNKS = 16
_SURVEY_MIN_ITEMS = 1 << 13
_ITEMSIZE = np.dtype(np.complex64).itemsize


class CaptureTooShort(Exception):
    pass


register_error(CaptureTooShort, "invalid_argument")


def slice_len(path: Path, offset: int, length: int) -> int:
    total = path.stat().st_size // _ITEMSIZE
    if offset >= total:
        return 0
    avail = total - offset
    return avail if length == 0 else min(length, avail)


def sample_iq(
    path: Path, offset: int = 0, length: int = 0, budget: int = _SURVEY_SAMPLE_ITEMS
) -> tuple[np.ndarray, int, int]:
    span = slice_len(path, offset, length)
    if span < _SURVEY_MIN_ITEMS:
        raise CaptureTooShort(
            f"{path.name}: slice of {span} complex samples is below the survey "
            f"floor of {_SURVEY_MIN_ITEMS}; widen capture_samples or the slice."
        )
    with path.open("rb") as f:
        if span <= budget:
            f.seek(offset * _ITEMSIZE)
            whole = np.fromfile(f, dtype=np.complex64, count=span)
            return whole, whole.size, span
        per = budget // _SURVEY_SAMPLE_CHUNKS
        starts = offset + np.linspace(0, span - per, _SURVEY_SAMPLE_CHUNKS).astype(
            np.int64
        )
        parts = []
        for s in starts:
            f.seek(int(s) * _ITEMSIZE)
            parts.append(np.fromfile(f, dtype=np.complex64, count=per))
    sample = np.concatenate(parts)
    return sample, sample.size, span


def iter_iq(
    path: Path, offset: int = 0, length: int = 0, chunk: int = _SURVEY_SAMPLE_ITEMS
) -> Iterator[np.ndarray]:
    remaining = slice_len(path, offset, length)
    with path.open("rb") as f:
        f.seek(offset * _ITEMSIZE)
        while remaining > 0:
            block = np.fromfile(f, dtype=np.complex64, count=min(chunk, remaining))
            if block.size == 0:
                break
            remaining -= int(block.size)
            yield block
