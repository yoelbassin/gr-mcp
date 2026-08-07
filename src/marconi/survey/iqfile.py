from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from marconi.errors import register_error

_SURVEY_SAMPLE_ITEMS = 1 << 20
_SURVEY_SCAN_BLOCK = 1 << 16
_SURVEY_MIN_ITEMS = 1 << 13
_ITEMSIZE = np.dtype(np.complex64).itemsize


class CaptureTooShort(Exception):
    pass


register_error(CaptureTooShort, "invalid_argument")


def slice_len(path: Path, offset: int, length: int) -> int:
    if offset < 0 or length < 0:
        raise ValueError(f"offset and length must be >= 0, got {offset=} {length=}")
    total = path.stat().st_size // _ITEMSIZE
    if offset >= total:
        return 0
    avail = total - offset
    return avail if length == 0 else min(length, avail)


def _most_active_start(path: Path, offset: int, span: int, budget: int) -> int:
    """First-sample index of the highest-power contiguous ``budget``-sample
    window in [offset, offset+span). A coarse per-block power scan (one block in
    memory at a time) lets an over-budget read land on signal instead of a
    leading idle gap, without stitching non-contiguous chunks into a signal that
    was never contiguous."""
    win = max(budget // _SURVEY_SCAN_BLOCK, 1)
    nblocks = -(-span // _SURVEY_SCAN_BLOCK)
    if nblocks <= win:
        return offset
    powers = np.zeros(nblocks, dtype=np.float64)
    with path.open("rb") as f:
        f.seek(offset * _ITEMSIZE)
        for i in range(nblocks):
            count = min(_SURVEY_SCAN_BLOCK, span - i * _SURVEY_SCAN_BLOCK)
            block = np.fromfile(f, dtype=np.complex64, count=count)
            if block.size:
                powers[i] = float(np.mean(np.abs(block) ** 2))
    csum = np.concatenate(([0.0], np.cumsum(powers)))
    best = int(np.argmax(csum[win:] - csum[:-win]))
    return min(offset + best * _SURVEY_SCAN_BLOCK, offset + span - budget)


def sample_iq(
    path: Path, offset: int = 0, length: int = 0, budget: int = _SURVEY_SAMPLE_ITEMS
) -> tuple[np.ndarray, int, int]:
    span = slice_len(path, offset, length)
    if span < _SURVEY_MIN_ITEMS:
        raise CaptureTooShort(
            f"{path.name}: slice of {span} complex samples is below the survey "
            f"floor of {_SURVEY_MIN_ITEMS}; widen capture_samples or the slice."
        )
    start = offset if span <= budget else _most_active_start(path, offset, span, budget)
    with path.open("rb") as f:
        f.seek(start * _ITEMSIZE)
        window = np.fromfile(f, dtype=np.complex64, count=min(span, budget))
    return window, window.size, span


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
