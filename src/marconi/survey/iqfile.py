from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import firwin, upfirdn

from marconi.errors import register_error

_SURVEY_SAMPLE_ITEMS = 1 << 20
_SURVEY_SCAN_BLOCK = 1 << 16
# Floor low enough to characterize a single short burst (~1-2k samples): the
# spectrum and inst-freq tone readout stay meaningful, and the symbol-rate
# search self-limits via its resolution-derived floor. Below this a slice is
# too small for even a coarse PSD.
_SURVEY_MIN_ITEMS = 1 << 10
_ITEMSIZE = np.dtype(np.complex64).itemsize
_CHANNELIZE_TAPS_PER_PHASE = 8


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
) -> tuple[npt.NDArray[np.complex64], int, int]:
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


def channelize_to_file(
    src: Path,
    dst: Path,
    sample_rate: float,
    *,
    center_hz: float,
    decim: int,
    offset: int = 0,
    length: int = 0,
    bandwidth_hz: float | None = None,
) -> tuple[int, float]:
    """Stream a sub-band of ``src`` into ``dst`` as cf32: mix ``center_hz`` to DC,
    low-pass, and decimate by ``decim`` (polyphase). Bounded memory — the raw slice
    is read in blocks and only the decimated output is accumulated. Returns
    (output_samples_written, output_sample_rate).

    The mixer runs phase-continuous across blocks; the anti-alias FIR carries its
    history across blocks (overlap-save), so the result is independent of the read
    chunking. ``bandwidth_hz`` is the filter passband width (cutoff = bandwidth_hz/2,
    matching the channelize stage); default passes most of the decimated band."""
    if decim < 1:
        raise ValueError(f"decim must be >= 1, got {decim}")
    if abs(center_hz) > 0.5 * sample_rate:
        raise ValueError(
            f"center_hz {center_hz:g} lies outside the +-{0.5 * sample_rate:g} Hz "
            f"Nyquist span of the {sample_rate:g} Hz capture; the mixer wraps mod "
            f"the sample rate and would silently tune an aliased sub-band"
        )
    out_rate = sample_rate / decim
    cutoff = (bandwidth_hz / 2.0) if bandwidth_hz is not None else 0.45 * out_rate
    cutoff = min(cutoff, 0.5 * out_rate)
    if cutoff >= 0.5 * sample_rate:
        # decim=1 translate-only: firwin needs 0 < normalized < 1; the band
        # passes whole either way
        cutoff = 0.499 * sample_rate
    taps_per_phase = _CHANNELIZE_TAPS_PER_PHASE
    numtaps = taps_per_phase * decim + 1  # numtaps-1 is a whole number of phases
    h = firwin(numtaps, cutoff / (0.5 * sample_rate))
    hist = np.zeros(numtaps - 1, dtype=np.complex128)
    chunk = max((_SURVEY_SAMPLE_ITEMS // decim) * decim, decim)
    g = 0
    written = 0
    with dst.open("wb") as out:
        for block in iter_iq(src, offset, length, chunk=chunk):
            idx = g + np.arange(block.size, dtype=np.float64)
            frac = (center_hz / sample_rate) * idx
            frac -= np.round(frac)  # keep phase precise across a long capture
            mixed = block.astype(np.complex128) * np.exp(-2j * np.pi * frac)
            seg = np.concatenate([hist, mixed])
            z = upfirdn(h, seg, up=1, down=decim)
            take = block.size // decim
            z[taps_per_phase : taps_per_phase + take].astype(np.complex64).tofile(out)
            written += take
            hist = seg[-(numtaps - 1) :]
            g += int(block.size)
    return written, out_rate


def iter_iq(
    path: Path, offset: int = 0, length: int = 0, chunk: int = _SURVEY_SAMPLE_ITEMS
) -> Iterator[npt.NDArray[np.complex64]]:
    remaining = slice_len(path, offset, length)
    with path.open("rb") as f:
        f.seek(offset * _ITEMSIZE)
        while remaining > 0:
            block = np.fromfile(f, dtype=np.complex64, count=min(chunk, remaining))
            if block.size == 0:
                break
            remaining -= int(block.size)
            yield block
