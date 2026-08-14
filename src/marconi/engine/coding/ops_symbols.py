from __future__ import annotations

from dataclasses import replace

import numpy as np

from marconi.deadline import check_deadline
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.types.enums import DcMode


def sync_symbols_rx(
    c: CodingCarrier, *, pattern: list[int], max_errors: int, pre_symbols: int
) -> CodingCarrier:
    sym = c.symbols
    if sym is None or sym.size < len(pattern):
        # the search found nothing — entry burst marks must not leak through
        # as sync positions
        return replace(c, marks=())
    pat = np.asarray(pattern, np.int64)
    want = pat != 0
    n_care = int(want.sum())
    signs = np.sign(np.asarray(sym, np.float64))
    size = signs.size - pat.size + 1
    agree = np.zeros(size, np.min_scalar_type(int(pat.size)))
    for j in np.flatnonzero(want):
        check_deadline()
        agree += signs[j : j + size] == pat[j]
    # non-overlapping, the bits-lane rule (_correlate's `reach`): an
    # alternating stream matched an alternating pattern at EVERY offset -
    # 3985 hits where the twin reports 250 - and each hit seeds a frame
    # downstream, so overlapping hits are overlapping frames
    marks: list[int] = []
    reach = 0
    for i in np.flatnonzero(agree >= n_care - max_errors):
        h = int(i)
        if h < reach:
            continue
        start = h - pre_symbols
        if start >= 0:
            marks.append(start)
        reach = h + int(pat.size)
    return replace(c, marks=tuple(marks))


def m_slice_rx(
    c: CodingCarrier, *, thresholds: list[float], levels: list[int]
) -> CodingCarrier:
    sym = c.symbols
    if sym is None:
        return c
    x = np.asarray(sym, np.float32)
    bad = int((~np.isfinite(x)).sum())
    if bad:
        # searchsorted files NaN and +inf into the TOP region, emitting
        # confident symbols from poison with nothing on the wire saying so.
        # run_rx refuses a non-finite capture, so these were manufactured by
        # the path itself - the same rule the soft-quality reading enforces.
        raise ValueError(
            f"m_slice input carries {bad} non-finite symbols of {x.size}; "
            "run_rx refuses a non-finite capture, so the path produced them "
            "- check for an agc over a zero-power span or a diverging "
            "equalizer upstream"
        )
    regions = np.searchsorted(np.asarray(thresholds, np.float32), x)
    mapped = np.asarray(levels, np.int16)[regions]
    return replace(c, symbols=mapped.astype(np.int16))


def normalize_rx(
    c: CodingCarrier,
    *,
    span_symbols: int,
    dc: DcMode = DcMode.MEDIAN,
    gain_percentile: float | None = None,
) -> CodingCarrier:
    dc = DcMode(dc)
    sym = c.symbols
    if sym is None or not c.marks:
        return c
    out = np.asarray(sym, np.float32).copy()
    for m in c.marks:
        check_deadline()
        lo, hi = int(m), min(int(m) + span_symbols, out.size)
        if hi - lo < 1:
            continue
        seg = out[lo:hi].astype(np.float64)
        if dc is DcMode.MEDIAN:
            seg = seg - np.median(seg)
        elif dc is DcMode.MEAN:
            seg = seg - seg.mean()
        if gain_percentile is not None:
            mag = np.abs(seg)
            sel = mag[mag > np.percentile(mag, gain_percentile)]
            scale = float(sel.mean()) if sel.size else 0.0
            if np.isfinite(scale) and scale > 0.0:
                seg = seg / scale
        out[lo:hi] = seg.astype(np.float32)
    return replace(c, symbols=out)
