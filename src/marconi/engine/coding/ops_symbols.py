from __future__ import annotations

from dataclasses import replace

import numpy as np

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
        agree += signs[j : j + size] == pat[j]
    hits = np.flatnonzero(agree >= n_care - max_errors)
    marks = tuple(int(h) - pre_symbols for h in hits if int(h) - pre_symbols >= 0)
    return replace(c, marks=marks)


def m_slice_rx(
    c: CodingCarrier, *, thresholds: list[float], levels: list[int]
) -> CodingCarrier:
    sym = c.symbols
    if sym is None:
        return c
    regions = np.searchsorted(
        np.asarray(thresholds, np.float32), np.asarray(sym, np.float32)
    )
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
