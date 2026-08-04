from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, welch

from marconi.engine.io.iqfile import iter_iq

_SURVEY_NPERSEG = 4096
_MAX_INLINE = 512


class EnvelopeStats(BaseModel):
    const_envelope_ratio: float
    amplitude_kurtosis: float
    mean_amplitude: float
    std_amplitude: float


class SpectrumStats(BaseModel):
    freqs_hz: list[float]
    psd_db: list[float]
    center_offset_hz: float
    peak_offset_hz: float
    occupied_bw_hz: float
    occupied_lo_hz: float
    occupied_hi_hz: float


def _downsample(a: np.ndarray, cap: int) -> np.ndarray:
    if a.size <= cap:
        return a
    groups = -(-a.size // cap)
    trimmed = a[: (a.size // groups) * groups]
    return trimmed.reshape(-1, groups).mean(axis=1)


def _spectrum(x: np.ndarray, sample_rate: float) -> SpectrumStats:
    nperseg = int(min(_SURVEY_NPERSEG, x.size))
    f, pxx = welch(
        x, fs=sample_rate, nperseg=nperseg, return_onesided=False, detrend=False
    )
    order = np.argsort(f)
    f, pxx = f[order], pxx[order]
    total = float(pxx.sum()) or 1.0
    centroid = float((f * pxx).sum() / total)
    peak = float(f[int(np.argmax(pxx))])
    cum = np.cumsum(pxx) / total
    lo = float(f[min(int(np.searchsorted(cum, 0.005)), f.size - 1)])
    hi = float(f[min(int(np.searchsorted(cum, 0.995)), f.size - 1)])
    fd = _downsample(f, _MAX_INLINE)
    pd = 10.0 * np.log10(np.maximum(_downsample(pxx, _MAX_INLINE), 1e-20))
    return SpectrumStats(
        freqs_hz=[float(v) for v in fd],
        psd_db=[float(v) for v in pd],
        center_offset_hz=centroid,
        peak_offset_hz=peak,
        occupied_bw_hz=hi - lo,
        occupied_lo_hz=lo,
        occupied_hi_hz=hi,
    )


def _envelope(x: np.ndarray) -> EnvelopeStats:
    a = np.abs(x).astype(np.float64)
    mean = float(a.mean())
    std = float(a.std())
    kurt = float(((a - mean) ** 4).mean() / std**4 - 3.0) if std > 0 else 0.0
    return EnvelopeStats(
        const_envelope_ratio=std / mean if mean > 0 else 0.0,
        amplitude_kurtosis=kurt,
        mean_amplitude=mean,
        std_amplitude=std,
    )


_SURVEY_IFREQ_BINS = 65
_SURVEY_RATE_K = 5
_SURVEY_CLOCK_CHUNKS = 16


class InstFreqStats(BaseModel):
    hist_centers_hz: list[float]
    hist_counts: list[int]
    peaks_hz: list[float]


class SymbolRateStats(BaseModel):
    candidates_hz: list[float]
    strengths: list[float]
    search_lo_hz: float
    search_hi_hz: float


def _clock_spectrum(
    sig: np.ndarray, sample_rate: float, chunks: int
) -> tuple[np.ndarray, np.ndarray]:
    per = sig.size // chunks
    if per < 2:
        per, chunks = sig.size, 1
    seg = sig[: per * chunks].reshape(chunks, per).astype(np.float64)
    seg = seg - seg.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.rfft(seg, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(per, d=1.0 / sample_rate)
    return freqs, mag


def _peaks_in_band(
    freqs: np.ndarray, mag: np.ndarray, lo: float, hi: float, k: int
) -> tuple[np.ndarray, np.ndarray]:
    band = (freqs >= lo) & (freqs <= hi)
    fb, mb = freqs[band], mag[band]
    if mb.size == 0:
        return np.array([]), np.array([])
    idx, _ = find_peaks(mb)
    if idx.size == 0:
        return np.array([]), np.array([])
    top = idx[np.argsort(mb[idx])[::-1][:k]]
    return fb[top], mb[top]


def _symbol_rate(
    x: np.ndarray, sample_rate: float, lo: float, hi: float
) -> SymbolRateStats:
    ya = np.abs(np.diff(np.abs(x)))
    dphi = np.angle(x[1:] * np.conj(x[:-1]))
    yf = np.abs(np.diff(dphi))
    fa, ma = _clock_spectrum(ya, sample_rate, _SURVEY_CLOCK_CHUNKS)
    fg, mg = _clock_spectrum(yf, sample_rate, _SURVEY_CLOCK_CHUNKS)
    norm = max(float(ma.max(initial=0.0)), float(mg.max(initial=0.0)), 1e-20)
    ca, sa = _peaks_in_band(fa, ma, lo, hi, _SURVEY_RATE_K)
    cg, sg = _peaks_in_band(fg, mg, lo, hi, _SURVEY_RATE_K)
    cand = np.concatenate([ca, cg])
    strg = np.concatenate([sa, sg]) / norm
    tol = max((hi - lo) * 0.01, sample_rate / max(x.size, 1))
    merged_f: list[float] = []
    merged_s: list[float] = []
    for i in np.argsort(strg)[::-1]:
        f_ = float(cand[i])
        if all(abs(f_ - m) > tol for m in merged_f):
            merged_f.append(f_)
            merged_s.append(float(strg[i]))
    return SymbolRateStats(
        candidates_hz=merged_f[:_SURVEY_RATE_K],
        strengths=merged_s[:_SURVEY_RATE_K],
        search_lo_hz=lo,
        search_hi_hz=hi,
    )


def _inst_freq(x: np.ndarray, sample_rate: float) -> InstFreqStats:
    f = np.angle(x[1:] * np.conj(x[:-1])) / (2 * np.pi) * sample_rate
    lo, hi = (float(v) for v in np.percentile(f, [0.5, 99.5]))
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(f, bins=_SURVEY_IFREQ_BINS, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    prom = max(float(counts.max()) * 0.05, 1.0)
    padded = np.concatenate(([0], counts, [0]))
    idx, _ = find_peaks(padded, prominence=prom, distance=2)
    return InstFreqStats(
        hist_centers_hz=[float(v) for v in centers],
        hist_counts=[int(v) for v in counts],
        peaks_hz=[float(centers[i - 1]) for i in idx],
    )


_SURVEY_BURST_WINDOW = 64
_SURVEY_BURST_FRACTION = 0.35


class BurstStats(BaseModel):
    count: int
    duty_cycle: float
    dominant_period_samples: int | None
    segments: list[tuple[int, int]]


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = [int(i) for i in np.flatnonzero(d == 1) + 1]
    ends = [int(i) for i in np.flatnonzero(d == -1) + 1]
    if mask[0]:
        starts = [0, *starts]
    if mask[-1]:
        ends = [*ends, int(mask.size)]
    return list(zip(starts, ends))


def _bursts(sample_x: np.ndarray, path: Path, offset: int, length: int) -> BurstStats:
    thr = _SURVEY_BURST_FRACTION * float(np.percentile(np.abs(sample_x) ** 2, 90))
    segs: list[tuple[int, int]] = []
    active_total = 0
    pos = 0
    pending: list[int] | None = None
    for block in iter_iq(path, offset, length):
        p = uniform_filter1d(
            np.abs(block).astype(np.float64) ** 2, _SURVEY_BURST_WINDOW
        )
        mask = p > thr
        active_total += int(mask.sum())
        for s, e in _find_runs(mask):
            gs, ge = pos + s, pos + e
            if pending is not None and gs - pending[1] <= _SURVEY_BURST_WINDOW:
                pending[1] = ge
            else:
                if pending is not None:
                    segs.append((pending[0], pending[1] - pending[0]))
                pending = [gs, ge]
        pos += int(block.size)
    if pending is not None:
        segs.append((pending[0], pending[1] - pending[0]))
    starts = [s for s, _ in segs]
    deltas = np.diff(starts)
    dom = int(np.median(deltas)) if deltas.size else None
    return BurstStats(
        count=len(segs),
        duty_cycle=active_total / pos if pos else 0.0,
        dominant_period_samples=dom,
        segments=segs,
    )
