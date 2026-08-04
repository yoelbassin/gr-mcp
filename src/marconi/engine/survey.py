from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy.signal import welch

_SURVEY_NPERSEG = 4096
_MAX_INLINE = 512


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
    lo = float(f[int(np.searchsorted(cum, 0.005))])
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
