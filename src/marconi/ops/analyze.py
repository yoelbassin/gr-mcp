import numpy as np
from scipy import signal as sp_signal

from marconi.models import CaptureRef, PSDResult, SignalPeak


def _read_samples(capture: CaptureRef) -> np.ndarray:
    return np.fromfile(capture.path, dtype=np.complex64)


def _welch(capture: CaptureRef, nperseg: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Welch PSD in dB, freqs absolute (Hz), ascending."""
    x = _read_samples(capture)
    if len(x) < 2:
        raise ValueError(f"capture too short for analysis: {len(x)} sample(s)")
    nperseg = min(nperseg, len(x))
    freqs, p = sp_signal.welch(
        x,
        fs=capture.sample_rate,
        nperseg=nperseg,
        return_onesided=False,
        detrend=False,
    )
    freqs = np.fft.fftshift(freqs) + capture.center_freq
    p_db = 10 * np.log10(np.fft.fftshift(p) + 1e-30)
    return freqs, p_db


def psd(capture: CaptureRef, nperseg: int = 4096) -> PSDResult:
    freqs, p_db = _welch(capture, nperseg)
    noise_floor = float(np.median(p_db))
    peak_idx, _ = sp_signal.find_peaks(
        p_db, height=noise_floor + 10.0, distance=max(1, len(p_db) // 512)
    )
    peaks = [
        SignalPeak(freq=float(freqs[i]), power_db=float(p_db[i])) for i in peak_idx
    ]
    peaks.sort(key=lambda p: p.power_db, reverse=True)
    return PSDResult(
        freqs=[float(f) for f in freqs],
        psd_db=[float(v) for v in p_db],
        noise_floor_db=noise_floor,
        peaks=peaks,
    )
