import numpy as np
from scipy import signal as sp_signal

from marconi.models import CaptureRef, DetectedSignal, PSDResult, SignalPeak


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


def find_signals(
    capture: CaptureRef,
    threshold_db: float = 6.0,
    min_bandwidth: float = 500.0,
    nperseg: int = 4096,
) -> list[DetectedSignal]:
    """Segment the PSD into contiguous regions above the noise floor."""
    freqs, p_db = _welch(capture, nperseg)
    noise_floor = float(np.median(p_db))
    bin_bw = float(freqs[1] - freqs[0])

    above = np.where(p_db > noise_floor + threshold_db)[0]
    if len(above) == 0:
        return []

    splits = np.where(np.diff(above) > 1)[0]
    groups = np.split(above, splits + 1)

    signals = []
    for g in groups:
        bandwidth = len(g) * bin_bw
        if bandwidth < min_bandwidth:
            continue
        p_lin = 10 ** (p_db[g] / 10)
        center = float(np.sum(freqs[g] * p_lin) / np.sum(p_lin))
        peak_db = float(p_db[g].max())
        signals.append(
            DetectedSignal(
                center_freq=center,
                bandwidth=float(bandwidth),
                peak_power_db=peak_db,
                snr_db=peak_db - noise_floor,
            )
        )
    signals.sort(key=lambda s: s.peak_power_db, reverse=True)
    return signals
