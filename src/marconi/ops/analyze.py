import numpy as np
from scipy import signal as sp_signal

from marconi.models import (
    Burst,
    CaptureRef,
    DetectedSignal,
    PSDResult,
    SignalMeasurement,
    SignalPeak,
)


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
    """Segment the PSD into contiguous regions above the noise floor.

    Each detected signal's ``bandwidth`` is the threshold-crossing extent —
    the span of PSD bins that exceed ``noise_floor + threshold_db`` — NOT a
    -3 dB bandwidth or 99%-power OBW.

    ``snr_db`` is the signal's peak power relative to the global median noise
    floor (i.e. ``peak_power_db - median(psd_db)``).

    Wrap-around handling: a tone near ±fs/2 can straddle the cyclic edge of
    the shifted PSD array, producing one above-threshold group that touches
    index 0 and another that touches index N-1.  When that pattern is
    detected the two groups are merged into one before computing per-signal
    statistics.  The power-weighted center frequency is computed with the
    minority-side bins "unwrapped" by ±sample_rate so the reported center
    lands near the true tone frequency rather than averaging the two edges.
    """
    freqs, p_db = _welch(capture, nperseg)
    noise_floor = float(np.median(p_db))
    bin_bw = float(freqs[1] - freqs[0])

    above = np.where(p_db > noise_floor + threshold_db)[0]
    if len(above) == 0:
        return []

    splits = np.where(np.diff(above) > 1)[0]
    groups = np.split(above, splits + 1)

    # Wrap-around merge: a tone near ±fs/2 produces one group touching index 0
    # and one touching index N-1.  Detect this and merge them, unwrapping the
    # minority-side frequencies so the power-weighted centroid is correct.
    N = len(p_db)
    merged: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if len(groups) >= 2 and int(groups[0][0]) == 0 and int(groups[-1][-1]) == N - 1:
        first_g, last_g = groups[0], groups[-1]
        groups = groups[1:-1]  # remaining interior groups (may be empty)
        if p_db[last_g].max() >= p_db[first_g].max():
            # Dominant peak at the high-frequency edge: shift the low-edge bins
            # up by one sample-rate period so they are contiguous with the
            # high-edge bins in frequency.
            m_freqs = np.concatenate(
                [freqs[last_g], freqs[first_g] + capture.sample_rate]
            )
        else:
            # Dominant peak at the low-frequency edge: shift the high-edge bins
            # down by one sample-rate period.
            m_freqs = np.concatenate(
                [freqs[last_g] - capture.sample_rate, freqs[first_g]]
            )
        m_indices = np.concatenate([last_g, first_g])
        merged = (m_indices, m_freqs, p_db[m_indices])

    signals = []

    def _append_signal(
        g_indices: np.ndarray, g_freqs: np.ndarray, g_pdb: np.ndarray
    ) -> None:
        bandwidth = len(g_indices) * bin_bw
        if bandwidth < min_bandwidth:
            return
        p_lin = 10 ** (g_pdb / 10)
        center = float(np.sum(g_freqs * p_lin) / np.sum(p_lin))
        peak_db = float(g_pdb.max())
        signals.append(
            DetectedSignal(
                center_freq=center,
                bandwidth=float(bandwidth),
                peak_power_db=peak_db,
                snr_db=peak_db - noise_floor,
            )
        )

    if merged is not None:
        _append_signal(*merged)

    for g in groups:
        _append_signal(g, freqs[g], p_db[g])

    signals.sort(key=lambda s: s.peak_power_db, reverse=True)
    return signals


def detect_bursts(
    capture: CaptureRef,
    window: float = 1e-3,
    threshold_db: float = 6.0,
) -> list[Burst]:
    """Detect on/off bursts from the smoothed power envelope.

    The threshold is median + threshold_db; an always-on signal therefore
    yields no bursts (its median power IS the signal).
    """
    x = _read_samples(capture)
    if len(x) < 2:
        raise ValueError(f"capture too short for analysis: {len(x)} sample(s)")
    fs = capture.sample_rate
    win = max(1, int(window * fs))
    power = np.convolve(np.abs(x) ** 2, np.ones(win) / win, mode="same")
    p_db = 10 * np.log10(power + 1e-30)
    threshold = float(np.median(p_db)) + threshold_db

    above = np.where(p_db > threshold)[0]
    if len(above) == 0:
        return []

    splits = np.where(np.diff(above) > 1)[0]
    groups = np.split(above, splits + 1)

    bursts = []
    for g in groups:
        duration = len(g) / fs
        if duration < 2 * window:
            continue
        bursts.append(
            Burst(
                start_time=float(g[0] / fs),
                duration=float(duration),
                mean_power_db=float(np.mean(p_db[g])),
            )
        )
    return bursts


def measure(
    capture: CaptureRef,
    center_freq: float,
    search_bandwidth: float = 200e3,
    nperseg: int = 4096,
) -> SignalMeasurement:
    """Measure the signal nearest center_freq within the search window.

    occupied_bw_99 is the 99% power containment bandwidth within the
    window; snr_db is peak power relative to the global median noise floor.

    A noise-only window returns snr_db ≈ 3 dB at default parameters (max of
    many chi-squared bins vs the median floor); treat a signal as reliably
    present only above ~8 dB.
    """
    freqs, p_db = _welch(capture, nperseg)
    noise_floor = float(np.median(p_db))

    sel = (freqs >= center_freq - search_bandwidth / 2) & (
        freqs <= center_freq + search_bandwidth / 2
    )
    if not np.any(sel):
        raise ValueError("search window is outside the capture's spectrum")

    f_sel = freqs[sel]
    p_sel_db = p_db[sel]
    p_lin = 10 ** (p_sel_db / 10)
    total = float(np.sum(p_lin))

    csum = np.cumsum(p_lin) / total
    lo = float(f_sel[min(int(np.searchsorted(csum, 0.005)), len(f_sel) - 1)])
    hi = float(f_sel[min(int(np.searchsorted(csum, 0.995)), len(f_sel) - 1)])

    bin_bw = float(freqs[1] - freqs[0])
    return SignalMeasurement(
        center_freq=float(f_sel[int(np.argmax(p_sel_db))]),
        occupied_bw_99=hi - lo,
        power_db=10 * float(np.log10(total * bin_bw + 1e-30)),
        snr_db=float(p_sel_db.max()) - noise_floor,
    )
