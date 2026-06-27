from __future__ import annotations

from pathlib import Path

import numpy as np


def channel(
    iq_path: Path,
    out_path: Path,
    *,
    snr_db: float | None = None,
    cfo_hz: float = 0.0,
    sto: float = 0.0,
    sfo_ppm: float = 0.0,
    sample_rate: float = 1.0,
    seed: int = 0,
) -> Path:
    """Apply realistic impairments to an IQ file: fractional sample-timing
    offset (circular FFT delay), carrier frequency offset, sample-clock offset
    (resample), then power-calibrated AWGN. Order is STO -> CFO -> SFO -> AWGN."""
    x = np.fromfile(iq_path, dtype=np.complex64).astype(np.complex128)
    rng = np.random.default_rng(seed)
    if sto:
        fr = np.fft.fftfreq(len(x))
        x = np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * fr * sto))
    if cfo_hz:
        n = np.arange(len(x))
        x = x * np.exp(2j * np.pi * cfo_hz * n / sample_rate)
    if sfo_ppm:
        from scipy.signal import resample

        x = resample(x, int(round(len(x) * (1.0 + sfo_ppm * 1e-6))))
    if snr_db is not None:
        p = float(np.mean(np.abs(x) ** 2)) or 1.0
        amp = np.sqrt(p / (2 * 10 ** (snr_db / 10)))
        x = x + amp * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    x.astype(np.complex64).tofile(out_path)
    return out_path


def write_bits(path: Path, bits: np.ndarray) -> Path:
    np.asarray(bits, dtype=np.uint8).tofile(path)
    return path


def read_bits(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8)


def aligned_ber(rx: np.ndarray, tx: np.ndarray, max_shift: int = 256) -> float:
    """Minimum BER of tx against rx over integer shifts 0..max_shift, requiring
    at least half of tx to overlap. Returns 0.0 on an exact (shifted) match."""
    rx = np.asarray(rx, dtype=np.uint8)
    tx = np.asarray(tx, dtype=np.uint8)
    best = 1.0
    for shift in range(min(max_shift, len(rx)) + 1):
        n = min(len(rx) - shift, len(tx))
        if n < len(tx) // 2:
            break
        best = min(best, float(np.mean(rx[shift : shift + n] != tx[:n])))
        if best == 0.0:
            break
    return best
