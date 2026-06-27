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


def read_complex(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.complex64)


def nearest_syms(z: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points)
    return np.argmin(np.abs(z[:, None] - pts[None, :]), axis=1)


def tx_sym_indices(bits: np.ndarray, k: int) -> np.ndarray:
    """Group bits into k-bit symbol indices, MSB first (matches pack_k_bits_bb)."""
    b = np.asarray(bits, dtype=int)[: (len(bits) // k) * k].reshape(-1, k)
    return b.dot(1 << np.arange(k)[::-1])


def resolved_ser(
    rx_sym: np.ndarray,
    tx_sym_idx: np.ndarray,
    points: np.ndarray,
    M: int,
    max_shift: int = 320,
    settle: int = 64,
) -> float:
    """Minimum symbol-error rate over the M carrier-phase rotations x timing
    shifts, after discarding a `settle` acquisition prefix. Coherent carrier
    recovery leaves an M-fold phase ambiguity (no preamble — acquisition is
    deferred); this resolves it the way aligned_ber resolves timing. SER 0
    <=> BER 0 for a Gray constellation."""
    best = 1.0
    roots = np.exp(2j * np.pi * np.arange(M) / M)
    rx_sym = rx_sym[settle:]
    for r in roots:
        rs = nearest_syms(rx_sym * r, points)
        for shift in range(min(max_shift, len(tx_sym_idx)) + 1):
            n = min(len(rs), len(tx_sym_idx) - shift)
            if n < len(rs) // 2:
                break
            ser = float(np.mean(rs[:n] != tx_sym_idx[shift : shift + n]))
            best = min(best, ser)
            if best == 0.0:
                return best
    return best


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
