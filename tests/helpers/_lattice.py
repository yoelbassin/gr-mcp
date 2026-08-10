"""Synthetic scattered-pilot lattice with known ground truth. Pilots sit on
carriers (k - KMIN) % 8 == 2*fs (period NS in time), so odd-offset carriers
are never pilots — corner-EVM checks use those."""

from __future__ import annotations

from typing import Any

import numpy as np

from marconi.engine.backends.gnuradio.embedded.pilot_lattice import PilotLattice

FFT_LEN = 64
CP_LEN = 16
SYM_LEN = 80
NS = 4
KMIN = -24
N_CARRIERS = 48
RATE = 8000.0
ACTIVE = [k for k in range(KMIN, KMIN + N_CARRIERS + 1) if k != 0]
EMIT = np.array(ACTIVE)
FP_CARRIERS = [-15, 5, 17]
FP_VALUES = [1 + 0j, 0 + 1j, -1 + 0j]
_QPSK = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)


def pilot_tables() -> tuple[list[list[int]], list[list[complex]]]:
    rng = np.random.default_rng(7)
    carriers = [[k for k in ACTIVE if (k - KMIN) % 8 == 2 * fs] for fs in range(NS)]
    values = [
        [complex(np.exp(2j * np.pi * rng.random())) for _ in ks] for ks in carriers
    ]
    return carriers, values


def data_mask() -> np.ndarray:
    carriers, _ = pilot_tables()
    mask = np.ones((NS, len(EMIT)), dtype=bool)
    for fs in range(NS):
        for j, k in enumerate(EMIT):
            if k in carriers[fs] or k in FP_CARRIERS:
                mask[fs, j] = False
    return mask


def sync_params() -> dict[str, Any]:
    carriers, values = pilot_tables()
    return {
        "fft_len": FFT_LEN,
        "cp_len": CP_LEN,
        "sym_len": SYM_LEN,
        "n_frame_syms": NS,
        "n_carriers": N_CARRIERS,
        "kmin": KMIN,
        "dc_search": 2,
        "warmup_syms": 12,
        "pilot_lens": [len(ks) for ks in carriers],
        "pilot_carriers": [int(k) for ks in carriers for k in ks],
        "pilot_i": [float(v.real) for vs in values for v in vs],
        "pilot_q": [float(v.imag) for vs in values for v in vs],
        "fp_carriers": [int(k) for k in FP_CARRIERS],
        "fp_i": [float(v.real) for v in FP_VALUES],
        "fp_q": [float(v.imag) for v in FP_VALUES],
    }


def eq_params() -> dict[str, Any]:
    p = sync_params()
    for k in ("cp_len", "sym_len"):
        del p[k]
    flat = {
        k: p.pop(k)
        for k in (
            "pilot_lens",
            "pilot_carriers",
            "pilot_i",
            "pilot_q",
            "fp_carriers",
            "fp_i",
            "fp_q",
        )
    }
    p["lattice"] = PilotLattice.from_flat(**flat)
    return p


def truth_grid(n_syms: int, start_fs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    carriers, values = pilot_tables()
    grid = _QPSK[rng.integers(0, 4, (n_syms, len(EMIT)))]
    kidx = {int(k): j for j, k in enumerate(EMIT)}
    for i in range(n_syms):
        fs = (start_fs + i) % NS
        for k, v in zip(carriers[fs], values[fs]):
            grid[i, kidx[k]] = v
        for k, v in zip(FP_CARRIERS, FP_VALUES):
            grid[i, kidx[k]] = v
    return grid


def clean_symbols(n_syms: int, seed: int = 1) -> np.ndarray:
    """Time-domain FFT windows (no CP, no impairments) for alignment checks."""
    grid = truth_grid(n_syms, 0, seed)
    dc0 = FFT_LEN // 2
    spec = np.zeros((n_syms, FFT_LEN), dtype=np.complex128)
    spec[:, EMIT + dc0] = grid
    return np.fft.ifft(np.fft.ifftshift(spec, axes=1), axis=1)


def _channel(n_syms: int) -> np.ndarray:
    ks = EMIT.astype(np.float64)
    t = np.arange(n_syms)[:, None]
    return 1.0 + 0.3 * np.exp(2j * np.pi * (3.0 * ks[None, :] / FFT_LEN) + 0.02j * t)


def make_spectra(
    n_syms: int,
    *,
    start_fs: int = 0,
    theta: float = 0.0,
    snr_db: float = 30.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """(truth_grid, fftshifted spectra) as the equalizer sees them
    post-acquisition+FFT: channel, common rotation exp(j*theta*m), AWGN."""
    grid = truth_grid(n_syms, start_fs, seed)
    rng = np.random.default_rng(seed + 1)
    spec = np.zeros((n_syms, FFT_LEN), dtype=np.complex128)
    dc0 = FFT_LEN // 2
    spec[:, EMIT + dc0] = grid * _channel(n_syms)
    spec *= np.exp(1j * theta * np.arange(n_syms))[:, None]
    spec += (
        10 ** (-snr_db / 20)
        * (rng.standard_normal(spec.shape) + 1j * rng.standard_normal(spec.shape))
        / np.sqrt(2)
    )
    return grid, spec.astype(np.complex64)


def make_iq(
    n_syms: int,
    *,
    cfo_hz: float = 0.0,
    snr_db: float = 30.0,
    lead_noise: int = 0,
    sto_frac: float = 0.0,
    seed: int = 1,
) -> np.ndarray:
    """Time-domain capture: IFFT+CP per symbol, multipath, CFO, fractional
    STO, AWGN, optional leading noise. start_fs is always 0."""
    t = clean_symbols(n_syms, seed)
    sig = np.concatenate([np.hstack([s[-CP_LEN:], s]) for s in t])
    sig = np.convolve(sig, [1.0, 0.0, 0.0, 0.25j])[: sig.size]
    # snr_db is referenced to actual signal power (IFFT spreads unit cells
    # to ~0.012/sample; without this the label overstates SNR by ~18 dB)
    sig = sig / np.sqrt(np.mean(np.abs(sig) ** 2))
    n = np.arange(sig.size)
    sig = sig * np.exp(2j * np.pi * cfo_hz * n / RATE)
    if sto_frac:
        sig = np.interp(n + sto_frac, n, sig.real) + 1j * np.interp(
            n + sto_frac, n, sig.imag
        )
    rng = np.random.default_rng(seed + 2)
    sig = sig + 10 ** (-snr_db / 20) * (
        rng.standard_normal(sig.size) + 1j * rng.standard_normal(sig.size)
    ) / np.sqrt(2)
    if lead_noise:
        head = 0.05 * (
            rng.standard_normal(lead_noise) + 1j * rng.standard_normal(lead_noise)
        )
        sig = np.concatenate([head, sig])
    return sig.astype(np.complex64)
