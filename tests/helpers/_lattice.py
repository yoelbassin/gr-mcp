"""Synthetic scattered-pilot lattice with known ground truth. Pilots sit on
carriers (k - kmin) % 8 == 2*fs (period n_frame_syms in time), so odd-offset
carriers are never pilots — corner-EVM checks use those."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from marconi.engine.backends.gnuradio.embedded.pilot_lattice import PilotLattice

FFT_LEN = 64
CP_LEN = 16
SYM_LEN = 80
NS = 4
KMIN = -24
N_CARRIERS = 48
RATE = 8000.0
FP_CARRIERS = [-15, 5, 17]
FP_VALUES = [1 + 0j, 0 + 1j, -1 + 0j]
_QPSK = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

_FLAT_KEYS = (
    "pilot_lens",
    "pilot_carriers",
    "pilot_i",
    "pilot_q",
    "fp_carriers",
    "fp_i",
    "fp_q",
)


@dataclass(frozen=True)
class Lattice:
    """One carrier plan and the ground truth generated against it.

    The geometry is a parameter rather than a module constant because the
    spans that break the equalizer are the edge ones — the widest the FFT
    admits, and the one bin past it. A helper pinned to a single interior
    geometry is how a structurally impossible span came to be blessed by a
    test that validated it and never ran it.
    """

    fft_len: int = FFT_LEN
    cp_len: int = CP_LEN
    kmin: int = KMIN
    n_carriers: int = N_CARRIERS
    n_frame_syms: int = NS
    dc_search: int = 2
    warmup_syms: int = 12

    @property
    def active(self) -> list[int]:
        return [k for k in range(self.kmin, self.kmin + self.n_carriers + 1) if k != 0]

    @property
    def emit(self) -> npt.NDArray[np.int64]:
        return np.array(self.active, dtype=np.int64)

    def pilot_tables(self) -> tuple[list[list[int]], list[list[complex]]]:
        rng = np.random.default_rng(7)
        carriers = [
            [k for k in self.active if (k - self.kmin) % 8 == 2 * fs]
            for fs in range(self.n_frame_syms)
        ]
        values = [
            [complex(np.exp(2j * np.pi * rng.random())) for _ in ks] for ks in carriers
        ]
        return carriers, values

    def sync_params(self) -> dict[str, Any]:
        carriers, values = self.pilot_tables()
        return {
            "fft_len": self.fft_len,
            "cp_len": self.cp_len,
            "sym_len": self.fft_len + self.cp_len,
            "n_frame_syms": self.n_frame_syms,
            "n_carriers": self.n_carriers,
            "kmin": self.kmin,
            "dc_search": self.dc_search,
            "warmup_syms": self.warmup_syms,
            "pilot_lens": [len(ks) for ks in carriers],
            "pilot_carriers": [int(k) for ks in carriers for k in ks],
            "pilot_i": [float(v.real) for vs in values for v in vs],
            "pilot_q": [float(v.imag) for vs in values for v in vs],
            "fp_carriers": [int(k) for k in FP_CARRIERS],
            "fp_i": [float(v.real) for v in FP_VALUES],
            "fp_q": [float(v.imag) for v in FP_VALUES],
        }

    def eq_params(self) -> dict[str, Any]:
        p = self.sync_params()
        for k in ("cp_len", "sym_len"):
            del p[k]
        flat = {k: p.pop(k) for k in _FLAT_KEYS}
        p["lattice"] = PilotLattice.from_flat(**flat)
        return p

    def data_mask(self) -> np.ndarray:
        carriers, _ = self.pilot_tables()
        emit = self.emit
        mask = np.ones((self.n_frame_syms, len(emit)), dtype=bool)
        for fs in range(self.n_frame_syms):
            for j, k in enumerate(emit):
                if k in carriers[fs] or k in FP_CARRIERS:
                    mask[fs, j] = False
        return mask

    def truth_grid(
        self, n_syms: int, start_fs: int, seed: int
    ) -> npt.NDArray[np.complex64]:
        rng = np.random.default_rng(seed)
        carriers, values = self.pilot_tables()
        emit = self.emit
        grid = _QPSK[rng.integers(0, 4, (n_syms, len(emit)))]
        kidx = {int(k): j for j, k in enumerate(emit)}
        for i in range(n_syms):
            fs = (start_fs + i) % self.n_frame_syms
            for k, v in zip(carriers[fs], values[fs]):
                grid[i, kidx[k]] = v
            for k, v in zip(FP_CARRIERS, FP_VALUES):
                grid[i, kidx[k]] = v
        out: npt.NDArray[np.complex64] = grid.astype(np.complex64)
        return out

    def clean_symbols(self, n_syms: int, seed: int = 1) -> np.ndarray:
        """Time-domain FFT windows (no CP, no impairments) for alignment."""
        grid = self.truth_grid(n_syms, 0, seed)
        spec = np.zeros((n_syms, self.fft_len), dtype=np.complex128)
        spec[:, self.emit + self.fft_len // 2] = grid
        return np.fft.ifft(np.fft.ifftshift(spec, axes=1), axis=1)

    def _channel(self, n_syms: int) -> np.ndarray:
        ks = self.emit.astype(np.float64)
        t = np.arange(n_syms)[:, None]
        return 1.0 + 0.3 * np.exp(
            2j * np.pi * (3.0 * ks[None, :] / self.fft_len) + 0.02j * t
        )

    def make_spectra(
        self,
        n_syms: int,
        *,
        start_fs: int = 0,
        theta: float = 0.0,
        snr_db: float = 30.0,
        seed: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """(truth_grid, fftshifted spectra) as the equalizer sees them
        post-acquisition+FFT: channel, common rotation exp(j*theta*m), AWGN."""
        grid = self.truth_grid(n_syms, start_fs, seed)
        rng = np.random.default_rng(seed + 1)
        spec = np.zeros((n_syms, self.fft_len), dtype=np.complex128)
        spec[:, self.emit + self.fft_len // 2] = grid * self._channel(n_syms)
        spec *= np.exp(1j * theta * np.arange(n_syms))[:, None]
        spec += (
            10 ** (-snr_db / 20)
            * (rng.standard_normal(spec.shape) + 1j * rng.standard_normal(spec.shape))
            / np.sqrt(2)
        )
        return grid, spec.astype(np.complex64)


DEFAULT = Lattice()
ACTIVE = DEFAULT.active
EMIT = DEFAULT.emit


def pilot_tables() -> tuple[list[list[int]], list[list[complex]]]:
    return DEFAULT.pilot_tables()


def data_mask() -> np.ndarray:
    return DEFAULT.data_mask()


def sync_params() -> dict[str, Any]:
    return DEFAULT.sync_params()


def eq_params() -> dict[str, Any]:
    return DEFAULT.eq_params()


def truth_grid(n_syms: int, start_fs: int, seed: int) -> npt.NDArray[np.complex64]:
    return DEFAULT.truth_grid(n_syms, start_fs, seed)


def clean_symbols(n_syms: int, seed: int = 1) -> np.ndarray:
    return DEFAULT.clean_symbols(n_syms, seed)


def make_spectra(
    n_syms: int,
    *,
    start_fs: int = 0,
    theta: float = 0.0,
    snr_db: float = 30.0,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    return DEFAULT.make_spectra(
        n_syms, start_fs=start_fs, theta=theta, snr_db=snr_db, seed=seed
    )


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
