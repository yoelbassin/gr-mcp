from __future__ import annotations

import numpy as np

from marconi.engine.backends.gnuradio.embedded.oerder_meyr import (
    estimate_tau,
    resample_symbols,
)


def _rrc(sps: int, span: int = 11, beta: float = 0.35) -> np.ndarray:
    t = (np.arange(span * sps + 1) - span * sps / 2) / sps
    with np.errstate(all="ignore"):
        num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(
            np.pi * t * (1 + beta)
        )
        h = num / (np.pi * t * (1 - (4 * beta * t) ** 2))
    h[np.abs(t) < 1e-8] = 1 - beta + 4 * beta / np.pi
    return (h / np.sqrt(np.sum(h**2))).astype(float)


def _shaped_qpsk(nsym: int, sps: int, tau: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    syms = np.exp(1j * (np.pi / 2) * rng.integers(0, 4, nsym))
    up = np.zeros(nsym * sps, complex)
    up[::sps] = syms
    x = np.convolve(up, _rrc(sps), "same")
    n = np.arange(x.size)
    return (
        np.interp(n - tau * sps, n, x.real) + 1j * np.interp(n - tau * sps, n, x.imag)
    ).astype(np.complex64)


def test_estimate_tau_recovers_known_offset() -> None:
    for tau in (-0.3, 0.0, 0.2, 0.45):
        x = _shaped_qpsk(400, 8, tau)
        est = estimate_tau(x, 8)
        # wrap-aware error in [-0.5, 0.5)
        err = (est - tau + 0.5) % 1.0 - 0.5
        assert abs(err) < 0.05, (tau, est, err)


def test_resample_hits_the_symbols() -> None:
    x = _shaped_qpsk(200, 8, 0.0, seed=3)
    tau = estimate_tau(x, 8)
    sym = resample_symbols(x, 8, tau)
    assert sym.size == x.size // 8
    r = abs(
        np.mean(np.exp(1j * 4 * np.angle(sym[np.abs(sym) > np.median(np.abs(sym))])))
    )
    assert r > 0.9, r  # clean 4-point QPSK constellation
