from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from helpers._fakegr import FAKE_GR, drive

from marconi.engine.backends.gnuradio.embedded.oerder_meyr import (
    estimate_tau,
    make_oerder_meyr,
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


def _finality_probe(total: int) -> SimpleNamespace:
    return SimpleNamespace(expected_items=total, exhausted=lambda: True)


def _drive(x: np.ndarray, sps: float, chunk: int, window: int = 64) -> np.ndarray:
    blk = make_oerder_meyr(FAKE_GR, sps=sps, window=window)
    blk.eof_probe = _finality_probe(x.size)
    return drive(blk, x.astype(np.complex64), chunk=chunk, out_dtype=np.complex64)


def test_block_decodes_a_burst_at_a_fractional_offset() -> None:
    x = _shaped_qpsk(600, 8, tau=0.3, seed=5)
    out = _drive(x, 8, chunk=x.size)
    assert abs(out.size - x.size // 8) <= 1
    z = out[np.abs(out) > np.median(np.abs(out))]
    r = abs(np.mean(np.exp(1j * 4 * np.angle(z))))
    assert r > 0.9, r


def test_output_is_chunk_independent() -> None:
    x = _shaped_qpsk(600, 8, tau=0.2, seed=6)
    whole = _drive(x, 8, chunk=x.size)
    pieced = _drive(x, 8, chunk=137)  # odd chunk, crosses windows
    n = min(whole.size, pieced.size)
    assert np.allclose(whole[:n], pieced[:n], atol=1e-4), "chunking changed output"


def test_no_flush_without_a_finality_probe() -> None:
    x = _shaped_qpsk(600, 8, tau=0.0, seed=7)
    blk = make_oerder_meyr(FAKE_GR, sps=8, window=64)  # no eof_probe set
    out = drive(blk, x.astype(np.complex64), chunk=x.size, out_dtype=np.complex64)
    # a full window is withheld at EOF when finality is unknown (sim-pad rule)
    assert out.size <= x.size // 8
