"""msk_demod: coherent MSK (h=0.5 CPFSK) to one soft float per symbol.
Scheduler-free via FAKE_GR/drive — adversarial chunking and zero-input
wakeups; the real-scheduler path is covered by test_msk_roundtrip."""

from __future__ import annotations

import numpy as np
import pytest
from phy._dsp import aligned_ber
from phy._fakegr import FAKE_GR, drive

from marconi.phy.backends.gnuradio.embedded.msk import make_msk_demod

_SPS = 20

# cycles/sample = 2.5 Hz at 48 kHz, half the ±5 Hz Gate-B CFO bound measured on
# the real capture (task-2 report §6: ±5 Hz PASS, ±7 Hz FAIL). sps=20 pins the
# implied rate at baud·sps = 2400·20 = 48 kHz.
_CFO_PINNED = 2.5 / 48000.0


def _msk_iq(bits: np.ndarray, sps: int, cfo_cycles: float = 0.0) -> np.ndarray:
    df = np.where(bits.repeat(sps) == 1, 0.25 / sps, -0.25 / sps) + cfo_cycles
    return np.exp(2j * np.pi * np.cumsum(df)).astype(np.complex64)


def _hard(soft: np.ndarray) -> np.ndarray:
    return (soft > 0).astype(np.uint8)


def _best_ber(soft: np.ndarray, bits: np.ndarray) -> float:
    # Coherent MSK detection (matched filter + alternating I/Q-rail decision)
    # recovers the data DIFFERENTIALLY-ENCODED (NRZI): the payload rides in the
    # ±90°/bit phase STEP, not the absolute constellation point, so the demod's
    # rail decisions are the NRZI stream of the bits. `_msk_iq` transmits raw
    # (non-precoded) MSK, so the per-protocol NRZI decode (consecutive XOR) —
    # which lives in the bits layer downstream, like the AIS `differential`
    # stage, keeping the demod protocol-agnostic — is applied here before
    # scoring. Real precoded links (e.g. ACARS) decode directly; the block is
    # unchanged either way. min(ber, 1-ber) absorbs the residual 180° polarity.
    h = _hard(soft)
    nrzi = (h[1:] ^ h[:-1]).astype(np.uint8)
    return min(aligned_ber(nrzi, bits), aligned_ber(1 - nrzi, bits))


def test_clean_msk_ber0_across_chunk_sizes() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 4096).astype(np.uint8)
    sig = _msk_iq(bits, _SPS)
    for chunk in (997, 8191, 1 << 16):
        out = drive(
            make_msk_demod(FAKE_GR, sps=float(_SPS)),
            sig,
            chunk=chunk,
            out_dtype=np.float32,
        )
        assert out.size >= bits.size - 64, f"tail loss: {out.size}/{bits.size}"
        assert _best_ber(out, bits) == 0.0


def test_small_cfo_tracked() -> None:
    # tolerance bound pinned from the Task-2 report's measured CFO range
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 4096).astype(np.uint8)
    sig = _msk_iq(bits, _SPS, cfo_cycles=_CFO_PINNED)  # cycles/sample; PIN
    out = drive(
        make_msk_demod(FAKE_GR, sps=float(_SPS)), sig, chunk=4096, out_dtype=np.float32
    )
    assert out.size >= bits.size - 64
    assert _best_ber(out, bits) == 0.0


def test_eof_drains_pending_tail() -> None:
    # drive() issues zero-input wakeups after each chunk; a correct
    # forecast_drain block emits everything it ever buffered
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, 256).astype(np.uint8)
    sig = _msk_iq(bits, _SPS)
    out = drive(
        make_msk_demod(FAKE_GR, sps=float(_SPS)),
        sig,
        chunk=sig.size,
        out_dtype=np.float32,
    )
    assert out.size >= bits.size - 8


def test_no_per_bit_numpy_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # The decision-directed loop is per-bit sequential; the measured perf
    # killer is per-bit numpy ufunc dispatch on ~sps-sample arrays
    # (165 kbit/s vs 310 for the scalar rotor). A wall-clock gate cannot
    # hold a 2x margin under xdist load and P/E-core variance, so gate the
    # mechanism: numpy invocations must scale with work calls, not bits.
    calls = {"n": 0}

    def counted(fn):  # type: ignore[no-untyped-def]
        def wrapper(*a, **k):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return fn(*a, **k)

        return wrapper

    for name in ("exp", "arange", "dot", "concatenate", "asarray", "zeros"):
        monkeypatch.setattr(np, name, counted(getattr(np, name)))
    rng = np.random.default_rng(4)
    n = 40_000  # 4000 bits at sps=10
    iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    drive(make_msk_demod(FAKE_GR, sps=10.0), iq, chunk=8192, out_dtype=np.float32)
    assert calls["n"] < 400, f"{calls['n']} numpy calls for 4000 bits"


def test_burst_between_noise_keeps_lock() -> None:
    # A real ACARS capture is bursty: signal-free stretches surround each frame.
    # The block must re-lock on the burst rather than lose lock permanently on
    # noise, so the burst's bit pattern must reappear intact. Noise spans are a
    # multiple of sps, placing the burst on a bit-clock boundary; the matched
    # filter still bleeds ~2 bits of noise into each burst edge and the PLL needs
    # a few bits to re-settle, so a small guard band is trimmed and the burst
    # core is asserted as a contiguous subsequence (aligned_ber shift search).
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 512).astype(np.uint8)
    burst = _msk_iq(bits, _SPS)
    n_noise = 50 * _SPS
    noise = rng.standard_normal((2, n_noise)) + 1j * rng.standard_normal((2, n_noise))
    sig = np.concatenate([noise[0], burst, noise[1]]).astype(np.complex64)

    out = drive(
        make_msk_demod(FAKE_GR, sps=float(_SPS)), sig, chunk=4096, out_dtype=np.float32
    )
    guard = 8
    core = bits[guard:-guard]
    assert _best_ber(out, core) == 0.0
