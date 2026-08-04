"""msk_demod: coherent MSK (h=0.5 CPFSK) to one soft float per symbol.
Scheduler-free via FAKE_GR/drive — adversarial chunking and zero-input
wakeups; the real-scheduler path is covered by test_msk_roundtrip."""

from __future__ import annotations

import numpy as np
import pytest
from helpers._dsp import AlignmentNotFound, aligned_ber_best
from helpers._fakegr import FAKE_GR, drive

from marconi.engine.backends.gnuradio.embedded.msk import (
    _polyphase_taps,
    make_msk_demod,
)

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
    # unchanged either way. Both polarities are tried: the residual 180°
    # ambiguity makes the inverted rail a legitimate hypothesis, not a control.
    h = _hard(soft)
    nrzi = (h[1:] ^ h[:-1]).astype(np.uint8)
    return aligned_ber_best([nrzi, 1 - nrzi], bits)


@pytest.mark.parametrize("sps", [10, _SPS])  # scalar medium at 10, vectorized at 20
def test_clean_msk_ber0_across_chunk_sizes(sps: int) -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 4096).astype(np.uint8)
    sig = _msk_iq(bits, sps)
    for chunk in (997, 8191, 1 << 16):
        out = drive(
            make_msk_demod(FAKE_GR, sps=float(sps)),
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


def test_no_per_bit_numpy_dispatch_below_crossover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Below _VECTOR_SPS_MIN the decision-directed loop runs on Python scalars:
    # per-bit numpy ufunc dispatch on ~sps-sample arrays measured slower there
    # (165 kbit/s vs 310 for the scalar rotor at sps=10). A wall-clock gate
    # cannot hold a 2x margin under xdist load and P/E-core variance, so gate
    # the mechanism: numpy invocations must scale with work calls, not bits.
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


def test_vectorized_medium_engages_above_crossover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # At and above _VECTOR_SPS_MIN the per-sample Python rotor is the measured
    # perf killer (scalar decays linearly with sps: 73 kbit/s at sps=64), so
    # the block must switch to per-bit vectorized ops: one np.dot per bit is
    # the mechanism's fingerprint. BER-0 pins that the medium swap is lossless.
    calls = {"n": 0}
    real_dot = np.dot

    def counted(*a, **k):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real_dot(*a, **k)

    monkeypatch.setattr(np, "dot", counted)
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, 2048).astype(np.uint8)
    sps = 16
    out = drive(
        make_msk_demod(FAKE_GR, sps=float(sps)),
        _msk_iq(bits, sps),
        chunk=8192,
        out_dtype=np.float32,
    )
    assert calls["n"] >= bits.size - 64, f"{calls['n']} dot calls for {bits.size} bits"
    assert _best_ber(out, bits) == 0.0


def test_mf_oversample_param_threads_into_taps_and_round_trips() -> None:
    # the param controls the polyphase branch count (not ignored), end to end
    assert _polyphase_taps(20.0, 41, 8).shape[0] == 9
    assert _polyphase_taps(20.0, 41, 12).shape[0] == 13
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 4096).astype(np.uint8)
    out = drive(
        make_msk_demod(FAKE_GR, sps=float(_SPS), mf_oversample=8),
        _msk_iq(bits, _SPS),
        chunk=8191,
        out_dtype=np.float32,
    )
    assert _best_ber(out, bits) == 0.0


def test_loop_pole_controls_carrier_tracking() -> None:
    # loop_pole=1.0 freezes the leaky integrator (fc = fc, never updates): the
    # carrier loop is open, so the small CFO the default pole tracks out is no
    # longer corrected. Proves the pole is a live knob, not the module constant.
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 4096).astype(np.uint8)
    sig = _msk_iq(bits, _SPS, cfo_cycles=_CFO_PINNED)
    tracked = drive(
        make_msk_demod(FAKE_GR, sps=float(_SPS)), sig, chunk=4096, out_dtype=np.float32
    )
    frozen = drive(
        make_msk_demod(FAKE_GR, sps=float(_SPS), loop_pole=1.0),
        sig,
        chunk=4096,
        out_dtype=np.float32,
    )
    assert _best_ber(tracked, bits) == 0.0
    try:
        frozen_ber = _best_ber(frozen, bits)
    except AlignmentNotFound:
        frozen_ber = 0.5  # so far from aligned it reads as random
    assert frozen_ber > 0.0


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
