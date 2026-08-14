"""equalizer: a blind CMA equalizer inverts channel ISI, reopening a closed eye.

The stage exists because every off-air gate so far had a benign channel; a real
multipath/frequency-selective channel spreads each symbol into its neighbours
(inter-symbol interference) and closes the eye. The equalizer adapts an FIR
inverse blindly -- no training sequence, so no protocol datasheet enters the
product -- exploiting only that the modulation is (near) constant modulus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.stages.conditioning import EqualizerStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_SR, _SYM = 1.0, 1.0  # the equalizer runs symbol-spaced, on a 1-sps stream

# A causal 4-tap real channel: every symbol bleeds into the next three. Real
# taps + real BPSK keep the whole path on the real axis, so CMA's only residual
# ambiguity is a global sign -- no carrier loop needed to read it out.
_CHANNEL = np.array([1.0, 0.55, 0.30, 0.15], np.complex64)


def _equalize(src: Path, snk: Path, *, num_taps: int, step_size: float) -> np.ndarray:
    pipe = compile_modem(
        Modem(
            symbol_rate=_SYM,
            path=[EqualizerStep(num_taps=num_taps, step_size=step_size)],
        ),
        stage_registry(),
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=120.0).status == "ok"
    return np.fromfile(snk, dtype=np.complex64)


def _isi_bpsk(tmp_path: Path, n: int, seed: int = 3) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n)
    syms = (2 * bits - 1).astype(np.complex64)
    rx = np.convolve(syms, _CHANNEL)[:n].astype(np.complex64)
    src = tmp_path / "in.cf32"
    rx.tofile(src)
    return src, bits


def _min_ber(rx_bits: np.ndarray, tx_bits: np.ndarray, lo: int, hi: int) -> float:
    """BER over [lo, hi), minimized over the equalizer's group delay and CMA's
    global sign ambiguity -- both are unobservable to a blind equalizer."""
    idx = np.arange(lo, hi)
    best = 1.0
    for lag in range(-30, 30):
        e = float(np.mean(rx_bits[idx] != tx_bits[idx + lag]))
        best = min(best, e, 1.0 - e)
    return best


def test_equalizer_reopens_an_isi_closed_eye(tmp_path: Path) -> None:
    ensure_worker_warm()
    n = 6000
    src, bits = _isi_bpsk(tmp_path, n)
    rx = np.fromfile(src, dtype=np.complex64)

    # ISI closes the eye: hard decisions on the raw stream are badly wrong.
    raw = (rx.real > 0).astype(int)
    assert _min_ber(raw, bits, 200, n - 40) > 0.05

    out = _equalize(src, tmp_path / "out.cf32", num_taps=21, step_size=0.006)
    assert out.size == n, "equalizer must be rate-preserving"

    eqb = (out.real > 0).astype(int)
    # Once CMA has converged, the second half decodes with zero errors.
    assert _min_ber(eqb, bits, n // 2, n - 40) == 0.0


def test_equalizer_is_rate_preserving_and_clears_amplitude() -> None:
    stage = stage_registry()["equalizer"]
    assert stage.rate_factor(EqualizerStep()) == 1.0
    normalized = Descriptor(Level.IQ, ItemType.C, amplitude=Amplitude.RMS_UNITY)
    out = stage.out_descriptor(normalized, EqualizerStep())
    assert out.level is Level.IQ
    # CMA drives the amplitude to a constant modulus, but its steady state is
    # approximate and phase-ambiguous: a consumer needing a known scale re-agc's.
    assert out.amplitude is Amplitude.UNKNOWN


def test_equalizer_builds_a_single_cma_block() -> None:
    ctx = CompileContext(Descriptor(Level.IQ, ItemType.C), _SR, _SYM)
    stage_registry()["equalizer"].emit_rx(
        ctx, EqualizerStep(num_taps=15, step_size=0.01, modulus=1.0)
    )
    blocks = ctx.build("t", _SR).blocks
    assert [b.kind for b in blocks] == ["cma_equalizer"]
    assert blocks[0].params["num_taps"] == 15


def test_equalizer_rejects_invalid_params() -> None:
    model = stage_registry()["equalizer"].step_model
    for bad in (
        {"num_taps": 0},
        {"step_size": 0.0},
        {"step_size": -1.0},
        {"modulus": 0.0},
    ):
        with pytest.raises(ValidationError):
            model.model_validate(bad)


def test_modulus_is_capped_at_the_soft_decoder_linear_range() -> None:
    # GR's calc_soft_dec is exact to |z| ~ 8 and collapses NON-MONOTONICALLY
    # beyond (measured: LLR 5.34 at |z|=10, 0.04 at |z|=100 - identical to
    # |z|=0.01), and equalizer.modulus drives |y| straight to it. 4 keeps a
    # 2x margin; hard slicing and scale-invariant stats are unaffected either
    # way, a sum-product LDPC fed the collapsed LLRs is not.
    import pytest
    from pydantic import ValidationError

    from marconi.engine.stages.conditioning import EqualizerStep

    with pytest.raises(ValidationError):
        EqualizerStep(conv="equalizer", modulus=100.0)
    assert EqualizerStep(conv="equalizer", modulus=4.0).modulus == 4.0
