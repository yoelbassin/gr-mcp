"""squelch: mute the gaps so clock recovery does not free-run through them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.core.descriptor import Amplitude, Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compile_context import CompileContext
from marconi.phy.compiler import CompileError, compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
# The squelch is the unit under test, so the source declares the normalized
# scale its threshold is relative to rather than an agc establishing it.
IQ_NORMALIZED = Descriptor(Level.IQ, "c", amplitude=Amplitude.RMS_UNITY)
_SR, _SYM = 8.0, 1.0


def _run(
    src: Path, snk: Path, threshold_db: float, agc: ModemStep | None = None
) -> np.ndarray:
    path = [] if agc is None else [agc]
    path.append(
        ModemStep(
            conv="squelch",
            params={"threshold_db": threshold_db, "alpha_symbols": 1.0},
        )
    )
    pipe = compile_modem(
        ModemSpec(symbol_rate=_SYM, path=path),
        stage_registry(),
        direction="rx",
        sample_rate=_SR,
        start=IQ if agc is not None else IQ_NORMALIZED,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=120.0).status == "ok"
    return np.fromfile(snk, dtype=np.complex64)


def _bursty(tmp_path: Path, n: int) -> Path:
    rng = np.random.default_rng(0)
    loud = rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    quiet = (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)) * 1e-3
    src = tmp_path / "in.cf32"
    np.concatenate([loud, quiet]).astype(np.complex64).tofile(src)
    return src


def test_squelch_mutes_the_gap_and_keeps_the_loud_part(tmp_path: Path) -> None:
    """Rate-preserving: same sample count out, quiet half zeroed."""
    ensure_worker_warm()
    n = 8192
    src = _bursty(tmp_path, n)
    out = _run(src, tmp_path / "out.cf32", threshold_db=-20.0)
    assert out.size == 2 * n, "squelch must not change the sample count"
    settle = n // 4
    tail = out[n + settle :]
    assert np.count_nonzero(tail) == 0, "quiet region was not muted"
    assert np.count_nonzero(out[settle:n]) > 0.9 * (n - settle), "loud region was muted"


def test_squelch_requires_a_known_scale() -> None:
    """An absolute dB threshold on an unnormalized stream is meaningless."""
    stage = stage_registry()["squelch"]
    assert stage.accepts_amplitude == frozenset({Amplitude.RMS_UNITY})
    with pytest.raises(CompileError, match="rms_unity"):
        compile_modem(
            ModemSpec(
                symbol_rate=_SYM,
                path=[ModemStep(conv="squelch", params={"threshold_db": -20.0})],
            ),
            stage_registry(),
            direction="rx",
            sample_rate=_SR,
            start=IQ,
            source_io={"path": "in.cf32"},
            sink_io={"path": "out.cf32"},
        )


def test_squelch_is_rate_and_amplitude_transparent() -> None:
    stage = stage_registry()["squelch"]
    assert stage.rate_factor({"threshold_db": -20.0}) == 1.0
    normalized = Descriptor(Level.IQ, "c", amplitude=Amplitude.RMS_UNITY)
    out = stage.out_descriptor(normalized, {"threshold_db": -20.0})
    assert out.amplitude is Amplitude.RMS_UNITY


def test_squelch_never_gates_the_stream() -> None:
    """gate=True would make output length signal-dependent, which no
    rate_factor can express; the stage must not expose it."""
    ctx = CompileContext(Descriptor(Level.IQ, "c"), _SR, _SYM)
    stage_registry()["squelch"].emit_rx(ctx, {"threshold_db": -20.0})
    blocks = ctx.build("t", _SR).blocks
    assert [b.kind for b in blocks] == ["pwr_squelch_cc"]
    assert blocks[0].params["gate"] is False
    with pytest.raises(ValidationError):
        stage_registry()["squelch"].params_model.model_validate(
            {"threshold_db": -20.0, "gate": True}
        )


def test_squelch_rejects_invalid_time_constants() -> None:
    model = stage_registry()["squelch"].params_model
    for bad in ({"alpha_symbols": 0.0}, {"ramp_symbols": -1.0}):
        with pytest.raises(ValidationError):
            model.model_validate({"threshold_db": -20.0, **bad})


def test_a_short_agc_window_defeats_the_squelch(tmp_path: Path) -> None:
    """Composition hazard, measured: an agc whose window is shorter than the
    gap lifts the noise floor between bursts to meet the threshold, so nothing
    is muted. The window must span the gaps."""
    ensure_worker_warm()
    n = 8192
    src = _bursty(tmp_path, n)
    short = ModemStep(conv="agc", params={"mode": "power", "window_symbols": 16.0})
    long = ModemStep(conv="agc", params={"mode": "power", "window_symbols": 4096.0})
    tail = slice(n + n // 4, None)
    muted_short = np.count_nonzero(_run(src, tmp_path / "s.cf32", -20.0, short)[tail])
    muted_long = np.count_nonzero(_run(src, tmp_path / "l.cf32", -20.0, long)[tail])
    assert muted_short > 0, "a short agc window no longer defeats the squelch"
    assert muted_long == 0, "a gap-spanning agc window should leave the gap muted"
