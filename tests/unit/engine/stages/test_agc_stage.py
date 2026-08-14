from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.errors import CompileError
from marconi.engine.compile.ir import GrBlock
from marconi.engine.stages.conditioning import AgcStep, ResampleStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.bounds import MAX_AGC_HISTORY_SAMPLES
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import AgcMode, ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

_NORMALIZED = Descriptor(Level.IQ, ItemType.C, Carrier.HARD, Amplitude.RMS_UNITY)


def _emit(
    params: Mapping[str, Any], rate: float = 8.0, symbol_rate: float = 1.0
) -> list[GrBlock]:
    ctx = CompileContext(Descriptor(Level.IQ, ItemType.C), rate, symbol_rate)
    step = AgcStep(**params)
    stage_registry()["agc"].emit_rx(ctx, step)
    return ctx.build("t", rate).blocks


def test_agc_is_registered_as_conditioning() -> None:
    stage = stage_registry()["agc"]
    assert stage.family == "conditioning"
    assert (stage.from_level, stage.to_level) == (Level.IQ, Level.IQ)
    assert stage.rate_factor(AgcStep()) == 1.0


@pytest.mark.parametrize(
    "mode,statistic",
    [
        ("feedforward", Amplitude.PEAK_UNITY),
        ("feedback", Amplitude.MEAN_MAG_UNITY),
        ("power", Amplitude.RMS_UNITY),
    ],
)
def test_agc_mode_selects_the_amplitude_statistic(
    mode: str, statistic: Amplitude
) -> None:
    unknown = Descriptor(Level.IQ, ItemType.C)
    out = stage_registry()["agc"].out_descriptor(unknown, AgcStep(mode=AgcMode(mode)))
    assert out.amplitude is statistic


def test_feedforward_is_the_default_mode() -> None:
    blocks = _emit({})
    assert [b.kind for b in blocks] == ["feedforward_agc_cc"]


def test_window_symbols_converts_to_samples_via_sps() -> None:
    blocks = _emit({"window_symbols": 4.0}, rate=8.0, symbol_rate=1.0)
    assert blocks[0].params["nsamples"] == 32


def test_window_conversion_tracks_a_different_sps() -> None:
    blocks = _emit({"window_symbols": 4.0}, rate=100.0, symbol_rate=25.0)
    assert blocks[0].params["nsamples"] == 16


def test_feedback_mode_emits_the_feedback_block_with_symbol_rates() -> None:
    blocks = _emit(
        {
            "mode": "feedback",
            "attack_symbols": 2.0,
            "decay_symbols": 8.0,
            "max_gain": 100.0,
        },
        rate=8.0,
        symbol_rate=1.0,
    )
    assert [b.kind for b in blocks] == ["agc2_cc"]
    assert blocks[0].params["attack_rate"] == pytest.approx(1.0 / 16.0)
    assert blocks[0].params["decay_rate"] == pytest.approx(1.0 / 64.0)
    assert blocks[0].params["max_gain"] == pytest.approx(100.0)


def test_params_of_the_other_mode_are_rejected() -> None:
    with pytest.raises(Exception, match="does not use"):
        _emit({"mode": "feedforward", "max_gain": 10.0})


def test_unknown_param_is_rejected() -> None:
    with pytest.raises(Exception):
        _emit({"bogus": 1.0})


def _compile_path(path: list[Any], rate: float, symbol_rate: float) -> list[GrBlock]:
    return compile_modem(
        Modem(symbol_rate=symbol_rate, path=path),
        stage_registry(),
        sample_rate=rate,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    ).blocks


def _compile_agc(
    params: Mapping[str, Any], rate: float = 8.0, symbol_rate: float = 1.0
) -> list[GrBlock]:
    return _compile_path([AgcStep(**params)], rate=rate, symbol_rate=symbol_rate)


def test_power_mode_emits_the_rms_normalization_dag() -> None:
    kinds = [b.kind for b in _compile_agc({"mode": "power"})]
    assert kinds == [
        "iq_file_source",
        "complex_to_float",
        "rms_cf",
        # the divisor floor: rms_cf reads EXACTLY zero over a zero-power span
        # and 0/0 is NaN, not a large gain (see conditioning._RMS_FLOOR)
        "add_const_ff",
        "divide_ff",
        "divide_ff",
        "float_to_complex",
        "iq_file_sink",
    ]


def test_power_mode_alpha_converts_via_sps() -> None:
    def _rms_alpha(rate: float, symbol_rate: float) -> float:
        blocks = _compile_agc(
            {"mode": "power", "window_symbols": 64.0},
            rate=rate,
            symbol_rate=symbol_rate,
        )
        rms = next(b for b in blocks if b.kind == "rms_cf")
        alpha = rms.params["alpha"]
        assert isinstance(alpha, float)
        return alpha

    assert _rms_alpha(rate=8.0, symbol_rate=1.0) == pytest.approx(1.0 / 512.0)
    assert _rms_alpha(rate=100.0, symbol_rate=25.0) == pytest.approx(1.0 / 256.0)


def test_a_feedforward_window_cannot_outgrow_the_history_it_makes_the_worker_hold() -> (
    None
):
    """window_symbols x sps is a PRODUCT, and neither factor's own ceiling
    bounds it: 2.048 Msps into a 50 baud symbol_rate is sps 40960, so the
    field's own MAX_WINDOW_SYMBOLS asked feedforward_agc_cc for a
    42,949,672,960-sample history — 343 GB — from a spec validate_modem
    called valid, and the block raised a raw pybind TypeError naming a GR
    block instead of the field the caller typed.

    Two-sided on purpose: the bound is a ceiling, and one that also refuses
    the value AT it would break every working recipe below."""
    at_ceiling = _compile_agc(
        {"window_symbols": float(MAX_AGC_HISTORY_SAMPLES)}, rate=1.0, symbol_rate=1.0
    )
    history = next(b for b in at_ceiling if b.kind == "feedforward_agc_cc")
    assert history.params["nsamples"] == MAX_AGC_HISTORY_SAMPLES
    with pytest.raises(CompileError, match="window_symbols"):
        _compile_agc(
            {"window_symbols": float(MAX_AGC_HISTORY_SAMPLES + 1)},
            rate=1.0,
            symbol_rate=1.0,
        )


def test_the_agc_history_bound_reads_the_product_not_the_window_alone() -> None:
    """A window of 2 symbols is unremarkable and passes every field check; the
    sps the compiler derived is what turns it into 81,920 samples of history.
    A bound that only looked at window_symbols would call this spec valid."""
    with pytest.raises(CompileError, match="window_symbols") as caught:
        _compile_agc({"window_symbols": 2.0}, rate=2_048_000.0, symbol_rate=50.0)
    text = str(caught.value)
    assert "40960" in text, text
    assert str(MAX_AGC_HISTORY_SAMPLES) in text, text


def test_the_history_bound_reads_the_agc_s_own_boundary_not_the_source_rate() -> None:
    """The check and the emitter must derive nsamples from the SAME number, or
    the bound guards a history the compiler never builds. Behind a /16
    decimation the sps at the agc's input is 2560, not the source's 40960, and
    the same window is then 16x cheaper — a bound reading the source rate would
    refuse a spec that fits, and one reading only the emitted block would be
    checking after the fact."""
    fits: list[Any] = [
        ResampleStep(interpolation=1, decimation=16),
        AgcStep(window_symbols=16.0),
    ]
    blocks = _compile_path(fits, rate=2_048_000.0, symbol_rate=50.0)
    history = next(b for b in blocks if b.kind == "feedforward_agc_cc")
    assert history.params["nsamples"] == 40960

    too_wide: list[Any] = [
        ResampleStep(interpolation=1, decimation=16),
        AgcStep(window_symbols=32.0),
    ]
    with pytest.raises(CompileError, match="window_symbols") as caught:
        _compile_path(too_wide, rate=2_048_000.0, symbol_rate=50.0)
    assert "2560" in str(caught.value), str(caught.value)


def test_the_widest_feedforward_window_in_the_tree_still_compiles() -> None:
    """The ceiling has to clear every working recipe: the largest history any
    spec in the suite asks for is 16384 samples (4096 symbols at sps 4, the
    open-loop OOK path). Instrumented across tests/unit + tests/integration,
    that was the maximum of 61 compiled feedforward AGCs."""
    blocks = _compile_agc({"window_symbols": 4096.0}, rate=4.0, symbol_rate=1.0)
    history = next(b for b in blocks if b.kind == "feedforward_agc_cc")
    assert history.params["nsamples"] == 16384


def test_the_power_window_is_an_iir_rate_and_carries_no_history_to_bound() -> None:
    """The ceiling is the feedforward mode's alone. power reads the same field
    as a single-pole IIR coefficient 1/(symbols*sps): no window is buffered and
    no per-sample rescan happens, so binding it to the history bound would
    refuse a spec that costs nothing."""
    kinds = [
        b.kind
        for b in _compile_agc(
            {"mode": "power", "window_symbols": float(MAX_AGC_HISTORY_SAMPLES * 4)},
            rate=2_048_000.0,
            symbol_rate=50.0,
        )
    ]
    assert "rms_cf" in kinds


@pytest.mark.parametrize("bad_param", ["max_gain", "attack_symbols", "decay_symbols"])
def test_power_mode_rejects_feedback_only_params(bad_param: str) -> None:
    with pytest.raises(Exception, match="does not use"):
        _emit({"mode": "power", bad_param: 2.0})


def test_feedback_mode_rejects_window_symbols() -> None:
    with pytest.raises(Exception, match="does not use"):
        _emit({"mode": "feedback", "window_symbols": 4.0})


def test_power_mode_rejects_reference() -> None:
    with pytest.raises(Exception, match="does not use"):
        _emit({"mode": "power", "reference": 2.0})


def test_power_mode_accepts_window_symbols_only() -> None:
    kinds = [b.kind for b in _compile_agc({"mode": "power", "window_symbols": 64.0})]
    assert kinds == [
        "iq_file_source",
        "complex_to_float",
        "rms_cf",
        # the divisor floor: rms_cf reads EXACTLY zero over a zero-power span
        # and 0/0 is NaN, not a large gain (see conditioning._RMS_FLOOR)
        "add_const_ff",
        "divide_ff",
        "divide_ff",
        "float_to_complex",
        "iq_file_sink",
    ]


@pytest.mark.parametrize("mode", ["feedforward", "feedback", "power"])
def test_agc_runs_and_normalizes_a_scaled_stream(mode: str, tmp_path: Path) -> None:
    ensure_worker_warm()
    src, snk = tmp_path / "in.cf32", tmp_path / "out.cf32"
    n = 200_000
    (0.001 * np.exp(2j * np.pi * np.arange(n) / 9.0)).astype(np.complex64).tofile(src)
    params: dict[str, Any] = {"mode": mode}
    if mode == "feedback":
        params.update({"attack_symbols": 1.0, "decay_symbols": 4.0})
    pipe = compile_modem(
        Modem(symbol_rate=1.0, path=[AgcStep(**params)]),
        stage_registry(),
        sample_rate=8.0,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=180.0).status == "ok"
    tail = np.abs(np.fromfile(snk, dtype=np.complex64)[n // 2 :])
    assert 0.5 < float(tail.mean()) < 2.0, float(tail.mean())


@pytest.mark.parametrize("mode", ["feedforward", "feedback", "power"])
def test_agc_never_manufactures_non_finite_samples(mode: str, tmp_path: Path) -> None:
    """A dead lead is an ordinary capture shape — a disconnected antenna, a
    zero-padded slice, a squelched gap — and rms_cf's IIR estimate over an
    exactly-zero span is EXACTLY zero, so the power mode's divide was 0/0.

    Measured before the divisor floor: a capture whose first half was zeros
    came back 50% non-finite under status "ok"; survey on that stream then told
    the operator their own capture was corrupt, and stream_stats summarized the
    surviving half without saying so. The power mode is not an exotic choice —
    it is the only one that delivers rms_unity, so it is what the compiler
    tells the agent to insert for qam_demod and squelch."""
    ensure_worker_warm()
    src, snk = tmp_path / "in.cf32", tmp_path / "out.cf32"
    n = 40_000
    x = np.zeros(n, np.complex64)
    x[n // 2 :] = (0.01 * np.exp(2j * np.pi * np.arange(n // 2) / 9.0)).astype(
        np.complex64
    )
    x.tofile(src)
    params: dict[str, Any] = {"mode": mode}
    if mode == "feedback":
        params.update({"attack_symbols": 1.0, "decay_symbols": 4.0})
    pipe = compile_modem(
        Modem(symbol_rate=1.0, path=[AgcStep(**params)]),
        stage_registry(),
        sample_rate=8.0,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=180.0).status == "ok"
    out = np.fromfile(snk, dtype=np.complex64)
    bad = int(np.count_nonzero(~np.isfinite(out)))
    assert bad == 0, f"{mode}: {bad} of {out.size} samples are NaN/inf"
    # the dead span stays dead rather than becoming a full-scale gain
    assert float(np.abs(out[: n // 4]).max()) < 1.0
