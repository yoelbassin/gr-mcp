from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.stages.conditioning import AgcStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import AgcMode, ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

_NORMALIZED = Descriptor(Level.IQ, ItemType.C, Carrier.HARD, Amplitude.RMS_UNITY)


def _emit(params: Mapping[str, Any], rate: float = 8.0, symbol_rate: float = 1.0):
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


def _compile_agc(
    params: Mapping[str, Any], rate: float = 8.0, symbol_rate: float = 1.0
):
    return compile_modem(
        Modem(symbol_rate=symbol_rate, path=[AgcStep(**params)]),
        stage_registry(),
        direction="rx",
        sample_rate=rate,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    ).blocks


def test_power_mode_emits_the_rms_normalization_dag() -> None:
    kinds = [b.kind for b in _compile_agc({"mode": "power"})]
    assert kinds == [
        "iq_file_source",
        "complex_to_float",
        "rms_cf",
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
        return float(rms.params["alpha"])

    assert _rms_alpha(rate=8.0, symbol_rate=1.0) == pytest.approx(1.0 / 512.0)
    assert _rms_alpha(rate=100.0, symbol_rate=25.0) == pytest.approx(1.0 / 256.0)


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
        direction="rx",
        sample_rate=8.0,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=180.0).status == "ok"
    tail = np.abs(np.fromfile(snk, dtype=np.complex64)[n // 2 :])
    assert 0.5 < float(tail.mean()) < 2.0, float(tail.mean())
