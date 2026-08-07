from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem, compile_pipeline
from marconi.engine.stages.conditioning import TranslateStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)


def _modem(f0: float) -> Modem:
    return Modem(symbol_rate=1.0, path=[TranslateStep(center_hz=f0)])


def test_translate_emits_rotator_shifting_center_to_dc() -> None:
    fs, f0 = 48_000.0, 6_000.0
    pipe = compile_modem(
        _modem(f0),
        stage_registry(),
        direction="rx",
        sample_rate=fs,
        start=IQ,
        source_io={"path": "in.cf32"},
        sink_io={"path": "out.cf32"},
    )
    assert [b.kind for b in pipe.blocks] == [
        "iq_file_source",
        "rotator_cc",
        "iq_file_sink",
    ]
    rot = next(b for b in pipe.blocks if b.kind == "rotator_cc")
    assert math.isclose(
        cast(float, rot.params["phase_inc"]), -2.0 * math.pi * f0 / fs, rel_tol=1e-9
    )


def test_translate_preserves_rate_and_amplitude() -> None:
    # a pure rotation changes neither the sample rate nor |z|
    cp = compile_pipeline(
        Modem(symbol_rate=1.0, path=[TranslateStep(center_hz=1_000.0)]),
        stage_registry(),
        direction="rx",
        sample_rate=48_000.0,
        start=Descriptor(Level.IQ, ItemType.C, Carrier.HARD, Amplitude.RMS_UNITY),
        source_io={"path": "in.cf32"},
        sink_io={"path": "out.cf32"},
    )
    assert cp.rates[-1] == 48_000.0
    assert cp.final.amplitude is Amplitude.RMS_UNITY


def test_translate_shifts_a_tone_to_dc(tmp_path: Path) -> None:
    fs, f0, n = 48_000.0, 6_000.0, 1 << 14
    tone = np.exp(2j * np.pi * f0 * np.arange(n) / fs).astype(np.complex64)
    src = tmp_path / "in.cf32"
    tone.tofile(src)
    out = tmp_path / "out.cf32"
    pipe = compile_modem(
        _modem(f0),
        stage_registry(),
        direction="rx",
        sample_rate=fs,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(out)},
    )
    ensure_worker_warm()
    r = GnuRadioBackend().run_pipeline(pipe, timeout=60.0)
    assert r.status == "ok", r
    y = np.fromfile(out, np.complex64)
    freqs = np.fft.fftfreq(y.size, d=1.0 / fs)
    peak = freqs[int(np.argmax(np.abs(np.fft.fft(y))))]
    assert abs(peak) < 50.0
