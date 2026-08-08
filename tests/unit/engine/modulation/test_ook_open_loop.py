"""Chain-level coverage for ook_envelope's open-loop (loop_bw=0) wiring: the
compiled RX chain must route complex_to_mag through burst_sampler instead of
the closed-loop symbol_sync_ff, so each burst's own timing phase is recovered
instead of one phase frozen for the whole capture. Open-loop is AGC-free (a
sliding-window agc steps its gain mid-burst on pulsed signals and defeats
burst_sampler's fixed-threshold detection/slicing; burst_sampler normalizes
each burst internally instead) - only the closed-loop structural test still
pairs ook_envelope with agc. Reuses the compiled-chain harness
tests/unit/engine/modulation/psk/test_dqpsk_stages.py exercises (run_rx over
synthetic IQ written to a tmp_path) and the _ppm_burst-style synthetic
construction tests/unit/engine/backends/test_burst_block.py uses for the
block's own FAKE_GR-driven coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers._dsp import channel

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.ook.stages import OokEnvelopeStep
from marconi.engine.run import run_rx
from marconi.engine.stages.conditioning import AgcStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
_SPS = 2
_SAMPLE_RATE = float(_SPS)
_SYMBOL_RATE = 1.0
_PAYLOAD = "10110010" * 8
_PHASES = (0, 1, 0, 1)  # adversarial: the winning phase alternates per burst
_SEED = 23

# Lead/trail padding sized generously so the floor has settled before the
# first burst and the last burst's tail confirmation has room to complete;
# the tighter inter-burst gap is what actually exercises the
# frozen-vs-per-burst timing-phase difference. (No longer working around
# feedforward_agc_cc's lookahead under-producing near a short capture's
# edges - the open-loop path carries no agc stage at all.)
_LEAD = 20_000
_GAP = 2_500
_TRAIL = 20_000


def _ppm_burst(payload: str, sps: int, phase: int) -> np.ndarray:
    # bit 1 -> pulse-first chip pair (1, 0); bit 0 -> (0, 1); sps samples/chip
    chips: list[int] = []
    for bit in payload:
        chips += [1, 0] if bit == "1" else [0, 1]
    sig = np.repeat(np.asarray(chips, np.float32), sps)
    return np.concatenate([np.zeros(phase, np.float32), sig])


def _ambient(n: int, rng: np.random.Generator, level: float = 0.05) -> np.ndarray:
    return np.abs(rng.normal(0.0, level, n)).astype(np.float32)


def _chip_string(payload: str) -> str:
    return "".join("10" if bit == "1" else "01" for bit in payload)


def _synthetic_capture() -> np.ndarray:
    rng = np.random.default_rng(_SEED)
    stream = [_ambient(_LEAD, rng)]
    last = len(_PHASES) - 1
    for i, phase in enumerate(_PHASES):
        stream.append(_ppm_burst(_PAYLOAD, _SPS, phase))
        stream.append(_ambient(_TRAIL if i == last else _GAP, rng))
    return np.concatenate(stream)


def _modem(loop_bw: float) -> Modem:
    # closed-loop (loop_bw>0) still needs an upstream amplitude convention;
    # open-loop (loop_bw=0) is AGC-free by design (burst_sampler normalizes
    # internally), so the path carries no agc stage at all
    path: list[Step] = []
    if loop_bw > 0.0:
        path.append(AgcStep(window_symbols=4096.0))
    path += [OokEnvelopeStep(loop_bw=loop_bw), SliceStep()]
    return Modem(symbol_rate=_SYMBOL_RATE, path=path)


def _run_open_loop(workdir: Path, capture: Path) -> np.ndarray:
    r = run_rx(
        _modem(0.0),
        stage_registry(),
        sample_rate=_SAMPLE_RATE,
        start=IQ,
        workdir=workdir,
        source_io={"path": str(capture)},
    )
    assert r.status == "ok", r
    assert r.bitstream is not None
    return read_bits(r.bitstream.path)


def test_open_loop_recovers_each_burst_exactly_once(tmp_path: Path) -> None:
    ensure_worker_warm()
    env = _synthetic_capture()
    clean = tmp_path / "clean.cf32"
    env.astype(np.complex64).tofile(clean)
    # sto=0.5: a half-sample fractional timing offset, the point of maximum
    # interpolation ambiguity for a step edge. Below this a frozen interpolant
    # phase decodes an idealized rectangular chip pulse fine regardless of
    # which phase it freezes at (measured), so this is what actually puts a
    # single frozen phase at risk across bursts, matching burst_sampler's own
    # "initialization luck" real-capture failure mode (see embedded/burst.py).
    capture = channel(clean, tmp_path / "imp.cf32", sto=0.5, sample_rate=_SAMPLE_RATE)

    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    rx_a = _run_open_loop(run_a, capture)
    rx_b = _run_open_loop(run_b, capture)

    want = _chip_string(_PAYLOAD)
    found = "".join(str(bit) for bit in rx_a)
    assert found.count(want) == len(_PHASES), found.count(want)
    assert np.array_equal(rx_a, rx_b)


def test_closed_loop_chain_is_unchanged() -> None:
    pipe = compile_modem(
        _modem(0.045),
        stage_registry(),
        direction="rx",
        sample_rate=_SAMPLE_RATE,
        start=IQ,
        source_io={"path": "in.iq"},
        sink_io={"path": "out.bits"},
    )
    kinds = [b.kind for b in pipe.blocks]
    assert "symbol_sync_ff" in kinds
    assert "burst_sampler" not in kinds
    assert "feedforward_agc_cc" in kinds  # closed-loop still pairs with agc


def test_open_loop_chain_routes_through_burst_sampler() -> None:
    pipe = compile_modem(
        _modem(0.0),
        stage_registry(),
        direction="rx",
        sample_rate=_SAMPLE_RATE,
        start=IQ,
        source_io={"path": "in.iq"},
        sink_io={"path": "out.bits"},
    )
    kinds = [b.kind for b in pipe.blocks]
    assert "feedforward_agc_cc" not in kinds  # AGC-free: open-loop normalizes itself
    start = kinds.index("complex_to_mag")
    end = kinds.index("binary_slicer")
    assert kinds[start:end] == [
        "complex_to_mag",
        "burst_sampler",
        "multiply_const_ff",
        "add_const_ff",
    ]
    mul = next(b for b in pipe.blocks if b.kind == "multiply_const_ff")
    add = next(b for b in pipe.blocks if b.kind == "add_const_ff")
    assert mul.params["value"] == 2.0
    assert add.params["value"] == -1.0
