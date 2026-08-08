"""Bursty OOK/PPM round-trip through the REAL tx and open-loop rx engine
paths, end to end: a payload is tx'd through ook_envelope's emit_tx at
native sps=8 (no hand-synthesized envelope, unlike the sps=1/sps=2 siblings
test_ook_native_rate.py and tests/unit/engine/modulation/test_ook_open_loop.py,
which build the waveform directly in numpy), spliced as two burst copies
inside pure-noise idle gaps, impaired with a fractional STO and AWGN, then
rx'd with the AGC-free open-loop recipe `[ook_envelope(loop_bw=0), slice]` -
no agc stage anywhere in the path, per Tasks 12-13's per-burst
normalization.

Idle gaps are pure noise (no burst content) because the failure mode under
test is burst_sampler's own idle/burst segmentation against a tracked floor
(see embedded/burst.py's rise/fall thresholds), not per-burst timing phase -
that is what the sps=2 sibling's alternating-phase bursts already cover.
The second scenario scales one burst 5x relative to the other to exercise
per-burst normalization directly: burst_sampler scales every emitted chip
against its OWN burst's 95th-percentile grid level, so a big and a small
burst inside the same capture must both decode cleanly with no agc anywhere
in the path.

channel()'s snr_db is relative to the mean power of the WHOLE array it is
handed, same as every sibling in this directory - a single global AWGN pass
over the fully assembled capture, idle padding included (see
test_ook_native_rate.py's own "measured... SNR 15-25 dB" note for the same
convention at sps=1). That mean is sample-count-weighted, so a 5x-louder
burst pulls it - and the resulting noise floor - up unless idle padding
dilutes the bursts' share of the total; too little padding measurably
starves the quiet burst's effective SNR (10/25 seeds put a single-chip error
in the amplitude-1.0 burst at this file's payload length with LEAD/GAP/TRAIL
=8_000). LEAD/GAP/TRAIL=40_000 measured clean at BER 0 across 60/60 seeds
for both scenarios and is what is committed below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers._dsp import channel, read_bits, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.ook.stages import OokEnvelopeStep
from marconi.engine.run import run_rx
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
_SPS = 8
_SAMPLE_RATE = float(_SPS)
_SYMBOL_RATE = 1.0
_PAYLOAD = "01101100" * 8  # 64 bits -> 128 chips -> 1024 samples/burst at sps=8
_SEED = 83
_SNR_DB = 15.0
_STO = 0.37  # fractional (not integer): sps=8 leaves ample interpolation
# margin above the sps=1 Nyquist-ringing ceiling test_ook_native_rate.py
# measured there (sto=0.25 already fails on every seed at sps=1) - an 8-wide
# chip has clean interior samples away from a fractional-delay edge, so no
# near-0.5 avoidance is needed at this oversampling.
_LEAD = 40_000
_GAP = 40_000
_TRAIL = 40_000  # >> burst_sampler's fall confirmation (32 chips = 256
# samples at this sps) with wide margin, so the second burst's tail is
# flushed well before the capture's final sub-1024-sample remainder that the
# block's fixed-size floor-update blocking never processes - the "withholds
# an unfinished tail at EOF" behavior embedded/burst.py documents. Also the
# dilution the module docstring measures as needed for the 5x-amplitude
# scenario's shared noise-power reference.
_AMBIENT_LEVEL = 0.01
_BIG_SCALE = 5.0  # ~5x the baseline burst amplitude


def _chip_string(payload: str) -> str:
    return "".join("10" if bit == "1" else "01" for bit in payload)


def _tx_modem() -> Modem:
    return Modem(symbol_rate=_SYMBOL_RATE, path=[OokEnvelopeStep(), SliceStep()])


def _rx_modem() -> Modem:
    path: list[Step] = [OokEnvelopeStep(loop_bw=0.0), SliceStep()]
    return Modem(symbol_rate=_SYMBOL_RATE, path=path)


def _unit_burst(tmp_path: Path) -> np.ndarray:
    # a single pulse-position chip-pair burst (payload bit 1 -> chip pair
    # (1, 0), bit 0 -> (0, 1)) generated through the real tx chain:
    # chunks_to_symbols maps each chip bit to +/-1, then ook_envelope's
    # emit_tx un-centres and upsamples it to sps=8 IQ - the chip-pair
    # encoding is this test's construction (ook_envelope itself emits one
    # chip per input bit), the same scheme the sps=1/sps=2 siblings use.
    chips = np.array([int(c) for c in _chip_string(_PAYLOAD)], dtype=np.uint8)
    bp = write_bits(tmp_path / "burst.bits", chips)
    ip = tmp_path / "burst.cf32"
    pipe = compile_modem(
        _tx_modem(),
        stage_registry(),
        direction="tx",
        sample_rate=_SAMPLE_RATE,
        start=IQ,
        source_io={"path": str(bp)},
        sink_io={"path": str(ip)},
    )
    assert GnuRadioBackend().run_pipeline(pipe).status == "ok"
    return np.fromfile(ip, dtype=np.complex64)


def _ambient(n: int, rng: np.random.Generator) -> np.ndarray:
    mag = np.abs(rng.normal(0.0, _AMBIENT_LEVEL, n)).astype(np.float32)
    return mag.astype(np.complex64)


def _capture(tmp_path: Path, scale_a: float, scale_b: float) -> Path:
    burst = _unit_burst(tmp_path)
    rng = np.random.default_rng(_SEED)
    a = (burst * np.complex64(scale_a)).astype(np.complex64)
    b = (burst * np.complex64(scale_b)).astype(np.complex64)
    env = np.concatenate(
        [_ambient(_LEAD, rng), a, _ambient(_GAP, rng), b, _ambient(_TRAIL, rng)]
    )
    clean = tmp_path / "clean.cf32"
    env.tofile(clean)
    return channel(
        clean,
        tmp_path / "imp.cf32",
        snr_db=_SNR_DB,
        sto=_STO,
        sample_rate=_SAMPLE_RATE,
        seed=_SEED,
    )


def _run_open_loop(workdir: Path, capture: Path) -> np.ndarray:
    r = run_rx(
        _rx_modem(),
        stage_registry(),
        sample_rate=_SAMPLE_RATE,
        start=IQ,
        workdir=workdir,
        source_io={"path": str(capture)},
    )
    assert r.status == "ok", r
    assert r.bitstream is not None
    return read_bits(r.bitstream.path)


def _burst_ber(found: str, want: str, start: int) -> float:
    window = found[start : start + len(want)]
    if len(window) != len(want):
        return 1.0
    mismatches = sum(1 for a, b in zip(window, want) if a != b)
    return mismatches / len(want)


def _assert_two_bursts_ber0(rx: np.ndarray) -> None:
    found = "".join(str(bit) for bit in rx)
    want = _chip_string(_PAYLOAD)
    starts: list[int] = []
    cursor = 0
    for _ in range(2):
        idx = found.find(want, cursor)
        assert idx != -1, f"only {len(starts)} burst(s) located so far: {starts}"
        starts.append(idx)
        cursor = idx + len(want)
    assert found.find(want, cursor) == -1, "found an unexpected 3rd burst copy"
    for start in starts:
        assert _burst_ber(found, want, start) == 0.0


def test_same_amplitude_bursts_decode_at_ber0(tmp_path: Path) -> None:
    ensure_worker_warm()
    capture = _capture(tmp_path, 1.0, 1.0)

    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    rx_a = _run_open_loop(run_a, capture)
    rx_b = _run_open_loop(run_b, capture)

    _assert_two_bursts_ber0(rx_a)
    assert np.array_equal(rx_a, rx_b)


def test_5x_amplitude_mismatched_bursts_decode_at_ber0(tmp_path: Path) -> None:
    ensure_worker_warm()
    capture = _capture(tmp_path, 1.0, _BIG_SCALE)

    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    rx_a = _run_open_loop(run_a, capture)
    rx_b = _run_open_loop(run_b, capture)

    _assert_two_bursts_ber0(rx_a)
    assert np.array_equal(rx_a, rx_b)
