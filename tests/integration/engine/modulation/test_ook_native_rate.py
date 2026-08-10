"""Native-rate (sps=1) open-loop OOK decode through the real engine: once
min_input_sps_for lets ook_envelope(loop_bw=0) accept a capture already at
the symbol rate, no resample stage sits ahead of it - the anti-imaging
low-pass a rate conversion would otherwise apply is exactly what smears a
single-sample-wide chip. Impairs with fractional STO and AWGN per the suite's
sim-oracle rule (an integer-only offset would hide an OSR=1 interpolation
bug). Models on test_qam_roundtrip.py's run_rx harness in this directory and
reuses the pulse-position chip-pair synthesis from
tests/unit/engine/modulation/test_ook_open_loop.py (same file the sps=2
compiled-chain coverage lives in)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers._dsp import channel

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.bitfile import read_bits
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
_SYMBOL_RATE = 1.0
_SAMPLE_RATE = _SYMBOL_RATE  # native rate: sps == 1, the seam under test
_PAYLOAD = "11010010" * 8
_PHASES = (0, 1, 0, 1)  # coarse per-burst start offsets within the capture
_SEED = 71
_SNR_DB = 20.0
_STO = 0.2  # see test_native_rate_open_loop_recovers_every_burst_at_ber0
_LEAD = 6_000
_GAP = 1_200
_TRAIL = 6_000


def _pulse_pair_burst(payload: str, phase: int) -> np.ndarray:
    # one chip pair per bit at native sps=1: "1" -> pulse-first chip pair
    # (1, 0), "0" -> (0, 1); phase is a whole-chip lead-in, not a fractional
    # timing offset (that comes from channel()'s sto below)
    chips: list[int] = []
    for bit in payload:
        chips += [1, 0] if bit == "1" else [0, 1]
    sig = np.asarray(chips, np.float32)
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
        stream.append(_pulse_pair_burst(_PAYLOAD, phase))
        stream.append(_ambient(_TRAIL if i == last else _GAP, rng))
    return np.concatenate(stream)


def _modem() -> Modem:
    path: list[Step] = [OokEnvelopeStep(loop_bw=0.0), SliceStep()]
    return Modem(symbol_rate=_SYMBOL_RATE, path=path)


def _run(workdir: Path, capture: Path) -> np.ndarray:
    r = run_rx(
        _modem(),
        stage_registry(),
        sample_rate=_SAMPLE_RATE,
        start=IQ,
        workdir=workdir,
        source_io={"path": str(capture)},
    )
    assert r.status == "ok", r
    assert r.bitstream is not None
    return read_bits(r.bitstream.path)


def test_final_burst_flushes_on_an_unpadded_capture(tmp_path: Path) -> None:
    # the shipped live shape: iq_file_source -> complex_to_mag -> burst_sampler
    # at native rate, every hop ratio 1. A capture that ends AT its last
    # burst's end has no in-stream fall edge — only EOF finality can flush the
    # withheld burst, so the probe must be granted through ratio-1 hops.
    ensure_worker_warm()
    rng = np.random.default_rng(_SEED)
    stream = [_ambient(_LEAD, rng)]
    for phase in _PHASES[:-1]:
        stream.append(_pulse_pair_burst(_PAYLOAD, phase))
        stream.append(_ambient(_GAP, rng))
    stream.append(_pulse_pair_burst(_PAYLOAD, 0))  # ends at the burst's end
    cap = tmp_path / "unpadded.cf32"
    np.concatenate(stream).astype(np.complex64).tofile(cap)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    bits = _run(run_dir, cap)
    want = _chip_string(_PAYLOAD)
    found = "".join(str(bit) for bit in bits)
    assert found.count(want) == len(_PHASES), found.count(want)


def test_native_rate_open_loop_recovers_every_burst_at_ber0(tmp_path: Path) -> None:
    ensure_worker_warm()
    env = _synthetic_capture()
    clean = tmp_path / "clean.cf32"
    env.astype(np.complex64).tofile(clean)
    # A chip-pair-per-bit pattern at native sps=1 alternates every sample -
    # it already sits at the Nyquist rate, unlike the sps=2 sibling test's
    # deliberate worst-case sto=0.5 (chosen there for a single step edge).
    # channel()'s STO is an ideal (sinc) fractional delay: applied to this
    # already-Nyquist, non-bandlimited sequence it rings (measured: sto=0.2
    # BER 0 across 6 seeds and SNR 15-25 dB; sto=0.25 already fails on every
    # seed tried) - a real front-end bandlimits the analog signal before the
    # ADC ever samples it, so this ideal-delay ringing is a synthesis-model
    # ceiling, not the product's. sto=0.2 stays genuinely fractional (the
    # OSR=1 trap this suite's sim oracles guard against) while representing
    # the sample-clock jitter a real ADC free-running off the chip rate
    # would actually show.
    capture = channel(
        clean,
        tmp_path / "imp.cf32",
        snr_db=_SNR_DB,
        sto=_STO,
        sample_rate=_SAMPLE_RATE,
        seed=_SEED,
    )

    run_a, run_b = tmp_path / "run_a", tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    rx_a = _run(run_a, capture)
    rx_b = _run(run_b, capture)

    want = _chip_string(_PAYLOAD)
    found = "".join(str(bit) for bit in rx_a)
    assert found.count(want) == len(_PHASES), found.count(want)
    assert np.array_equal(rx_a, rx_b)
