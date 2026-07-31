"""The payoff: the blind equalizer joins a real decode through the 1-sps seam
that symbol_sync opens.

A frequency-selective (multipath) channel closes the eye -- matched filter and
Gardner timing alone leave the symbols badly wrong. Dropping a symbol-spaced CMA
equalizer into the seam between timing and the decision inverts the channel's
inter-symbol interference and the same symbols decode. This is the composition
psk_demod's monolith could not express, now that symbol_sync exposes the seam.

The channel carries multipath but no carrier offset, so the residual rotation the
blind equalizer leaves is static -- resolved_ser resolves it exactly as the psk
gate does. The input is presented pre-normalized (as an agc would leave it): a
fast streaming agc fighting a strong echo slips Gardner's timing, an agc/timing
interaction that is its own concern (see test_psk_roundtrip's slow-agc note), not
the equalizer's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from engine._dsp import channel, read_complex, resolved_ser, tx_sym_indices, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.modulation.psk.stages import (
    PskDemapStep,
    PskDemodStep,
    SymbolSyncStep,
)
from marconi.engine.stages.conditioning import EqualizerStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
IQ_RMS = Descriptor(Level.IQ, ItemType.C, amplitude=Amplitude.RMS_UNITY)
_SR, _SYM, _SPS, _ORDER = 4.0, 1.0, 4, 2
# A causal echo at one full symbol (4 samples), 0.8 of the main path: severe
# symbol-rate ISI the matched filter + slicer cannot open on their own.
_MULTIPATH: list[complex] = [1.0, 0.0, 0.0, 0.0, 0.8]


def _bpsk_points() -> tuple[np.ndarray, int]:
    from gnuradio import digital  # in-process GR for oracle ground truth (allowed)

    c = digital.constellation_bpsk()
    return np.asarray(c.points()), c.bits_per_symbol()


def _compile(
    modem: Modem, direction: str, start: Descriptor, src: Path, snk: Path
) -> GrPipeline:
    return compile_modem(
        modem,
        stage_registry(),
        direction=direction,
        sample_rate=_SR,
        start=start,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


def _tx_multipath_iq(bits: np.ndarray, tmp_path: Path, seed: int) -> Path:
    """bits -> pulse-shaped IQ -> multipath+AWGN channel -> RMS-normalized file."""
    be = GnuRadioBackend()
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp = tmp_path / "tx.iq", tmp_path / "imp.iq"
    tx = Modem(
        symbol_rate=_SYM,
        path=[
            PskDemodStep(order=PskOrder(_ORDER)),
            PskDemapStep(order=PskOrder(_ORDER)),
        ],
    )
    assert be.run_pipeline(_compile(tx, "tx", IQ, bp, clean)).status == "ok"
    channel(
        clean,
        imp,
        snr_db=25.0,
        multipath=_MULTIPATH,
        sample_rate=_SR,
        seed=seed,
    )
    x = np.fromfile(imp, np.complex64)
    (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64).tofile(imp)
    return imp


def _rx(src: Path, snk: Path, *, equalize: bool) -> np.ndarray:
    path: list[Step] = [SymbolSyncStep(sps=_SPS)]
    if equalize:
        path.append(EqualizerStep(num_taps=21, step_size=0.01))
    modem = Modem(symbol_rate=_SYM, path=path)
    pipe = _compile(modem, "rx", IQ_RMS, src, snk)
    assert GnuRadioBackend().run_pipeline(pipe, timeout=120.0).status == "ok"
    return read_complex(snk)


def test_equalizer_enables_a_multipath_decode(tmp_path: Path) -> None:
    ensure_worker_warm()
    points, k = _bpsk_points()
    for seed in (1, 2):
        bits = np.random.default_rng(seed).integers(0, 2, 8000).astype(np.uint8)
        imp = _tx_multipath_iq(bits, tmp_path, seed)
        tsi = tx_sym_indices(bits, k)

        no_eq = _rx(imp, tmp_path / "no.cf32", equalize=False)
        ser_no = resolved_ser(no_eq, tsi, points, _ORDER, settle=300, max_shift=700)
        assert ser_no > 0.05, f"multipath should close the eye (seed {seed})"

        eq = _rx(imp, tmp_path / "eq.cf32", equalize=True)
        ser_eq = resolved_ser(eq, tsi, points, _ORDER, settle=1500, max_shift=1900)
        assert ser_eq < 0.01, f"equalizer should reopen the decode (seed {seed})"
