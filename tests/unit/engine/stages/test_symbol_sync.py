"""symbol_sync: the timing-recovery half of psk_demod, exposed as its own
IQ->IQ stage so a 1-sps seam opens between timing and carrier recovery -- the
seam a symbol-spaced equalizer needs to join a real decode.

psk_demod fuses matched-filter + Gardner timing + costas and hides the sps
decimation inside its IQ->SYMBOLS level change (rate_factor 1.0). This stage
stays at IQ and decimates honestly: an explicit `sps` param drives
rate_factor = 1/sps, checked against the pipeline's delivered rate exactly as
Channelize's explicit `decim` is (issue 06).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers._dsp import channel, read_complex, resolved_ser, tx_sym_indices, write_bits
from pydantic import ValidationError

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.modulation.psk.stages import (
    PskDemapStep,
    PskDemodStep,
    SymbolSyncStep,
)
from marconi.engine.stages.conditioning import AgcStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import AgcMode, ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_SR, _SYM, _SPS = 4.0, 1.0, 4


def _const_points(order: int) -> tuple[np.ndarray, int]:
    from gnuradio import digital  # in-process GR for oracle ground truth (allowed)

    c = {2: digital.constellation_bpsk, 4: digital.constellation_qpsk}[order]()
    return np.asarray(c.points()), c.bits_per_symbol()


def _tx_iq(order: int, bits: np.ndarray, tmp_path: Path) -> Path:
    bp = write_bits(tmp_path / "in.bits", bits)
    clean = tmp_path / "tx.iq"
    modem = Modem(
        symbol_rate=_SYM,
        path=[
            PskDemodStep(order=PskOrder(order)),
            PskDemapStep(order=PskOrder(order)),
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="tx",
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(bp)},
        sink_io={"path": str(clean)},
    )
    assert GnuRadioBackend().run_pipeline(pipe).status == "ok"
    return clean


def _recover(src: Path, snk: Path) -> np.ndarray:
    modem = Modem(
        symbol_rate=_SYM,
        path=[
            AgcStep(mode=AgcMode.POWER, window_symbols=32.0),
            SymbolSyncStep(sps=_SPS),
            AgcStep(mode=AgcMode.POWER),
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=120.0).status == "ok"
    return read_complex(snk)


def test_symbol_sync_recovers_timing_ser0(tmp_path: Path) -> None:
    """Matched filter + Gardner alone, no carrier loop: a fractional timing
    offset is recovered and the symbols decode with zero errors (the static
    residual rotation is resolved by resolved_ser, as in the psk gate)."""
    ensure_worker_warm()
    order = 2
    points, k = _const_points(order)
    bits = np.random.default_rng(1).integers(0, 2, 6000).astype(np.uint8)
    clean = _tx_iq(order, bits, tmp_path)
    imp = tmp_path / "imp.iq"
    channel(clean, imp, snr_db=28.0, sto=1.7, sfo_ppm=50.0, sample_rate=_SR, seed=1)
    rx = _recover(imp, tmp_path / "rx.cf32")
    tsi = tx_sym_indices(bits, k)
    assert resolved_ser(rx, tsi, points, order, settle=300, max_shift=700) == 0.0


def test_symbol_sync_decimates_by_sps() -> None:
    stage = stage_registry()["symbol_sync"]
    step = SymbolSyncStep(sps=_SPS)
    assert stage.rate_factor(step) == 1.0 / _SPS
    assert stage.required_input_rate(step, _SYM) == _SPS * _SYM
    out = stage.out_descriptor(Descriptor(Level.IQ, ItemType.C), step)
    assert out.level is Level.IQ
    assert out.amplitude is Amplitude.UNKNOWN


def test_symbol_sync_builds_matched_filter_then_gardner() -> None:
    ctx = CompileContext(Descriptor(Level.IQ, ItemType.C), _SR, _SYM)
    stage_registry()["symbol_sync"].emit_rx(ctx, SymbolSyncStep(sps=_SPS))
    kinds = [b.kind for b in ctx.build("t", _SR).blocks]
    assert kinds == ["rrc_filter_ccf", "symbol_sync_cc"]


def test_symbol_sync_rejects_a_rate_that_does_not_deliver_sps(tmp_path: Path) -> None:
    """The explicit sps must match the rate the pipeline actually delivers, or
    the matched filter and decimation disagree with the compiler's rate model."""
    modem = Modem(
        symbol_rate=_SYM,
        # start at 4 sps but claim 8: the required-input-rate guard must fire.
        path=[SymbolSyncStep(sps=8)],
    )
    with pytest.raises(CompileError, match="input sample rate"):
        compile_modem(
            modem,
            stage_registry(),
            direction="rx",
            sample_rate=_SR,
            start=Descriptor(Level.IQ, ItemType.C, amplitude=Amplitude.RMS_UNITY),
            source_io={"path": "in.iq"},
            sink_io={"path": "out.iq"},
        )


def test_symbol_sync_requires_a_known_scale() -> None:
    stage = stage_registry()["symbol_sync"]
    assert stage.accepts_amplitude == frozenset(
        {Amplitude.PEAK_UNITY, Amplitude.MEAN_MAG_UNITY, Amplitude.RMS_UNITY}
    )
    with pytest.raises(CompileError):
        compile_modem(
            Modem(
                symbol_rate=_SYM,
                path=[SymbolSyncStep(sps=_SPS)],
            ),
            stage_registry(),
            direction="rx",
            sample_rate=_SR,
            start=IQ,  # UNKNOWN amplitude
            source_io={"path": "in.iq"},
            sink_io={"path": "out.iq"},
        )


def test_symbol_sync_rejects_invalid_params() -> None:
    model = stage_registry()["symbol_sync"].step_model
    for bad in ({"sps": 1}, {"sps": 4, "alpha": 0.0}, {"sps": 4, "loop_bw": 0.0}):
        with pytest.raises(ValidationError):
            model.model_validate(bad)


def test_symbol_sync_output_carries_soft_iq() -> None:
    stage = stage_registry()["symbol_sync"]
    out = stage.out_descriptor(
        Descriptor(Level.IQ, ItemType.C, carrier=Carrier.HARD), SymbolSyncStep(sps=_SPS)
    )
    assert out.item_type == "c"
