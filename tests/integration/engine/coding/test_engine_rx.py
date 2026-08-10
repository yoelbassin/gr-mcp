from pathlib import Path

import numpy as np
import pytest

from marconi.engine.backends.base import Backend, BlockCensus, RunResult
from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.coding.stages_bits import SyncWordStep
from marconi.engine.coding.stages_symbols import SymbolMapStep, SyncSymbolsStep
from marconi.engine.compile.compiler import (
    compile_modem,
    compile_pipeline,
)
from marconi.engine.compile.errors import CompileError
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.io.bitfile import read_bits, write_bits
from marconi.engine.io.source import SourceSlice
from marconi.engine.modulation.css.stages import (
    ChirpSyncStep,
    CssDemapStep,
    DechirpStep,
)
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.run import run_rx
from marconi.engine.stages.conditioning import InvertStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem, Symbolstream

IQ = Descriptor(Level.IQ, ItemType.C)
BITS = Descriptor(Level.BITS, ItemType.B)
SOFT_SYMBOLS = Descriptor(Level.SYMBOLS, ItemType.F, carrier=Carrier.SOFT)
HARD_SYMBOLS = Descriptor(Level.SYMBOLS, ItemType.S, carrier=Carrier.HARD)
_CSS_SF, _CSS_OSR, _CSS_ZP, _CSS_SYM = 7, 2, 4, 1.0
_FLAG_BITS = np.concatenate(
    [
        np.zeros(3, np.uint8),
        np.array([0, 1, 1, 1, 1, 1, 1, 0], np.uint8),
        np.ones(24, np.uint8),
    ]
)
_SOFT_VALUES = np.array([-0.5, 0.75, 1.25, -1.0, 0.5], np.float32)


def _flag_stream(tmp_path: Path) -> Bitstream:
    p = tmp_path / "in.u8"
    write_bits(p, _FLAG_BITS)
    return Bitstream(path=p, num_bits=int(_FLAG_BITS.size))


def _flag_soft_symbols(tmp_path: Path) -> Path:
    soft = np.where(_FLAG_BITS == 1, 1.0, -1.0).astype(np.float32)
    p = tmp_path / "flag.f32"
    soft.tofile(p)
    return p


def _sync_word_modem() -> Modem:
    return Modem(
        symbol_rate=1.0,
        path=[SyncWordStep(sync="7e")],
    )


def _gr_modem() -> Modem:
    return Modem(
        symbol_rate=1.0,
        path=[
            SliceStep(),
            SyncWordStep(sync="7e"),
        ],
    )


def _sync_symbols_modem() -> Modem:
    return Modem(
        symbol_rate=1.0,
        path=[SyncSymbolsStep(pattern=[1, -1])],
    )


def _symbol_map_modem() -> Modem:
    return Modem(
        symbol_rate=1.0,
        path=[SymbolMapStep(code_bits=1, data_bits=1, table=[0, 1])],
    )


def _css_tx_modem(sf: int, oversample: int, zero_pad: int) -> Modem:
    return Modem(
        symbol_rate=_CSS_SYM,
        path=[
            ChirpSyncStep(
                sf=sf,
                oversample=oversample,
                zero_pad=zero_pad,
                preamble_len=8,
                sfd_symbols=2.25,
                sync_symbols=2,
            ),
            DechirpStep(sf=sf, oversample=oversample, zero_pad=zero_pad),
            CssDemapStep(sf=sf),
        ],
    )


def _css_symbols_rx_modem(sf: int, oversample: int, zero_pad: int) -> Modem:
    return Modem(
        symbol_rate=_CSS_SYM,
        path=[
            ChirpSyncStep(
                sf=sf,
                oversample=oversample,
                zero_pad=zero_pad,
                preamble_len=8,
                sfd_symbols=2.25,
                sync_symbols=2,
            ),
            DechirpStep(sf=sf, oversample=oversample, zero_pad=zero_pad),
        ],
    )


def _soft_symbolstream(tmp_path: Path, marks: list[int]) -> Symbolstream:
    p = tmp_path / "syms.f32"
    _SOFT_VALUES.tofile(p)
    return Symbolstream(
        path=p, num_symbols=int(_SOFT_VALUES.size), item_type="f", marks=marks
    )


class _CannedBackend(Backend):
    def __init__(self, result: RunResult) -> None:
        self.result = result

    @property
    def name(self) -> str:
        return "canned"

    def instantiate(self, pipeline: GrPipeline) -> object:
        raise NotImplementedError

    def run_pipeline(self, pipeline: GrPipeline, timeout: float = 30.0) -> RunResult:
        return self.result


def test_pure_coding_run_seeds_windows_and_reports_census(tmp_path: Path) -> None:
    res = run_rx(
        _sync_word_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=_flag_stream(tmp_path),
    )
    assert res.status == "ok"
    assert res.windows == [11]
    assert res.bitstream is not None
    assert read_bits(res.bitstream.path).size == 35
    assert [r.kind for r in res.census] == ["sync_word"]
    assert res.census[0].windows_out == 1


def test_gr_plus_coding_run_shares_windows_and_census(tmp_path: Path) -> None:
    ensure_worker_warm()
    res = run_rx(
        _gr_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=SOFT_SYMBOLS,
        workdir=tmp_path,
        source=SourceSlice(path=_flag_soft_symbols(tmp_path)),
    )
    assert res.status == "ok", res.error
    assert res.windows == [11]
    assert len(res.census) >= 3
    assert res.census[-1].kind == "sync_word"


def test_gr_only_hard_symbols_final_get_i16_suffix(tmp_path: Path) -> None:
    ensure_worker_warm()
    sf, osr, zp = _CSS_SF, _CSS_OSR, _CSS_ZP
    rate = osr * (1 << sf) * _CSS_SYM
    bits = np.random.default_rng(5).integers(0, 2, sf * 20).astype(np.uint8)
    bits_path = tmp_path / "in.u8"
    write_bits(bits_path, bits)
    iq_path = tmp_path / "css.iq"
    tx = compile_modem(
        _css_tx_modem(sf, osr, zp),
        stage_registry(),
        direction="tx",
        sample_rate=rate,
        start=IQ,
        source_io={"path": str(bits_path)},
        sink_io={"path": str(iq_path)},
    )
    assert GnuRadioBackend().run_pipeline(tx).status == "ok"

    rx_modem = _css_symbols_rx_modem(sf, osr, zp)
    cp = compile_pipeline(
        rx_modem,
        stage_registry(),
        direction="rx",
        sample_rate=rate,
        start=IQ,
        source_io={"path": str(iq_path)},
        sink_io={"path": "unused"},
    )
    assert cp.coding is None
    assert cp.final.level is Level.SYMBOLS and cp.final.item_type == "s"

    res = run_rx(
        rx_modem,
        stage_registry(),
        sample_rate=rate,
        start=IQ,
        workdir=tmp_path,
        source=SourceSlice(path=iq_path),
    )
    assert res.status == "ok", res
    assert res.symbolstream is not None
    assert res.symbolstream.item_type == "s"
    assert res.symbolstream.path.suffix == ".i16"
    assert res.symbolstream.path.is_file()
    assert res.symbolstream.num_symbols >= bits.size // sf

    on_disk = np.fromfile(res.symbolstream.path, dtype=np.int16)
    assert on_disk.size == res.symbolstream.num_symbols


def test_empty_gr_run_propagates_status_and_stalled_at(tmp_path: Path) -> None:
    canned = RunResult(
        status="empty",
        error="terminal sink wrote 0 items (no items past slice)",
        stalled_at="b1",
        census=[BlockCensus(block="b0", kind="soft_bits_file_source", items_out=0)],
    )
    # a real GR run creates the sink file even when it writes 0 items
    (tmp_path / "seam.dat").touch()
    res = run_rx(
        _gr_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=SOFT_SYMBOLS,
        workdir=tmp_path,
        source=SourceSlice(path=tmp_path / "in.f32"),
        backend=_CannedBackend(canned),
    )
    assert res.status == "empty"
    assert res.stalled_at == "b1"
    assert res.error == canned.error
    # empty means no product stream — but the run keeps its full report: the
    # census and the quality verdict the run_rx docstring promises
    # unconditionally
    assert res.bitstream is None and res.symbolstream is None
    assert res.quality is not None
    # the coding tail now runs (on zero bits) so its census row appears —
    # the gradient a parameter search needs even on an empty run
    assert [r.kind for r in res.census] == ["soft_bits_file_source", "sync_word"]


def test_soft_symbols_final_round_trips_float_values(tmp_path: Path) -> None:
    res = run_rx(
        _sync_symbols_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=SOFT_SYMBOLS,
        workdir=tmp_path,
        input_stream=_soft_symbolstream(tmp_path, marks=[]),
    )
    assert res.status == "ok"
    assert res.symbolstream is not None
    assert res.symbolstream.item_type == "f"
    assert res.symbolstream.num_symbols == int(_SOFT_VALUES.size)
    out = np.fromfile(res.symbolstream.path, np.float32)
    assert np.array_equal(out, _SOFT_VALUES)
    assert [r.kind for r in res.census] == ["sync_symbols"]


def test_marks_split_acquisition_harvest_from_coding_state(tmp_path: Path) -> None:
    res = run_rx(
        _sync_symbols_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=SOFT_SYMBOLS,
        workdir=tmp_path,
        input_stream=_soft_symbolstream(tmp_path, marks=[0]),
    )
    assert res.status == "ok"
    assert res.marks == [0]
    assert res.symbolstream is not None
    assert res.symbolstream.marks == [2]


def test_iq_terminal_pipeline_returns_conditioned_iq(tmp_path: Path) -> None:
    # an IQ-final path (no demod) is now a first-class conditioning result: the
    # run returns the conditioned sub-band itself as a complex ("c") stream, to
    # feed back into survey/stream_stats or the caller's own soft work.
    iq = tmp_path / "in.cf32"
    np.ones(64, np.complex64).tofile(iq)
    modem = Modem(symbol_rate=1.0, path=[InvertStep()])
    res = run_rx(
        modem,
        stage_registry(),
        sample_rate=8.0,
        start=Descriptor(Level.IQ, ItemType.C),
        workdir=tmp_path,
        source=SourceSlice(path=iq),
    )
    assert res.status == "ok"
    assert res.symbolstream is not None
    assert res.symbolstream.item_type == "c"
    assert res.symbolstream.num_symbols == 64


def test_nonfinite_iq_source_is_an_error_before_the_gr_run(tmp_path: Path) -> None:
    iq = np.ones(64, np.complex64)
    iq[3] = np.nan + 0j
    p = tmp_path / "in.cf32"
    iq.tofile(p)
    modem = Modem(
        symbol_rate=2400.0,
        path=[
            FskStep(deviation=600.0),
            SliceStep(),
        ],
    )
    res = run_rx(
        modem,
        stage_registry(),
        sample_rate=48000.0,
        start=Descriptor(Level.IQ, ItemType.C),
        workdir=tmp_path,
        source=SourceSlice(path=p),
        backend=_CannedBackend(RunResult(status="ok")),
    )
    assert res.status == "error"
    assert res.error is not None and "non-finite" in res.error and "3" in res.error


def test_nonfinite_soft_symbol_input_is_an_error(tmp_path: Path) -> None:
    soft = np.array([0.5, -1.0, np.inf, 1.0], np.float32)
    p = tmp_path / "syms.f32"
    soft.tofile(p)
    stream = Symbolstream(path=p, num_symbols=int(soft.size), item_type="f")
    res = run_rx(
        _sync_symbols_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=SOFT_SYMBOLS,
        workdir=tmp_path,
        input_stream=stream,
    )
    assert res.status == "error"
    assert res.error is not None and "non-finite" in res.error and "2" in res.error


def test_pure_coding_path_without_input_stream_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CompileError, match="input_stream"):
        run_rx(
            _sync_word_modem(),
            stage_registry(),
            sample_rate=1.0,
            start=BITS,
            workdir=tmp_path,
        )


def test_gr_segment_with_input_stream_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CompileError, match="source_io"):
        run_rx(
            _gr_modem(),
            stage_registry(),
            sample_rate=1.0,
            start=SOFT_SYMBOLS,
            workdir=tmp_path,
            input_stream=_flag_stream(tmp_path),
        )


def test_bitstream_input_against_symbol_boundary_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CompileError, match="Bitstream"):
        run_rx(
            _sync_symbols_modem(),
            stage_registry(),
            sample_rate=1.0,
            start=SOFT_SYMBOLS,
            workdir=tmp_path,
            input_stream=_flag_stream(tmp_path),
        )


def test_symbolstream_input_against_bit_boundary_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CompileError, match="Symbolstream"):
        run_rx(
            _sync_word_modem(),
            stage_registry(),
            sample_rate=1.0,
            start=BITS,
            workdir=tmp_path,
            input_stream=_soft_symbolstream(tmp_path, marks=[]),
        )


def test_symbolstream_item_type_mismatch_against_boundary_is_an_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(CompileError, match="item_type"):
        run_rx(
            _symbol_map_modem(),
            stage_registry(),
            sample_rate=1.0,
            start=HARD_SYMBOLS,
            workdir=tmp_path,
            input_stream=_soft_symbolstream(tmp_path, marks=[]),
        )
