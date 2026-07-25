from pathlib import Path

import numpy as np
import pytest

from marconi.core.bitfile import read_bits, write_bits
from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream, Symbolstream
from marconi.phy.backends.base import Backend, BlockCensus, RunResult
from marconi.phy.backends.gnuradio.runner import ensure_worker_warm
from marconi.phy.engine import run_rx
from marconi.phy.ir import GrPipeline
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

BITS = Descriptor(Level.BITS, "b")
SOFT_SYMBOLS = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
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


def _sync_word_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=1.0,
        path=[ModemStep(conv="sync_word", params={"sync": "7e"})],
    )


def _gr_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="slice"),
            ModemStep(conv="sync_word", params={"sync": "7e"}),
        ],
    )


def _sync_symbols_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=1.0,
        path=[ModemStep(conv="sync_symbols", params={"pattern": [1, -1]})],
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
        source_io={"path": str(_flag_soft_symbols(tmp_path))},
    )
    assert res.status == "ok", res.error
    assert res.windows == [11]
    assert len(res.census) >= 3
    assert res.census[-1].kind == "sync_word"


def test_empty_gr_run_propagates_status_and_stalled_at(tmp_path: Path) -> None:
    canned = RunResult(
        status="empty",
        error="terminal sink wrote 0 items (no items past slice)",
        stalled_at="b1",
        census=[BlockCensus(block="b0", kind="soft_bits_file_source", items_out=0)],
    )
    res = run_rx(
        _gr_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=SOFT_SYMBOLS,
        workdir=tmp_path,
        source_io={"path": str(tmp_path / "in.f32")},
        backend=_CannedBackend(canned),
    )
    assert res.status == "empty"
    assert res.stalled_at == "b1"
    assert res.error == canned.error
    assert res.bitstream is None and res.symbolstream is None
    assert [r.kind for r in res.census] == ["soft_bits_file_source"]


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


def test_pure_coding_path_without_input_stream_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input_stream"):
        run_rx(
            _sync_word_modem(),
            stage_registry(),
            sample_rate=1.0,
            start=BITS,
            workdir=tmp_path,
        )


def test_gr_segment_with_input_stream_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_io"):
        run_rx(
            _gr_modem(),
            stage_registry(),
            sample_rate=1.0,
            start=SOFT_SYMBOLS,
            workdir=tmp_path,
            input_stream=_flag_stream(tmp_path),
        )
