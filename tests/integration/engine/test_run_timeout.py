import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from marconi.deadline import RunTimeout, check_deadline, remaining
from marconi.engine import run as run_module
from marconi.engine.backends.base import Backend, BlockCensus, Diagnostic, RunResult
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.io.bitfile import read_bits, write_bits
from marconi.engine.io.source import SourceSlice
from marconi.engine.quality import QualityReport
from marconi.engine.run import run_rx
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry, step_models
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType, RunStatus
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem


def _tiny_capture(tmp_path: Path) -> Path:
    # a short clean tone-ish IQ; content irrelevant — the run must time out first
    n = 200_000
    x = np.exp(1j * 2 * np.pi * 0.01 * np.arange(n)).astype(np.complex64)
    p = tmp_path / "cap.cf32"
    x.tofile(p)
    return p


def _fsk_modem() -> Modem:
    return Modem.from_spec(
        {"symbol_rate": 1000.0, "path": [{"conv": "fsk", "deviation": 1.0}]},
        step_models(),
    )


def test_run_rx_times_out_deterministically(tmp_path: Path) -> None:
    cap = _tiny_capture(tmp_path)
    with pytest.raises(RunTimeout):
        run_rx(
            _fsk_modem(),
            stage_registry(),
            sample_rate=8000.0,
            start=Descriptor(Level.IQ, ItemType.C),
            workdir=tmp_path,
            source=SourceSlice(path=cap),
            timeout=0.0,
        )


def test_run_rx_normal_timeout_still_succeeds(tmp_path: Path) -> None:
    cap = _tiny_capture(tmp_path)
    result = run_rx(
        _fsk_modem(),
        stage_registry(),
        sample_rate=8000.0,
        start=Descriptor(Level.IQ, ItemType.C),
        workdir=tmp_path,
        source=SourceSlice(path=cap),
        timeout=180.0,
    )
    assert result.status in ("ok", "empty")  # ran to completion, not timed out


_SOFT_SYMBOLS = Descriptor(Level.SYMBOLS, ItemType.F, carrier=Carrier.SOFT)
_SHORT_TIMEOUT = 0.5
_PAST_DEADLINE = 0.05
_LATE_BITS = 32
_LATE_CENSUS = [BlockCensus(block="b1", kind="slice", items_in=32, items_out=32)]
_LATE_MARKS = [0, 16]


def _sink_path(pipeline: GrPipeline) -> Path:
    sink = next(b for b in pipeline.blocks if b.id == pipeline.terminal_sink)
    return Path(str(sink.params["path"]))


class _LateWorkerBackend(Backend):
    """A worker that writes its sink and answers 'ok', optionally only after
    the clock ran out — the runner's own contract (the pipe outranks the clock,
    plus a teardown grace), so an ok result past the deadline is reachable
    whenever the payload lands in the last seconds of the budget."""

    def __init__(self, bits: npt.NDArray[np.uint8], *, oversleep: bool) -> None:
        self._bits = bits
        self._oversleep = oversleep

    @property
    def name(self) -> str:
        return "late-worker"

    def instantiate(self, pipeline: GrPipeline) -> object:
        raise NotImplementedError

    def run_pipeline(self, pipeline: GrPipeline, timeout: float = 30.0) -> RunResult:
        write_bits(_sink_path(pipeline), self._bits)
        if self._oversleep:
            time.sleep(max(0.0, timeout) + _PAST_DEADLINE)
        return RunResult(
            status=RunStatus.OK,
            census=list(_LATE_CENSUS),
            diagnostics=[Diagnostic(block="b1", key="bursts", marks=list(_LATE_MARKS))],
        )


def _soft_symbol_file(tmp_path: Path) -> Path:
    p = tmp_path / "syms.f32"
    np.linspace(-1.0, 1.0, _LATE_BITS, dtype=np.float32).tofile(p)
    return p


def _slice_modem() -> Modem:
    return Modem(symbol_rate=1.0, path=[SliceStep()])


class _TimedOutWorkerBackend(Backend):
    """The other wall-clock branch: the worker itself ran out of time and the
    runner answered with a TIMEOUT RunResult, naming where the signal died."""

    @property
    def name(self) -> str:
        return "timed-out-worker"

    def instantiate(self, pipeline: GrPipeline) -> object:
        raise NotImplementedError

    def run_pipeline(self, pipeline: GrPipeline, timeout: float = 30.0) -> RunResult:
        return RunResult(
            status=RunStatus.TIMEOUT,
            error="worker exceeded its wall-clock budget",
            stalled_at="b1",
            census=list(_LATE_CENSUS),
            diagnostics=[Diagnostic(block="b1", key="bursts", marks=list(_LATE_MARKS))],
        )


def _run_with(tmp_path: Path, backend: Backend) -> "run_module.PipelineResult":
    return run_rx(
        _slice_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=_SOFT_SYMBOLS,
        workdir=tmp_path,
        source=SourceSlice(path=_soft_symbol_file(tmp_path)),
        backend=backend,
        timeout=_SHORT_TIMEOUT,
    )


def test_a_worker_that_delivered_late_is_reported_not_discarded(
    tmp_path: Path,
) -> None:
    """A run whose GR segment answered 'ok' after the deadline passed met a
    check_deadline() sitting OUTSIDE the try that converts a timeout into a
    result: the census gradient, the marks and the diagnostics were thrown away
    with a raw RunTimeout, on exactly the run an operator has to retune from.
    The sink is COMPLETE on such a run, and an all-GR tail only renames it and
    reads its size, so the decode is reported too — a deadline honored one
    statement earlier left a finished stream on disk with no wire reference."""
    result = _run_with(
        tmp_path, _LateWorkerBackend(np.ones(_LATE_BITS, np.uint8), oversleep=True)
    )
    assert result.status == RunStatus.TIMEOUT
    assert result.census == _LATE_CENSUS
    assert result.marks == _LATE_MARKS
    assert result.diagnostics and result.diagnostics[0].key == "bursts"
    assert result.bitstream is not None
    assert result.bitstream.num_bits == _LATE_BITS
    assert int(read_bits(result.bitstream.path).sum()) == _LATE_BITS


def test_a_worker_that_timed_out_mid_run_reports_where_it_died(
    tmp_path: Path,
) -> None:
    """The other branch the run_rx docstring's timeout paragraph promises: a
    deadline landing while the GR pipeline is mid-run is REPORTED — status
    timeout with the census and diagnostics the worker did produce, and the
    block it stalled at — never raised at the caller."""
    result = _run_with(tmp_path, _TimedOutWorkerBackend())
    assert result.status == RunStatus.TIMEOUT
    assert result.census == _LATE_CENSUS
    assert result.diagnostics and result.diagnostics[0].key == "bursts"
    assert result.stalled_at == "b1"
    assert "wall-clock" in (result.error or "")


def test_a_timeout_while_scoring_keeps_the_stream_already_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GR segment and the coding tail both finished; only the quality pass
    outran the clock. The TIMEOUT result was rebuilt from the segment alone, so
    a decode sitting complete on disk was reported as having produced nothing."""

    def _scoring_that_outruns_the_clock(**kwargs: object) -> QualityReport:
        time.sleep(remaining() + _PAST_DEADLINE)
        check_deadline()
        raise AssertionError("the deadline passed and check_deadline let it by")

    monkeypatch.setattr(run_module, "assess_quality", _scoring_that_outruns_the_clock)
    result = _run_with(
        tmp_path, _LateWorkerBackend(np.ones(_LATE_BITS, np.uint8), oversleep=False)
    )
    assert result.status == RunStatus.TIMEOUT
    assert result.bitstream is not None
    assert result.bitstream.num_bits == _LATE_BITS
    assert int(read_bits(result.bitstream.path).sum()) == _LATE_BITS
    assert result.census == _LATE_CENSUS
