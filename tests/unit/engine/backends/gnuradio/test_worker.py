import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marconi.engine.backends.base import Diagnostic, RunResult, find_diagnostic
from marconi.engine.backends.gnuradio import worker as worker_mod
from marconi.engine.backends.gnuradio.runner import (
    GnuRadioBackend,
    _resolve_result,
    _run_in_subprocess,
    ensure_worker_warm,
)
from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline
from marconi.engine.types.enums import RunStatus


def _passthrough(src: Path, dst: Path) -> GrPipeline:
    return GrPipeline(
        name="pt",
        sample_rate=1.0,
        blocks=[
            GrBlock(id="s", kind="iq_file_source", params={"path": str(src)}),
            GrBlock(id="k", kind="iq_file_sink", params={"path": str(dst)}),
        ],
        connections=[GrConnection(src_block="s", dst_block="k")],
    )


def test_run_pipeline_passthrough_byte_exact(tmp_path: Path) -> None:
    ensure_worker_warm()
    rng = np.random.default_rng(1)
    data = (rng.standard_normal(800) + 1j * rng.standard_normal(800)).astype(
        np.complex64
    )
    src = tmp_path / "in.iq"
    dst = tmp_path / "out.iq"
    data.tofile(src)
    res = GnuRadioBackend().run_pipeline(_passthrough(src, dst))
    assert res.status == "ok"
    assert np.array_equal(np.fromfile(dst, dtype=np.complex64), data)


def test_bad_kind_returns_error_without_killing_parent() -> None:
    pipe = GrPipeline(
        name="bad",
        sample_rate=1.0,
        blocks=[GrBlock(id="z", kind="nonesuch", params={})],
        connections=[],
    )
    res = GnuRadioBackend().run_pipeline(pipe)
    assert res.status == "error" and res.error  # surfaced, parent alive


def test_parent_process_stays_gnuradio_free() -> None:
    code = textwrap.dedent(
        """
        import sys, importlib.abc

        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "gnuradio" or name.startswith("gnuradio."):
                    raise ImportError("gnuradio banned in parent")
                return None

        sys.meta_path.insert(0, _Block())
        from marconi.engine.backends.gnuradio.runner import (
            GnuRadioBackend, ensure_worker_warm,
        )
        GnuRadioBackend()
        ensure_worker_warm()
        assert "gnuradio" not in sys.modules, "parent imported gnuradio"
        print("OK")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr


# ─── Failure paths ────────────────────────────────────────────────


def _raising_basic_flowgraph(sink: Path) -> Any:
    from gnuradio import blocks as gb
    from gnuradio import gr

    class _Boom(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="boom", in_sig=[np.complex64], out_sig=[np.complex64]
            )

        def forecast(self, noutput_items: int, ninputs: int) -> list[int]:
            return [1] * ninputs

        def general_work(self, input_items: Any, output_items: Any) -> int:
            raise RuntimeError("boom-marker")

    tb = gr.top_block("boom")
    boom = _Boom()
    tb.connect(gb.vector_source_c([0.1 + 0.1j] * 4096, False), boom)
    tb.connect(boom, gb.file_sink(gr.sizeof_gr_complex, str(sink), False))
    tb._py_instances = {"boom": boom}
    return tb


def _raising_sync_flowgraph(sink: Path) -> Any:
    from gnuradio import blocks as gb
    from gnuradio import gr

    class _SyncBoom(gr.sync_block):
        def __init__(self) -> None:
            gr.sync_block.__init__(
                self, name="syncboom", in_sig=[np.complex64], out_sig=[np.complex64]
            )

        def work(self, input_items: Any, output_items: Any) -> int:
            raise RuntimeError("boom-marker")

    tb = gr.top_block("syncboom")
    boom = _SyncBoom()
    tb.connect(gb.vector_source_c([0.1 + 0.1j] * 4096, False), boom)
    tb.connect(boom, gb.file_sink(gr.sizeof_gr_complex, str(sink), False))
    tb._py_instances = {"syncboom": boom}
    return tb


@pytest.mark.parametrize(
    "build_raising",
    [_raising_basic_flowgraph, _raising_sync_flowgraph],
    ids=["general_work", "sync_work"],
)
def test_embedded_raise_reports_error_and_keeps_sink(
    tmp_path: Path, monkeypatch: Any, build_raising: Any
) -> None:
    """A raise on a GR block thread must not hang tb.run() (the pre-trampoline
    behavior) and must surface as status=error with the traceback, with the
    caller-named sink left in place."""
    import marconi.engine.backends.gnuradio.build as build_mod

    sink = tmp_path / "out.iq"
    monkeypatch.setattr(
        build_mod, "build_top_block", lambda pipeline: build_raising(sink)
    )
    pipe = GrPipeline(
        name="boom",
        sample_rate=1.0,
        blocks=[GrBlock(id="k", kind="iq_file_sink", params={"path": str(sink)})],
        connections=[],
    )
    box: dict[str, RunResult] = {}
    t = threading.Thread(
        target=lambda: box.setdefault("r", worker_mod._run_flowgraph(pipe)),
        daemon=True,
    )
    t.start()
    t.join(60.0)
    assert "r" in box, "flowgraph hung: crash trampoline failed to unwind tb.run()"
    res = box["r"]
    assert res.status == "error"
    assert "boom-marker" in (res.error or "")
    assert "RuntimeError" in (res.error or "")  # full traceback, not a bare repr
    assert sink.exists(), "caller-named sink deleted; partial output is evidence"
    assert str(sink) in res.artifacts


def test_scheduler_abort_flags_error() -> None:
    """A block demanding more items than the stream buffer holds aborts the
    scheduler while tb.run() returns normally (probed); the worker
    must convert the captured block_executor message into status=error.

    Statistical gate: the abort MESSAGE is racy inside GR itself — probed
    2026-07 under load, 9/20 processes print nothing when exiting right after
    tb.run(), and 2/20 still nothing after a 1s settle — so the mechanism
    must fire in one of eight isolated attempts, and the detector it feeds is
    best-effort by construction (see _flag_scheduler_abort)."""
    code = textwrap.dedent(
        """
        import time
        import numpy as np
        from gnuradio import blocks as gb
        from gnuradio import gr

        class _Greedy(gr.basic_block):
            def __init__(self):
                gr.basic_block.__init__(
                    self, name="greedy",
                    in_sig=[np.complex64], out_sig=[np.complex64],
                )

            def forecast(self, noutput_items, ninputs):
                return [1 << 20] * ninputs

            def general_work(self, input_items, output_items):
                self.consume(0, len(input_items[0]))
                return 0

        tb = gr.top_block("greedy")
        greedy = _Greedy()
        snk = gb.vector_sink_c()
        tb.connect(gb.vector_source_c([0j] * 4096, False), greedy)
        tb.connect(greedy, snk)
        tb.run()
        time.sleep(0.5)
        """
    )
    text = ""
    for _ in range(8):
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert out.returncode == 0, out.stderr
        text = out.stdout + out.stderr
        if "block_executor" in text:
            break
    assert "block_executor" in text, "no scheduler abort on stdio in 8 attempts"
    flagged = worker_mod._flag_scheduler_abort(RunResult(status=RunStatus.OK), text)
    assert flagged.status == "error"
    assert "scheduler abort" in (flagged.error or "")
    clean = worker_mod._flag_scheduler_abort(RunResult(status=RunStatus.OK), "")
    assert clean.status == "ok"


def test_timeout_reports_and_keeps_partial_sink(tmp_path: Path) -> None:
    ensure_worker_warm()
    rng = np.random.default_rng(2)
    data = (rng.standard_normal(4096) + 1j * rng.standard_normal(4096)).astype(
        np.complex64
    )
    src = tmp_path / "in.iq"
    data.tofile(src)
    dst = tmp_path / "out.iq"
    pipe = GrPipeline(
        name="forever",
        sample_rate=1.0,
        blocks=[
            GrBlock(
                id="s",
                kind="iq_file_source",
                params={"path": str(src), "repeat": True},
            ),
            GrBlock(id="k", kind="iq_file_sink", params={"path": str(dst)}),
        ],
        connections=[GrConnection(src_block="s", dst_block="k")],
    )
    res = GnuRadioBackend().run_pipeline(pipe, timeout=1.5)
    assert res.status == "timeout"
    assert "timeout" in (res.error or "")
    assert dst.exists(), "partial sink must survive the worker kill"
    assert dst.stat().st_size > 0
    assert str(dst) in res.artifacts


def test_abnormal_worker_exit_is_error_with_cause() -> None:
    ensure_worker_warm()
    res = _run_in_subprocess("this is not a pipeline", timeout=30.0)
    assert res.status == "error"
    assert "abnormally" in (res.error or "")
    assert "exitcode" in (res.error or "")
    # the child's traceback was captured and attached, not discarded
    assert "ValidationError" in (res.error or "")


def _bulky_result_worker(payload_json: str, conn: Any, capture_path: str) -> None:
    marks = list(range(60_000))
    result = RunResult(
        status=RunStatus.OK,
        diagnostics=[Diagnostic(block="probe", key="marks", marks=marks)],
    )
    conn.send(result.model_dump_json())
    conn.close()


def test_result_larger_than_pipe_buffer_is_not_a_timeout() -> None:
    """A result past the OS pipe buffer blocks the worker in conn.send until
    the parent drains it; the pre-fix join-before-recv waited out the whole
    deadline, killed the finished worker, and reported timeout — then raised
    an uncaught OSError off the truncated frame."""
    ensure_worker_warm()
    t0 = time.monotonic()
    res = _run_in_subprocess("{}", timeout=15.0, target=_bulky_result_worker)
    elapsed = time.monotonic() - t0
    assert res.status == "ok"
    row = find_diagnostic(res.diagnostics, "probe", "marks")
    assert row is not None and row.marks == list(range(60_000))
    assert elapsed < 10.0, f"large result waited out the deadline ({elapsed:.1f}s)"


def _wedged_teardown_worker(payload_json: str, conn: Any, capture_path: str) -> None:
    conn.send(RunResult(status=RunStatus.OK).model_dump_json())
    conn.close()
    time.sleep(120.0)  # a GR destructor hang after the result was delivered


def test_delivered_result_is_not_held_to_the_full_deadline() -> None:
    """Once the payload has arrived, the parent waits only the short teardown
    grace before reaping - a worker wedged in GR destruction must not hold a
    finished 5 s decode for the rest of a 180 s deadline."""
    ensure_worker_warm()
    t0 = time.monotonic()
    res = _run_in_subprocess("{}", timeout=60.0, target=_wedged_teardown_worker)
    elapsed = time.monotonic() - t0
    assert res.status == "ok"
    assert elapsed < 30.0, f"finished run held for {elapsed:.1f}s"


def test_result_pipe_outranks_timeout_kill() -> None:
    """A worker that reported ok but lingered in GR teardown past the deadline
    is killed — its completed result (and artifacts) must survive."""
    ok = RunResult(status=RunStatus.OK, artifacts=["/tmp/x.iq"]).model_dump_json()
    kept = _resolve_result(True, ok, -15, "")
    assert kept.status == "ok"
    assert kept.artifacts == ["/tmp/x.iq"]


def test_resolve_timeout_and_abnormal_shapes() -> None:
    t = _resolve_result(True, None, -9, "tail-text")
    assert t.status == "timeout" and "tail-text" in (t.error or "")
    a = _resolve_result(False, None, 1, "boom")
    assert a.status == "error"
    assert "exitcode=1" in (a.error or "") and "boom" in (a.error or "")


def _census(block: str, kind: str, **counts: int) -> Any:
    from marconi.engine.backends.base import BlockCensus

    return BlockCensus(block=block, kind=kind, **counts)


def _tapped_pipeline(terminal: str | None) -> GrPipeline:
    return GrPipeline(
        sample_rate=1.0,
        blocks=[
            GrBlock(id="iq_file_source_0", kind="iq_file_source", params={"path": "x"}),
            GrBlock(id="iq_file_sink_0", kind="iq_file_sink", params={"path": "tap"}),
            GrBlock(id="bits_file_sink_0", kind="bits_file_sink", params={"path": "o"}),
        ],
        terminal_sink=terminal,
    )


def test_empty_terminal_verdict_survives_item_bearing_taps() -> None:
    """Trace/soft taps share the sink block kinds and DO carry items on the
    conditioning stages; only the compiler-marked terminal decides empty."""
    result = RunResult(
        status=RunStatus.OK,
        artifacts=[],
        census=[
            _census("iq_file_source_0", "iq_file_source", items_out=100),
            _census("iq_file_sink_0", "iq_file_sink", items_in=100),
            _census("demod_0", "demod", items_in=100, items_out=0),
            _census("bits_file_sink_0", "bits_file_sink", items_in=0),
        ],
    )
    out = worker_mod._flag_empty_sink(result, _tapped_pipeline("bits_file_sink_0"))
    assert out.status == "empty"
    assert out.stalled_at == "demod_0"


def test_empty_aggregate_rule_holds_for_unmarked_ir() -> None:
    # hand-built dev IR never marks a terminal: the all-sinks rule remains,
    # so an item-bearing sink keeps the run ok
    result = RunResult(
        status=RunStatus.OK,
        artifacts=[],
        census=[
            _census("iq_file_sink_0", "iq_file_sink", items_in=100),
            _census("bits_file_sink_0", "bits_file_sink", items_in=0),
        ],
    )
    out = worker_mod._flag_empty_sink(result, _tapped_pipeline(None))
    assert out.status == "ok"


def test_scheduler_abort_matches_thread_body_wrapper_messages() -> None:
    """The OTHER half of the detector: an exception escaping a C++ block's
    work() is logged by the thread_body_wrapper logger, not block_executor,
    and GR 3.10.12's catch-all text is "caught unrecognized exception"
    ("unhandled" appears nowhere in the shipped binary - verified via
    strings(1) on libgnuradio-runtime). Both shapes verbatim from a live
    file_sink failure; a run whose block thread died reported status ok
    with a silently truncated stream."""
    from marconi.engine.backends.base import RunResult
    from marconi.engine.backends.gnuradio.worker import _flag_scheduler_abort
    from marconi.engine.types.enums import RunStatus

    ok = RunResult(status=RunStatus.OK)
    real = (
        "thread_body_wrapper :error: ERROR thread[thread-per-block[3]: "
        "<block file_sink(0)>]: file_sink write failed with error 8\n"
    )
    assert _flag_scheduler_abort(ok, real).status is RunStatus.ERROR
    unrecognized = "ERROR thread[thread-per-block[2]]: caught unrecognized exception\n"
    assert _flag_scheduler_abort(ok, unrecognized).status is RunStatus.ERROR
    benign = (
        "vmcircbuf :info: sysv_shm is not available\n"
        "thread_body_wrapper :info: starting 4 threads\n"
    )
    assert _flag_scheduler_abort(ok, benign).status is RunStatus.OK


def test_trampoline_wraps_hooks_inherited_through_the_mro() -> None:
    """vars(cls) only sees the exact class: a work() inherited from a base
    class was left unwrapped, and its raise re-opened the very hang the
    trampoline exists to close - silently, since the miss had no diagnostic."""
    from marconi.engine.backends.gnuradio.worker import _arm_crash_trampoline

    class _Base:
        def work(self, input_items: object, output_items: object) -> int:
            raise RuntimeError("boom from an inherited hook")

    class _Child(_Base):
        pass

    class _Tb:
        def __init__(self) -> None:
            self._py_instances = {"b0": _Child()}
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    tb = _Tb()
    crashes: list[str] = []
    _arm_crash_trampoline(tb, crashes)
    blk = tb._py_instances["b0"]
    assert blk.work(None, None) == -1  # trampolined: recorded, not raised
    assert crashes and "inherited hook" in crashes[0]
    assert tb.stopped
