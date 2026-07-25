from __future__ import annotations

import inspect
import os
import re
import sys
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from marconi.engine.backends.base import BackendError, BlockCensus, RunResult
from marconi.engine.compile.ir import GrPipeline

_SINK_KINDS = {
    "iq_file_sink",
    "bits_file_sink",
    "soft_bits_file_sink",
    "symbols_file_sink",
}

_SCHED_ABORT = re.compile(
    r"block_executor.*error|caught unhandled exception", re.IGNORECASE
)


def sink_paths(pipeline: GrPipeline) -> list[str]:
    return [
        str(b.params["path"])
        for b in pipeline.blocks
        if b.kind in _SINK_KINDS and "path" in b.params
    ]


def _trampolined_work(
    fn: Callable[[Any, Any], int], tb: Any, crashes: list[str]
) -> Callable[[Any, Any], int]:
    def wrapped(input_items: Any, output_items: Any) -> int:
        try:
            return int(fn(input_items, output_items))
        except Exception:  # noqa: BLE001
            crashes.append(traceback.format_exc())
            tb.stop()
            return -1  # WORK_DONE

    return wrapped


def _trampolined_forecast(
    fn: Callable[[int, int], list[int]], tb: Any, crashes: list[str]
) -> Callable[[int, int], list[int]]:
    def wrapped(noutput_items: int, ninputs: int) -> list[int]:
        try:
            return fn(noutput_items, ninputs)
        except Exception:  # noqa: BLE001
            crashes.append(traceback.format_exc())
            tb.stop()
            return [0] * ninputs

    return wrapped


def _arm_crash_trampoline(tb: Any, crashes: list[str]) -> None:
    """A raise on a GR block thread never unwinds tb.run() — the graph hangs
    until the runner kills the worker (probed). Wrap every Python-defined
    block hook so the exception is recorded and the graph stops instead."""
    for blk in getattr(tb, "_py_instances", {}).values():
        cls = type(blk)
        if inspect.isfunction(vars(cls).get("general_work")):
            blk.general_work = _trampolined_work(blk.general_work, tb, crashes)
        if inspect.isfunction(vars(cls).get("work")):
            blk.work = _trampolined_work(blk.work, tb, crashes)
        if inspect.isfunction(vars(cls).get("forecast")):
            blk.forecast = _trampolined_forecast(blk.forecast, tb, crashes)


def _harvest_diagnostics(tb: Any) -> dict[str, dict[str, int | list[int]]]:
    out: dict[str, dict[str, int | list[int]]] = {}
    for bid, blk in getattr(tb, "_py_instances", {}).items():
        diag = getattr(blk, "diagnostics", None)
        if isinstance(diag, dict) and diag:
            out[str(bid)] = {
                str(k): [int(m) for m in v] if isinstance(v, list) else int(v)
                for k, v in diag.items()
            }
    return out


def _port_count(blk: Any, counter: str) -> int | None:
    """None where the count is unavailable: an absent port raises, and some
    blocks (pfb_arb_resampler) do not expose the counter at all."""
    try:
        return int(getattr(blk, counter)(0))
    except Exception:  # noqa: BLE001
        return None


def _harvest_census(tb: Any, pipeline: GrPipeline) -> list[BlockCensus]:
    """GR keeps exact per-block item counters and they outlive the run, so the
    whole chain can be measured without inserting a single probe. Never let a
    diagnostic fail a run that otherwise worked."""
    try:
        instances = getattr(tb, "_py_instances", {})
        return [
            BlockCensus(
                block=b.id,
                kind=b.kind,
                items_in=_port_count(instances[b.id], "nitems_read"),
                items_out=_port_count(instances[b.id], "nitems_written"),
            )
            for b in pipeline.blocks
            if b.id in instances
        ]
    except Exception:  # noqa: BLE001
        return []


def _run_flowgraph(pipeline: GrPipeline) -> RunResult:
    from marconi.engine.backends.gnuradio.build import build_top_block  # lazy

    try:
        tb = build_top_block(pipeline)
    except BackendError as e:
        return RunResult(status="error", error=str(e))
    crashes: list[str] = []
    _arm_crash_trampoline(tb, crashes)
    try:
        tb.run()
    except Exception as e:  # noqa: BLE001
        return RunResult(
            status="error",
            error=f"flowgraph raised: {e}",
            artifacts=sink_paths(pipeline),
            diagnostics=_harvest_diagnostics(tb),
            census=_harvest_census(tb, pipeline),
        )
    if crashes:
        return RunResult(
            status="error",
            error="embedded block raised:\n" + "\n".join(crashes),
            artifacts=sink_paths(pipeline),
            diagnostics=_harvest_diagnostics(tb),
            census=_harvest_census(tb, pipeline),
        )
    return RunResult(
        status="ok",
        artifacts=sink_paths(pipeline),
        diagnostics=_harvest_diagnostics(tb),
        census=_harvest_census(tb, pipeline),
    )


def _captured_text(capture_path: str | None) -> str:
    if capture_path is None:
        return ""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    try:
        return Path(capture_path).read_text(errors="replace")
    except OSError:
        return ""


def _flag_scheduler_abort(result: RunResult, captured: str) -> RunResult:
    """A scheduler abort (e.g. a block demanding more items than a stream
    buffer holds) lets tb.run() return normally with an empty sink; the only
    trace is the block_executor message on the worker's stdio. Best-effort by
    measurement, not contract: under load GR sometimes never prints the
    message at all (probed 2026-07: 9/20 immediate-exit runs, 2/20 even after
    a 1s settle), so ok + an empty sink is not proven-clean."""
    if result.status != "ok":
        return result
    hits = [ln.strip() for ln in captured.splitlines() if _SCHED_ABORT.search(ln)]
    if not hits:
        return result
    return RunResult(
        status="error",
        error="scheduler abort: " + "\n".join(hits),
        artifacts=result.artifacts,
        diagnostics=result.diagnostics,
        census=result.census,
    )


def _flag_empty_sink(result: RunResult, pipeline: GrPipeline) -> RunResult:
    """A graph that reached EOF but wrote nothing to its terminal sink decoded
    nothing; 'ok' would be a lie. Report 'empty' and name the first stage whose
    items_out fell to zero — the census gradient a parameter search needs. Runs
    only after the error/abort checks, so a real failure still outranks it and a
    missing census never fabricates an empty verdict."""
    if result.status != "ok" or not result.census:
        return result
    sink_ids = {b.id for b in pipeline.blocks if b.kind in _SINK_KINDS}
    sinks = [c for c in result.census if c.block in sink_ids]
    if not sinks or any(c.items_in is None or c.items_in > 0 for c in sinks):
        return result
    stall = next((c for c in result.census if c.items_out == 0), None)
    where = f" (no items past {stall.kind})" if stall else ""
    return result.model_copy(
        update={
            "status": "empty",
            "stalled_at": stall.block if stall else None,
            "error": f"terminal sink wrote 0 items{where}",
        }
    )


def run_pipeline_worker(
    payload_json: str, conn: Connection, capture_path: str | None = None
) -> None:
    if capture_path is not None:
        fd = os.open(capture_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
    pipeline = GrPipeline.model_validate_json(payload_json)
    result = _run_flowgraph(pipeline)
    result = _flag_scheduler_abort(result, _captured_text(capture_path))
    result = _flag_empty_sink(result, pipeline)
    conn.send(result.model_dump_json())
    conn.close()
