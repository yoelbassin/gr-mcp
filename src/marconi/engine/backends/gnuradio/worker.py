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

import numpy as np

from marconi.engine.backends.base import (
    BackendError,
    BlockCensus,
    Diagnostic,
    RunResult,
)
from marconi.engine.compile.ir import FILE_SINK_KINDS, GrPipeline
from marconi.engine.types.enums import RunStatus
from marconi.wire import replace

# Two logger channels, verified against GR 3.10.12: buffer/forecast
# complaints arrive via block_executor, but an exception escaping a C++
# block's work() is logged by thread_body_wrapper ("ERROR thread[...]"), and
# the catch-all text is "caught UNRECOGNIZED exception" - "unhandled" appears
# nowhere in the shipped binary (strings(1) on libgnuradio-runtime). The old
# pattern matched only the first channel, so a run whose block thread died
# reported status ok over a silently truncated stream.
_SCHED_ABORT = re.compile(
    r"block_executor\s*:\s*error|thread_body_wrapper\s*:\s*error"
    r"|ERROR thread\[|caught unrecognized exception",
    re.IGNORECASE,
)


def sink_paths(pipeline: GrPipeline) -> list[str]:
    return [
        str(b.params["path"])
        for b in pipeline.blocks
        if b.kind in FILE_SINK_KINDS and "path" in b.params
    ]


# gr::block::WORK_DONE: this block will produce nothing further.
_WORK_DONE = -1


def _trampolined_work(
    fn: Callable[[Any, Any], int], tb: Any, crashes: list[str]
) -> Callable[[Any, Any], int]:
    def wrapped(input_items: Any, output_items: Any) -> int:
        try:
            return int(fn(input_items, output_items))
        except Exception:  # noqa: BLE001
            crashes.append(traceback.format_exc())
            tb.stop()
            return _WORK_DONE

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


def _defines_python_hook(cls: type, name: str) -> bool:
    """Whether `name` resolves to a Python-defined function anywhere in the
    MRO (a pybind builtin is not one and must stay unwrapped). vars(cls)
    alone saw only the exact class, so a hook inherited from a base class
    was left unwrapped and its raise re-opened the very hang the trampoline
    exists to close - silently."""
    for klass in cls.__mro__:
        fn = vars(klass).get(name)
        if fn is not None:
            return inspect.isfunction(fn)
    return False


def _arm_crash_trampoline(tb: Any, crashes: list[str]) -> None:
    """A raise on a GR block thread never unwinds tb.run() — the graph hangs
    until the runner kills the worker (probed). Wrap every Python-defined
    block hook so the exception is recorded and the graph stops instead."""
    for blk in getattr(tb, "_py_instances", {}).values():
        cls = type(blk)
        if _defines_python_hook(cls, "general_work"):
            blk.general_work = _trampolined_work(blk.general_work, tb, crashes)
        if _defines_python_hook(cls, "work"):
            blk.work = _trampolined_work(blk.work, tb, crashes)
        if _defines_python_hook(cls, "forecast"):
            blk.forecast = _trampolined_forecast(blk.forecast, tb, crashes)


def _diagnostic_row(block: str, key: str, v: int | float | list[int]) -> Diagnostic:
    if isinstance(v, list):
        return Diagnostic(block=block, key=key, marks=[int(m) for m in v])
    if isinstance(v, bool):
        raise TypeError(f"diagnostic {block}.{key} is a bool; emit an int count")
    if isinstance(v, (int, np.integer)):
        return Diagnostic(block=block, key=key, count=int(v))
    if isinstance(v, (float, np.floating)):
        return Diagnostic(block=block, key=key, value=float(v))
    raise TypeError(
        f"diagnostic {block}.{key} has unharvestable type {type(v).__name__}"
    )


_UNHARVESTABLE = "diagnostics_unharvestable"


def _harvest_diagnostics(tb: Any) -> list[Diagnostic]:
    """Same rule as _harvest_census: never let a diagnostic fail a run that
    otherwise worked. _diagnostic_row raises on a value it cannot represent,
    and this runs on the ok path AND both error paths — unguarded, one stray
    value type turned a flowgraph that ran to completion into "worker exited
    abnormally" with no stream, census or quality, and masked the real error
    when there was one.

    A row that cannot be built is dropped but COUNTED, per block, under
    _UNHARVESTABLE. Dropping it silently was the other half of the rename trap
    the vocabulary tests cover: a producer whose value type went wrong looked
    exactly like a producer that had nothing to say, and the weakened verdict
    had no visible cause anywhere in the response."""
    rows: list[Diagnostic] = []
    dropped: dict[str, int] = {}
    for bid, blk in getattr(tb, "_py_instances", {}).items():
        diag = getattr(blk, "diagnostics", None)
        if not isinstance(diag, dict):
            continue
        for k, v in diag.items():
            try:
                rows.append(_diagnostic_row(str(bid), str(k), v))
            except (TypeError, ValueError):
                dropped[str(bid)] = dropped.get(str(bid), 0) + 1
    rows.extend(
        Diagnostic(block=bid, key=_UNHARVESTABLE, count=n)
        for bid, n in sorted(dropped.items())
    )
    return rows


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
        return RunResult(status=RunStatus.ERROR, error=str(e))
    crashes: list[str] = []
    _arm_crash_trampoline(tb, crashes)
    try:
        tb.run()
    except Exception as e:  # noqa: BLE001
        return RunResult(
            status=RunStatus.ERROR,
            error=f"flowgraph raised: {type(e).__name__}: {e}",
            artifacts=sink_paths(pipeline),
            diagnostics=_harvest_diagnostics(tb),
            census=_harvest_census(tb, pipeline),
        )
    if crashes:
        return RunResult(
            status=RunStatus.ERROR,
            error="embedded block raised:\n" + "\n".join(crashes),
            artifacts=sink_paths(pipeline),
            diagnostics=_harvest_diagnostics(tb),
            census=_harvest_census(tb, pipeline),
        )
    return RunResult(
        status=RunStatus.OK,
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
    if result.status is not RunStatus.OK:
        return result
    hits = [ln.strip() for ln in captured.splitlines() if _SCHED_ABORT.search(ln)]
    if not hits:
        return result
    return RunResult(
        status=RunStatus.ERROR,
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
    if result.status is not RunStatus.OK or not result.census:
        return result
    # taps share the sink kinds and DO carry items on conditioning stages —
    # only the compiler-marked terminal decides; the aggregate rule covers
    # hand-built dev IR that never marks one
    if pipeline.terminal_sink is not None:
        sink_ids = {pipeline.terminal_sink}
    else:
        sink_ids = {b.id for b in pipeline.blocks if b.kind in FILE_SINK_KINDS}
    sinks = [c for c in result.census if c.block in sink_ids]
    if not sinks or any(c.items_in is None or c.items_in > 0 for c in sinks):
        return result
    stall = next((c for c in result.census if c.items_out == 0), None)
    where = f" (no items past {stall.kind})" if stall else ""
    return replace(
        result,
        status=RunStatus.EMPTY,
        stalled_at=stall.block if stall else None,
        error=f"terminal sink wrote 0 items{where}",
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
