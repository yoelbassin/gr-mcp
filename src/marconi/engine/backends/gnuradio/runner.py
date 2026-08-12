from __future__ import annotations

import multiprocessing
import os
import tempfile
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any

from marconi.engine.backends.base import Backend, RunResult
from marconi.engine.backends.gnuradio.build import build_top_block
from marconi.engine.backends.gnuradio.worker import (
    run_pipeline_worker,
    sink_paths,
)
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.types.enums import RunStatus
from marconi.wire import replace

_GRACE_SECONDS = 5.0
_TAIL_CHARS = 4000
_PRELOAD = ["marconi.engine.backends.gnuradio._preload"]


def _make_context() -> BaseContext:
    requested = os.environ.get("MARCONI_PHY_MP_START", "forkserver")
    if requested not in ("forkserver", "spawn"):
        requested = "spawn"
    try:
        ctx = multiprocessing.get_context(requested)
    except ValueError:
        ctx = multiprocessing.get_context("spawn")
    if ctx.get_start_method() == "forkserver":
        ctx.set_forkserver_preload(_PRELOAD)
    return ctx


_CTX = _make_context()
_warmed = False


def _warmup_noop() -> None:
    pass


def ensure_worker_warm() -> None:
    """Start the forkserver daemon once from clean parent state so later runs
    fork a warm gnuradio image instead of paying a fresh import each time."""
    global _warmed
    if _warmed or _CTX.get_start_method() != "forkserver":
        return
    proc = _CTX.Process(target=_warmup_noop)  # type: ignore[attr-defined]
    proc.start()
    proc.join()
    _warmed = True


def _read_tail(path: str) -> str:
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return ""
    return text[-_TAIL_CHARS:].strip()


def _resolve_result(
    timed_out: bool, payload: str | None, exitcode: int | None, tail: str
) -> RunResult:
    """A worker that reported its result keeps it even if it was killed while
    lingering in GR teardown past the deadline — the pipe outranks the clock.
    Caller-named sinks are never deleted; partial output is evidence."""
    if payload is not None:
        return RunResult.model_validate_json(payload)
    suffix = f"\n{tail}" if tail else ""
    if timed_out:
        return RunResult(
            status=RunStatus.TIMEOUT,
            error=f"run exceeded timeout; worker killed{suffix}",
        )
    return RunResult(
        status=RunStatus.ERROR,
        error=f"backend worker exited abnormally (exitcode={exitcode}){suffix}",
    )


def _receive_payload(recv: Connection, timeout: float) -> str | None:
    """Drain the pipe before reaping the worker: a result past the OS pipe
    buffer blocks the worker in send until the parent reads, so join-first
    turns every large successful run into a timeout. A frame truncated by a
    kill mid-send surfaces as OSError, not just EOFError."""
    try:
        if recv.poll(timeout):
            return str(recv.recv())
    except (EOFError, OSError):
        pass
    return None


def _run_in_subprocess(
    payload_json: str,
    timeout: float,
    target: Callable[[str, Connection, str], None] = run_pipeline_worker,
) -> RunResult:
    recv, send = _CTX.Pipe(duplex=False)
    cap_fd, cap_path = tempfile.mkstemp(prefix="marconi-gr-", suffix=".log")
    os.close(cap_fd)
    try:
        proc = _CTX.Process(  # type: ignore[attr-defined]
            target=target, args=(payload_json, send, cap_path)
        )
        deadline = time.monotonic() + timeout
        proc.start()
        send.close()
        payload = _receive_payload(recv, timeout)
        # a delivered result caps the wait at teardown grace: a worker wedged
        # in GR destruction must not hold a finished run to the full deadline
        join_for = (
            _GRACE_SECONDS
            if payload is not None
            else max(0.0, deadline - time.monotonic())
        )
        proc.join(join_for)
        timed_out = proc.is_alive()
        if timed_out:
            proc.terminate()
            proc.join(_GRACE_SECONDS)
            if proc.is_alive():
                proc.kill()
                proc.join()
            if payload is None:
                payload = _receive_payload(recv, 0.0)
        recv.close()
        return _resolve_result(timed_out, payload, proc.exitcode, _read_tail(cap_path))
    finally:
        try:
            os.unlink(cap_path)
        except OSError:
            pass


class GnuRadioBackend(Backend):
    @property
    def name(self) -> str:
        return "gnuradio-3.10"

    def instantiate(self, pipeline: GrPipeline) -> Any:
        return build_top_block(pipeline)

    def run_pipeline(self, pipeline: GrPipeline, timeout: float = 30.0) -> RunResult:
        result = _run_in_subprocess(pipeline.model_dump_json(), timeout)
        if result.status is not RunStatus.OK and not result.artifacts:
            partial = [p for p in sink_paths(pipeline) if Path(p).exists()]
            result = replace(result, artifacts=partial)
        return result
