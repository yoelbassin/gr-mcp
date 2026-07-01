from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterable
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any

from marconi.phy.backends.base import Backend, BackendError, RunResult
from marconi.phy.backends.gnuradio.build import build_top_block
from marconi.phy.backends.gnuradio.worker import (
    run_pipeline_worker,
    sink_paths,
)
from marconi.phy.ir import GrPipeline

_GRACE_SECONDS = 5.0
_PRELOAD = ["marconi.phy.backends.gnuradio._preload"]


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


def _remove(paths: Iterable[str]) -> None:
    for p in paths:
        try:
            Path(p).unlink()
        except OSError:
            pass


def _run_in_subprocess(
    payload_json: str, timeout: float, cleanup: Iterable[str]
) -> RunResult:
    recv, send = _CTX.Pipe(duplex=False)
    proc = _CTX.Process(  # type: ignore[attr-defined]
        target=run_pipeline_worker, args=(payload_json, send)
    )
    proc.start()
    send.close()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(_GRACE_SECONDS)
        if proc.is_alive():
            proc.kill()
            proc.join()
        recv.close()
        _remove(cleanup)
        return RunResult(status="timeout", error="run exceeded timeout; worker killed")
    payload: str | None = None
    try:
        if recv.poll():
            payload = recv.recv()
    except EOFError:
        payload = None
    finally:
        recv.close()
    if payload is not None:
        return RunResult.model_validate_json(payload)
    _remove(cleanup)
    raise BackendError(f"backend worker exited abnormally (exitcode={proc.exitcode})")


class GnuRadioBackend(Backend):
    @property
    def name(self) -> str:
        return "gnuradio-3.10"

    def instantiate(self, pipeline: GrPipeline) -> Any:
        return build_top_block(pipeline)

    def run_pipeline(self, pipeline: GrPipeline, timeout: float = 30.0) -> RunResult:
        return _run_in_subprocess(
            pipeline.model_dump_json(), timeout, sink_paths(pipeline)
        )
