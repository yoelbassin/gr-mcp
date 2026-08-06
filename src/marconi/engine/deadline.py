from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from marconi.errors import register_error


class RunTimeout(Exception):
    """A run exceeded its wall-clock deadline."""


_deadline: ContextVar[float | None] = ContextVar("marconi_run_deadline", default=None)


@contextmanager
def set_deadline(timeout: float) -> Iterator[None]:
    token = _deadline.set(time.monotonic() + max(0.0, timeout))
    try:
        yield
    finally:
        _deadline.reset(token)


def check_deadline() -> None:
    dl = _deadline.get()
    if dl is not None and time.monotonic() >= dl:
        raise RunTimeout(
            "run exceeded its wall-clock timeout; decode a bounded slice "
            "(capture_offset/capture_samples) or raise the timeout"
        )


def remaining() -> float:
    dl = _deadline.get()
    return float("inf") if dl is None else max(0.0, dl - time.monotonic())


register_error(RunTimeout, "deadline_exceeded")
