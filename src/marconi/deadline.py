from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

from marconi.errors import register_error

_T = TypeVar("_T")


class RunTimeout(Exception):
    pass


_deadline: ContextVar[float | None] = ContextVar("marconi_run_deadline", default=None)


@contextmanager
def set_deadline(timeout: float) -> Iterator[None]:
    new = time.monotonic() + max(0.0, timeout)
    existing = _deadline.get()
    dl = new if existing is None else min(existing, new)
    token = _deadline.set(dl)
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


def bounded(items: Iterable[_T]) -> Iterator[_T]:
    """Iterate with the run's deadline checked once per item.

    The obligation is the same either way — every loop that walks a capture has
    to poll — but a hand-placed check is one an author has to remember, and
    four consecutive reviews found a fresh sibling loop that had not. Streaming
    reads go through this; tests/unit/test_deadline_coverage.py fails a loop
    that reads a file without one."""
    for item in items:
        check_deadline()
        yield item


def remaining() -> float:
    dl = _deadline.get()
    return float("inf") if dl is None else max(0.0, dl - time.monotonic())


register_error(RunTimeout, "deadline_exceeded")
