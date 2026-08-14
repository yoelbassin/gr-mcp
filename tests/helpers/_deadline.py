"""A run whose deadline expires MID-operation, deterministically.

`set_deadline(0.0)` is spent before the op is entered, so it is answered by
whatever poll the op makes FIRST and can say nothing about the loops after it:
segment_rx, _correlate's hit loop and the per-window decoders each passed that
gate while running seconds — in one case 2.4 s and 708 MB — past the deadline
they had already been handed. Letting a real deadline expire mid-run instead
would price the test on wall-clock and flake on a loaded machine.

So the clock itself is the fixture: `marconi.deadline` reads `time.monotonic`
through its own module global, and a counting stand-in that jumps forward after
N reads makes "the k-th poll is the one that fires" an exact statement. No
sleeps, no wall-clock arithmetic, nothing shared between workers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest

import marconi.deadline
from marconi.deadline import set_deadline


@dataclass
class PollClock:
    reads_before_jump: int
    reads: int = 0
    jump_seconds: float = 3600.0

    def monotonic(self) -> float:
        self.reads += 1
        return 0.0 if self.reads <= self.reads_before_jump else self.jump_seconds


@contextmanager
def deadline_expiring_after(
    monkeypatch: pytest.MonkeyPatch, polls: int
) -> Iterator[PollClock]:
    """A deadline that survives exactly `polls` check_deadline() calls.

    Poll number `polls + 1` — and every one after it — raises RunTimeout, so a
    loop with no poll in it never raises at all.
    """
    clock = PollClock(polls + 1)  # +1: set_deadline reads the clock too
    monkeypatch.setattr(marconi.deadline, "time", clock)
    with set_deadline(1.0):
        yield clock
