"""Pure-python stand-ins for the gr module: run embedded blocks' state
machines without a scheduler so call, chunk, and output-window timing can
be forced exactly (the adversarial schedules GR is allowed to produce)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np


class FakeTag:
    def __init__(self, offset: int, key: str = "burst", value: Any = None) -> None:
        self.offset = int(offset)
        self.key = key
        self.value = value


class _FakePmt:
    PMT_NIL = None

    @staticmethod
    def intern(s: str) -> str:
        return s

    @staticmethod
    def symbol_to_string(k: Any) -> str:
        return str(k)

    @staticmethod
    def to_double(v: Any) -> float:
        return float(v)


class _BasicBlock:
    def __init__(self, name: str = "", in_sig: Any = None, out_sig: Any = None):
        self._nread = 0
        self._nwritten = 0
        self.in_tags: list[FakeTag] = []
        self.out_tags: list[FakeTag] = []

    def consume(self, port: int, n: int) -> None:
        self._nread += int(n)

    def nitems_read(self, port: int) -> int:
        return self._nread

    def nitems_written(self, port: int) -> int:
        return self._nwritten

    def set_tag_propagation_policy(self, policy: Any) -> None:
        pass

    def get_tags_in_window(self, port: int, a: int, b: int) -> list[FakeTag]:
        lo, hi = self._nread + a, self._nread + b
        return [t for t in self.in_tags if lo <= t.offset < hi]

    def add_item_tag(self, port: int, offset: int, key: Any, value: Any = None) -> None:
        self.out_tags.append(FakeTag(int(offset), str(key), value))


FAKE_GR = SimpleNamespace(basic_block=_BasicBlock, pmt=_FakePmt, TPP_DONT=0)


def drive(
    blk: Any,
    sig: np.ndarray,
    chunk: int,
    out_len: int = 1 << 16,
    out_dtype: Any = None,
) -> np.ndarray:
    """Feed sig in `chunk`-sized pieces; after every piece, keep calling with
    ZERO input until quiescent — the wakeups the scheduler is allowed to make
    whenever forecast announces 0. Assumes the block consumes each offered
    piece (all-consuming blocks); the caller tracks nitems_written."""
    sig = np.asarray(sig)
    dtype = out_dtype if out_dtype is not None else sig.dtype
    got = []
    for start in range(0, len(sig), chunk):
        nxt = sig[start : start + chunk]
        while True:
            out = np.zeros(out_len, dtype)
            k = int(blk.general_work([nxt], [out]))
            blk._nwritten += k
            if k:
                got.append(out[:k].copy())
            nxt = sig[0:0]
            if k == 0:
                break
    return np.concatenate(got) if got else np.zeros(0, dtype)
