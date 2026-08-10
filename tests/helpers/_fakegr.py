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
    def __init__(self, name: str = "", in_sig: Any = None, out_sig: Any = None) -> None:
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


FAKE_GR = SimpleNamespace(
    basic_block=_BasicBlock, sync_block=_BasicBlock, pmt=_FakePmt, TPP_DONT=0
)


def drive(
    blk: Any,
    sig: np.ndarray,
    chunk: int,
    out_len: int = 1 << 16,
    out_dtype: Any = None,
) -> np.ndarray:
    """Feed sig in `chunk`-sized pieces; zero-input wakeups happen only while
    the block's own forecast announces zero demand — the scheduler contract in
    lifecycle.py. Assumes the block consumes each offered piece (all-consuming
    blocks); the caller tracks nitems_written."""
    sig = np.asarray(sig)
    dtype = out_dtype if out_dtype is not None else sig.dtype
    fc = getattr(blk, "forecast", None)
    got = []

    def _call(data: np.ndarray) -> int:
        out = np.zeros(out_len, dtype)
        k = int(blk.general_work([data], [out]))
        blk._nwritten += k
        if k:
            got.append(out[:k].copy())
        return k

    def _drain() -> None:
        for _ in range(1_000_000):
            if fc is not None and fc(out_len, 1) != [0]:
                return
            k = _call(sig[0:0])
            if k == 0:
                if fc is None:
                    return
                raise AssertionError(
                    "block announces zero demand but emits nothing: the EOF "
                    "flush would spin forever under the real scheduler"
                )
        raise AssertionError("drive: drain did not quiesce")

    for start in range(0, len(sig), chunk):
        _call(sig[start : start + chunk])
        _drain()
    return np.concatenate(got) if got else np.zeros(0, dtype)
