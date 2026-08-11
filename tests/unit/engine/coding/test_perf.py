from __future__ import annotations

import tracemalloc
from collections.abc import Callable

import numpy as np
import pytest

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.ops_bits import block_code_rx, bytes_to_bits, sync_word_rx
from marconi.engine.coding.ops_symbols import sync_symbols_rx
from marconi.engine.types.enums import EmitMode


def _measure_peak(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


class _CallCounter:
    """Wraps a callable and counts invocations -- an in-process control that
    proves vectorization without touching the clock: a wall-clock ratio
    measures the machine (flaky under xdist contention), but the number of
    numpy primitive calls a vectorized op makes does not grow with input
    size, while a per-item python loop's would. Same size in, same call
    count out is the signature of "no per-item loop"."""

    def __init__(self, fn: Callable[..., object]) -> None:
        self._fn = fn
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return self._fn(*args, **kwargs)


def test_block_code_correction_makes_input_size_independent_numpy_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = _CallCounter(np.flatnonzero)
    monkeypatch.setattr(np, "flatnonzero", counter)
    masks = [7, 14, 11, 13]

    small_bits = np.random.default_rng(4).integers(0, 2, 8_000, dtype=np.uint8)
    block_code_rx(
        CodingCarrier(bits=small_bits),
        code_bits=8,
        data_bits=4,
        parity_masks=masks,
        correct_single=True,
        emit=EmitMode.DATA,
    )
    small = counter.calls
    counter.calls = 0

    large_bits = np.random.default_rng(4).integers(0, 2, 8_000_000, dtype=np.uint8)
    block_code_rx(
        CodingCarrier(bits=large_bits),
        code_bits=8,
        data_bits=4,
        parity_masks=masks,
        correct_single=True,
        emit=EmitMode.DATA,
    )
    large = counter.calls

    assert small == large == 1, (
        f"{small} vs {large} flatnonzero calls across a 1000x size gap -- a "
        "per-word python correction loop would scale with word count"
    )


def _peak_bytes(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_sync_word_scan_memory_is_linear_in_input_not_pattern() -> None:
    """The correlating scan must never materialize an (n, m) array: with the
    bits-layer budget (2^30 bits) and a 32-bit sync, n x m is ~40 GiB inside a
    layer that promises a ~2-3x transient (core/bitfile.py)."""
    n = 1 << 22
    bits = np.random.default_rng(2).integers(0, 2, n, dtype=np.uint8)
    c = CodingCarrier(bits=bits)
    peak = _peak_bytes(lambda: sync_word_rx(c, sync="deadbeefcafef00d", max_errors=1))
    assert peak < 8 * n, (
        f"peak {peak >> 20} MiB scanning a {n >> 20} MiB unpacked bit array "
        f"with a 64-bit sync -- the scan is materializing n x m"
    )


def test_sync_symbols_scan_memory_is_linear_in_input_not_pattern() -> None:
    n = 1 << 20
    sym = np.random.default_rng(3).normal(0, 1, n).astype(np.float32)
    pattern = [1, -1] * 28 + [0] * 8
    c = CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym)
    peak = _peak_bytes(
        lambda: sync_symbols_rx(c, pattern=pattern, max_errors=2, pre_symbols=0)
    )
    assert peak < 32 * n, (
        f"peak {peak >> 20} MiB scanning {n >> 20} Msymbols with a 64-symbol "
        f"pattern -- the scan is materializing n x m"
    )


def _reference_sync_windows(
    bits: np.ndarray, pat: np.ndarray, max_errors: int
) -> list[int]:
    wins: list[int] = []
    reach = 0
    for i in range(bits.size - pat.size + 1):
        if i < reach:
            continue
        if int((bits[i : i + pat.size] != pat).sum()) <= max_errors:
            wins.append(i + pat.size)
            reach = i + pat.size
    return wins


def test_sync_word_scan_matches_naive_reference() -> None:
    rng = np.random.default_rng(5)
    for sync, max_errors in [("a7", 0), ("7e7e", 1), ("deadbeef", 2)]:
        pat = bytes_to_bits(bytes.fromhex(sync))
        for _ in range(10):
            bits = rng.integers(0, 2, int(rng.integers(8, 500)), dtype=np.uint8)
            out = sync_word_rx(
                CodingCarrier(bits=bits), sync=sync, max_errors=max_errors
            )
            want = _reference_sync_windows(bits, pat, max_errors)
            assert [w.start for w in out.windows or []] == want


def test_unpack_pack_roundtrip_and_budget() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 4 << 20).astype(np.uint8)
    vals = ops_bits._unpack_symbols(bits, 8)
    expect = np.asarray(bits, np.int64).reshape(-1, 8) @ (
        1 << np.arange(7, -1, -1, dtype=np.int64)
    )
    assert np.array_equal(vals, expect)
    assert np.array_equal(ops_bits._pack_symbols(vals, 8), bits)
    peak = _measure_peak(lambda: ops_bits._unpack_symbols(bits, 8))
    assert peak < 4 * bits.nbytes, f"peak {peak / bits.nbytes:.1f}x input"


def test_codebook_windowless_budget() -> None:
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 6 << 20).astype(np.uint8)
    carrier = CodingCarrier(bits=bits)
    peak = _measure_peak(
        lambda: ops_bits.codebook_rx(
            carrier, code_bits=6, data_bits=4, table=list(range(16))
        )
    )
    assert peak < 5 * bits.nbytes, f"peak {peak / bits.nbytes:.1f}x input"
