import tracemalloc
from collections.abc import Callable

import numpy as np

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier


def _measure_peak(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


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
