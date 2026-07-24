import time

import numpy as np
import pytest
from crc import Calculator, Configuration

from marconi.bits.carriers import RxCarrier
from marconi.bits.framing import _find_flags, block_code_rx, crc_value
from marconi.core import bitfile
from marconi.core.bitfile import CaptureTooLarge, read_bits

_HDLC = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)


def _best_of(fn, n=3):
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _reference_find_flags(bits):
    out: list[int] = []
    i, n = 0, len(bits)
    while i <= n - 8:
        if np.array_equal(bits[i : i + 8], _HDLC):
            out.append(i)
            i += 7
        else:
            i += 1
    return out


@pytest.mark.perf
def test_crc_uses_the_table_engine_not_the_bitwise_one():
    """Against an in-process control, not the clock: a wall-clock threshold
    measures the machine, so it fails whenever the suite gets busier. Both arms
    feel the same contention, so their ratio does not (measured 6.6-7.2x)."""
    data = bytes(1 << 17)
    config = Configuration(
        width=16,
        polynomial=0x1021,
        init_value=0xFFFF,
        final_xor_value=0,
        reverse_input=False,
        reverse_output=False,
    )
    bitwise = Calculator(config, optimized=False)
    fast = _best_of(lambda: crc_value(data, poly=0x1021, bits=16, init=0xFFFF))
    slow = _best_of(lambda: bitwise.checksum(data))
    assert crc_value(data, poly=0x1021, bits=16, init=0xFFFF) == bitwise.checksum(data)
    assert slow / fast >= 3.0, (
        f"only {slow / fast:.1f}x the bit-by-bit engine -- optimized=True not in "
        "effect?"
    )


@pytest.mark.perf
def test_flag_scan_beats_the_greedy_python_scan():
    """Same ratio discipline: the reference below is the per-bit loop the
    vectorized scan replaced (measured 253-357x)."""
    bits = np.random.default_rng(0).integers(0, 2, 200_000, dtype=np.uint8)
    fast = _best_of(lambda: _find_flags(bits))
    slow = _best_of(lambda: _reference_find_flags(bits), n=1)
    assert (
        slow / fast >= 20.0
    ), f"only {slow / fast:.0f}x the per-bit scan -- flag scan not vectorized?"


@pytest.mark.perf
def test_block_code_throughput_is_vectorized():
    bits = np.random.default_rng(4).integers(0, 2, 1_000_000, dtype=np.uint8)
    secs = _best_of(
        lambda: block_code_rx(
            RxCarrier(bits=bits),
            code_bits=8,
            data_bits=4,
            parity_masks=[7, 14, 11, 13],
        )
    )
    mbit_per_s = (bits.size / 1e6) / secs
    # per-word python loop ~1 Mbit/s, vectorized ~50+; 10 cleanly separates.
    assert mbit_per_s >= 10.0, f"{mbit_per_s:.1f} Mbit/s -- per-word python loop?"


def test_find_flags_matches_greedy_reference():
    rng = np.random.default_rng(1)
    for _ in range(20):
        bits = rng.integers(0, 2, int(rng.integers(8, 200)), dtype=np.uint8)
        assert _find_flags(bits) == _reference_find_flags(bits)
    shared = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], np.uint8)
    assert _find_flags(shared) == _reference_find_flags(shared) == [0, 7]
    assert _find_flags(np.zeros(4, np.uint8)) == []


def test_oversized_capture_raises_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(bitfile, "_MAX_BITS", 64)
    path = tmp_path / "big.bits"
    bitfile.write_bits(path, np.zeros(65, np.uint8))
    with pytest.raises(CaptureTooLarge) as e:
        read_bits(path)
    msg = str(e.value)
    assert "65 bits" in msg and "slice" in msg


def test_within_budget_reads_fine(tmp_path):
    path = tmp_path / "ok.bits"
    bitfile.write_bits(path, np.array([0, 1, 1, 0], np.uint8))
    assert read_bits(path).tolist() == [0, 1, 1, 0]
