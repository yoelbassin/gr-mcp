import time

import numpy as np
import pytest

from marconi.bits.framing import _find_flags, crc_value
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
def test_crc_throughput_uses_table_engine():
    data = bytes(1 << 19)  # 512 KiB
    secs = _best_of(lambda: crc_value(data, poly=0x1021, bits=16, init=0xFFFF))
    mb_per_s = (len(data) / 1e6) / secs
    # bit-by-bit engine ~0.30 MB/s, table engine ~2.1; 1.0 cleanly separates them.
    assert mb_per_s >= 1.0, f"{mb_per_s:.2f} MB/s -- optimized=True not in effect?"


@pytest.mark.perf
def test_flag_scan_throughput():
    bits = np.random.default_rng(0).integers(0, 2, 10_000_000, dtype=np.uint8)
    secs = _best_of(lambda: _find_flags(bits))
    mbit_per_s = (bits.size / 1e6) / secs
    assert mbit_per_s >= 100.0, f"{mbit_per_s:.0f} Mbit/s -- flag scan not vectorized?"


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
