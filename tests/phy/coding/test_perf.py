from __future__ import annotations

import numpy as np
import pytest
from crc import Calculator, Configuration
from crc._crc import TableBasedRegister
from helpers import framing
from helpers.crc import _calculator, crc_value

from marconi.core import bitfile
from marconi.core.bitfile import CaptureTooLarge, read_bits
from marconi.phy.coding.carrier import CodingCarrier
from marconi.phy.coding.ops_bits import block_code_rx

_HDLC = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)


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


class _CallCounter:
    """Wraps a callable and counts invocations -- an in-process control that
    proves vectorization without touching the clock: a wall-clock ratio
    measures the machine (flaky under xdist contention), but the number of
    numpy primitive calls a vectorized op makes does not grow with input
    size, while a per-item python loop's would. Same size in, same call
    count out is the signature of "no per-item loop"."""

    def __init__(self, fn):
        self._fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._fn(*args, **kwargs)


def test_crc_value_uses_the_table_engine_not_the_bitwise_one() -> None:
    """Not a wall-clock ratio: a structural check on the crc library's own
    register class. optimized=True selects TableBasedRegister; optimized=False
    (the bit-by-bit engine) selects plain Register -- this is the actual
    implementation choice the old timing ratio was a (flaky) proxy for."""
    data = bytes(1 << 12)
    config = Configuration(
        width=16,
        polynomial=0x1021,
        init_value=0xFFFF,
        final_xor_value=0,
        reverse_input=False,
        reverse_output=False,
    )
    bitwise = Calculator(config, optimized=False)
    assert crc_value(data, poly=0x1021, bits=16, init=0xFFFF) == bitwise.checksum(data)

    calc = _calculator(0x1021, 16, 0xFFFF, False, 0)
    assert isinstance(
        calc._crc_register, TableBasedRegister
    ), "crc.Calculator not constructed optimized=True -- bitwise engine in effect?"


def test_flag_scan_makes_input_size_independent_numpy_calls(monkeypatch) -> None:
    counter = _CallCounter(np.cumsum)
    monkeypatch.setattr(np, "cumsum", counter)

    framing._find_flags(np.random.default_rng(0).integers(0, 2, 200, dtype=np.uint8))
    small = counter.calls
    counter.calls = 0

    framing._find_flags(
        np.random.default_rng(0).integers(0, 2, 200_000, dtype=np.uint8)
    )
    large = counter.calls

    assert small == large == 1, (
        f"{small} vs {large} cumsum calls across a 1000x size gap -- a per-bit "
        "python loop would scale with input size"
    )


def test_block_code_correction_makes_input_size_independent_numpy_calls(
    monkeypatch,
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
        correct=True,
        emit="data",
    )
    small = counter.calls
    counter.calls = 0

    large_bits = np.random.default_rng(4).integers(0, 2, 8_000_000, dtype=np.uint8)
    block_code_rx(
        CodingCarrier(bits=large_bits),
        code_bits=8,
        data_bits=4,
        parity_masks=masks,
        correct=True,
        emit="data",
    )
    large = counter.calls

    assert small == large == 1, (
        f"{small} vs {large} flatnonzero calls across a 1000x size gap -- a "
        "per-word python correction loop would scale with word count"
    )


def test_find_flags_matches_greedy_reference():
    rng = np.random.default_rng(1)
    for _ in range(20):
        bits = rng.integers(0, 2, int(rng.integers(8, 200)), dtype=np.uint8)
        assert framing._find_flags(bits) == _reference_find_flags(bits)
    shared = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], np.uint8)
    assert framing._find_flags(shared) == _reference_find_flags(shared) == [0, 7]
    assert framing._find_flags(np.zeros(4, np.uint8)) == []


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
