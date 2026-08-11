"""Perf and reference gates for the shared test helpers. These live beside
their subjects (helpers.crc, helpers.framing), not beside the engine: the
call-count assertions stand in for wall-clock ratios, which are unmeasurable
under a saturated xdist run."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from crc import Calculator, Configuration
from crc._crc import TableBasedRegister
from helpers import framing
from helpers.crc import _calculator, crc_value

_HDLC = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)


def _reference_find_flags(bits: npt.NDArray[np.uint8]) -> list[int]:
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

    def __init__(self, fn: Callable[..., object]) -> None:
        self._fn = fn
        self.calls = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
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


def test_flag_scan_makes_input_size_independent_numpy_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_find_flags_matches_greedy_reference() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        bits = rng.integers(0, 2, int(rng.integers(8, 200)), dtype=np.uint8)
        assert framing._find_flags(bits) == _reference_find_flags(bits)
    shared = np.array([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], np.uint8)
    assert framing._find_flags(shared) == _reference_find_flags(shared)
    assert framing._find_flags(np.zeros(4, np.uint8)) == []
