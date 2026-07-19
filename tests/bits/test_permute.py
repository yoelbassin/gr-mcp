from __future__ import annotations

import numpy as np
import pytest

from marconi.bits.builder import ProgramBuilder
from marconi.bits.carriers import RxCarrier, _Frame
from marconi.bits.framing import permute_rx
from marconi.bits.registry import registry


def test_permute_gathers_per_block() -> None:
    perm = [2, 0, 3, 1]
    src = np.array([1, 0, 0, 1, 0, 1, 1, 0], np.uint8)
    out = permute_rx(RxCarrier(bits=src), perm=perm).bits
    assert list(out) == [
        src[2],
        src[0],
        src[3],
        src[1],
        src[4 + 2],
        src[4 + 0],
        src[4 + 3],
        src[4 + 1],
    ]


def test_permute_is_invertible_with_inverse_perm() -> None:
    perm = [3, 1, 0, 2]
    inv = [perm.index(i) for i in range(len(perm))]
    src = np.random.default_rng(0).integers(0, 2, 4 * 5, dtype=np.uint8)
    once = permute_rx(RxCarrier(bits=src), perm=perm).bits
    back = permute_rx(RxCarrier(bits=once), perm=inv).bits
    assert np.array_equal(back, src)


def test_permute_before_seeder_guard() -> None:
    with pytest.raises(ValueError):
        permute_rx(
            RxCarrier(bits=np.zeros(4, np.uint8), frames=[_Frame(start=0, cursor=0)]),
            perm=[0, 1, 2, 3],
        )


def test_permute_stage_registered_and_wired() -> None:
    reg = registry()
    assert "permute" in reg
    b = ProgramBuilder()
    reg["permute"].emit_rx(b, {"perm": [2, 0, 3, 1]})
    out = b.steps[0](RxCarrier(bits=np.array([1, 0, 0, 1], np.uint8))).bits
    assert list(out) == [0, 1, 1, 0]  # in[2], in[0], in[3], in[1]
