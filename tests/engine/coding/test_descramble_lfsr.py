from __future__ import annotations

import numpy as np
import pytest

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier, Window
from marconi.engine.stages.registry import stage_registry


def _ref_lfsr(mask: int, seed: int, length: int, n: int) -> np.ndarray:
    out, s = np.empty(n, np.uint8), seed
    for i in range(n):
        out[i] = s & 1
        fb = bin(s & mask).count("1") & 1
        s = (s >> 1) | (fb << (length - 1))
    return out


# x^7 + x^6 + 1 (primitive): taps on the two oldest state bits.
_MASK, _SEED, _LEN = 0b0000011, 0x7F, 7


def test_lfsr_descramble_matches_reference() -> None:
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 500).astype(np.uint8)
    out = ops_bits.descramble_rx(
        CodingCarrier(bits=bits), lfsr_mask=_MASK, lfsr_seed=_SEED, lfsr_len=_LEN
    )
    expect = bits ^ _ref_lfsr(_MASK, _SEED, _LEN, 500)
    assert out.bits.tolist() == expect.tolist()


def test_lfsr_is_an_m_sequence() -> None:
    zeros = np.zeros(2 * 127, np.uint8)
    seq = ops_bits.descramble_rx(
        CodingCarrier(bits=zeros), lfsr_mask=_MASK, lfsr_seed=_SEED, lfsr_len=_LEN
    ).bits
    assert seq[:127].sum() == 64  # balanced
    assert seq[:127].tolist() == seq[127:254].tolist()  # period 2^7 - 1
    assert seq[:126].tolist() != seq[1:127].tolist()  # not degenerate


def test_lfsr_resets_at_every_window() -> None:
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 2, 60).astype(np.uint8)
    stream = np.concatenate([frame, np.ones(13, np.uint8), frame])
    windows = [Window(start=0, cursor=0), Window(start=73, cursor=73)]
    out = ops_bits.descramble_rx(
        CodingCarrier(bits=stream, windows=windows),
        lfsr_mask=_MASK,
        lfsr_seed=_SEED,
        lfsr_len=_LEN,
    )
    a, b = out.bits[0:60], out.bits[73:133]
    assert a.tolist() == b.tolist()  # same whitening from the seed each window


def test_sequence_mode_is_unchanged() -> None:
    bits = np.ones(16, np.uint8)
    out = ops_bits.descramble_rx(CodingCarrier(bits=bits), sequence="ff00")
    assert out.bits.tolist() == [0] * 8 + [1] * 8


@pytest.mark.parametrize(
    "params",
    [
        {},  # neither mode
        {"sequence": "ff", "lfsr_mask": 3, "lfsr_seed": 1, "lfsr_len": 7},  # both
        {"lfsr_mask": 3},  # partial lfsr triple
        {"lfsr_mask": 0, "lfsr_seed": 1, "lfsr_len": 7},  # zero mask
        {"lfsr_mask": 3, "lfsr_seed": 0, "lfsr_len": 7},  # zero seed
        {"lfsr_mask": 3, "lfsr_seed": 1 << 8, "lfsr_len": 7},  # seed too wide
    ],
)
def test_descramble_param_modes_are_exclusive(params: dict[str, object]) -> None:
    with pytest.raises(Exception):
        stage_registry()["descramble"].params_model.model_validate(params)
