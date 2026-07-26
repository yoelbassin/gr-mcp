"""Golay(23,12) built from its generator polynomial — shared caller-side
test math for the block_code t=3 unit tests and the e2e Golay gate."""

from __future__ import annotations

import numpy as np

G = 0b110001110101  # x^11+x^10+x^6+x^5+x^4+x^2+1


def poly_mod(v: int) -> int:
    for shift in range(v.bit_length() - 1, 10, -1):
        if (v >> shift) & 1:
            v ^= G << (shift - 11)
    return v


def golay_masks() -> list[int]:
    unit_parity = [poly_mod(1 << (11 + d)) for d in range(12)]
    return [sum(((unit_parity[d] >> p) & 1) << d for d in range(12)) for p in range(11)]


def golay_codeword(data: int) -> np.ndarray:
    parity = poly_mod(data << 11)
    bits = np.empty(23, np.uint8)
    for j in range(12):
        bits[j] = (data >> j) & 1
    for p in range(11):
        bits[12 + p] = (parity >> p) & 1
    return bits
