"""Generic channel-coding primitives for the CSS family — pure numpy/Python, no
GNU Radio. Gray mapping, the diagonal-interleave permutation, and a parity-matrix
single-error block-FEC decoder. Protocols (LoRa) supply parameters; the math is generic.
"""

from __future__ import annotations

from math import ceil

_DATA_BITS = 4

# LoRa Hamming parity-check rows by code rate (1→CR4/5 … 4→CR4/8). CR4/5 and 4/6
# carry too few parity bits to locate an error (detect-only); CR4/7 and 4/8 correct.
HAMMING_PARITY: dict[int, list[list[int]]] = {
    1: [[1, 1, 1, 1]],
    2: [[1, 1, 1, 0], [0, 1, 1, 1]],
    3: [[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]],
    4: [[1, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],
}
_MIN_PARITY_FOR_CORRECTION = 3


def supported_cr(cr: int) -> bool:
    return cr in HAMMING_PARITY


def gray_encode(x: int) -> int:
    return x ^ (x >> 1)


def gray_decode(g: int) -> int:
    x = g
    while g := g >> 1:
        x ^= g
    return x


def demap_symbols(
    symbols: list[int], sf_app: int, *, n: int, divisor: int, offset: int
) -> list[int]:
    bits: list[int] = []
    for s in symbols:
        g = gray_encode(((s % n) // divisor - offset) % n)
        bits.extend((g >> (sf_app - 1 - j)) & 1 for j in range(sf_app))
    return bits


def diag_deinterleave_perm(sf_app: int, cw_len: int) -> list[int]:
    """Flat gather permutation P with output[i] = input[P[i]] for symbols packed
    MSB-first → codewords packed MSB-first (the LoRa diagonal de-interleaver)."""
    perm: list[int] = []
    for oc in range(sf_app):
        for k in range(cw_len):
            perm.append(
                (cw_len - 1 - k) * sf_app + ((cw_len - 1 - k) - oc - 1) % sf_app
            )
    return perm


def _column_syndrome(parity: list[list[int]], col: int) -> list[int]:
    return [
        (parity[p][col] if col < _DATA_BITS else int(col == _DATA_BITS + p))
        for p in range(len(parity))
    ]


def block_fec_decode(codeword: int, parity: list[list[int]], correct: bool) -> int:
    data = [int(bool(codeword & (1 << i))) for i in range(_DATA_BITS)]
    par = [int(bool(codeword & (1 << (_DATA_BITS + p)))) for p in range(len(parity))]
    syndrome = [
        (sum(d * r for d, r in zip(data, row)) + par[p]) % 2
        for p, row in enumerate(parity)
    ]
    if correct and len(parity) >= _MIN_PARITY_FOR_CORRECTION and any(syndrome):
        total = _DATA_BITS + len(parity)
        for col in range(total):
            if _column_syndrome(parity, col) == syndrome:
                codeword ^= 1 << col
                break
    return codeword & ((1 << _DATA_BITS) - 1)


def bits_to_uint(bits: list[int], start: int, length: int) -> int:
    v = 0
    for k in range(length):
        v = (v << 1) | (int(bits[start + k]) & 1)
    return v


def header_parity_ok(data_int: int, received_parity: int, masks: list[int]) -> bool:
    computed = 0
    n = len(masks)
    for i, mask in enumerate(masks):
        computed |= (bin(mask & data_int).count("1") & 1) << (n - 1 - i)
    return computed == received_parity


def css_explicit_frame_len(
    payload_len: int, has_crc: int, cr: int, sf: int, ldro: int, sf_reduction: int
) -> int:
    denom = sf - sf_reduction * ldro
    blocks = max(ceil((2 * payload_len - sf + 7 + 4 * has_crc) / denom), 0)
    return (cr + 4) * blocks
