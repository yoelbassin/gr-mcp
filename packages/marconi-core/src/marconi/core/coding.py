"""Generic linear block-code primitives — pure bit math, no GR, no DSP. Every
protocol-specific value (parity masks, data width) is caller-supplied."""

from __future__ import annotations


def gray_encode(x: int) -> int:
    return x ^ (x >> 1)


def gray_decode(g: int) -> int:
    x = g
    while g := g >> 1:
        x ^= g
    return x


def parity_rows(masks: list[int], data_bits: int) -> list[list[int]]:
    """Expand integer parity masks to 0/1 rows over ``data_bits`` data bits: bit
    b of mask m is row entry b. Keeps the coding IR flat (list[int]) while the
    decoder works on rows."""
    return [[(m >> b) & 1 for b in range(data_bits)] for m in masks]


def parity_for_cr(masks: list[int], cr: int) -> list[int]:
    """Slice the masks of the code with ``cr`` parity bits out of a flat table
    laid out for increasing cr (cr rows at offset cr*(cr-1)/2 — the systematic
    property that a code with cr parity bits has cr parity rows)."""
    off = cr * (cr - 1) // 2
    return masks[off : off + cr]


def supported_cr(masks: list[int], cr: int) -> bool:
    off = cr * (cr - 1) // 2
    return cr >= 1 and off + cr <= len(masks)


def can_correct(cr: int, data_bits: int) -> bool:
    """Whether ``cr`` parity bits can locate a single error among the
    ``data_bits + cr`` codeword positions (the Hamming bound). Below it the
    code is detect-only."""
    return (1 << cr) >= data_bits + cr + 1


def block_fec_decode(
    codeword: int, parity_masks: list[int], data_bits: int, correct: bool
) -> int:
    parity = parity_rows(parity_masks, data_bits)
    data = [int(bool(codeword & (1 << i))) for i in range(data_bits)]
    par = [int(bool(codeword & (1 << (data_bits + p)))) for p in range(len(parity))]
    syndrome = [
        (sum(d * r for d, r in zip(data, row)) + par[p]) % 2
        for p, row in enumerate(parity)
    ]
    if correct and any(syndrome):
        total = data_bits + len(parity)
        for col in range(total):
            column = [
                (parity[p][col] if col < data_bits else int(col == data_bits + p))
                for p in range(len(parity))
            ]
            if column == syndrome:
                codeword ^= 1 << col
                break
    return codeword & ((1 << data_bits) - 1)


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
