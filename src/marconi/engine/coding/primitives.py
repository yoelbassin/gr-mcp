"""Bit primitives the coding ops consume: Gray codes and the single-error
Hamming-bound test. Everything protocol-shaped (parity-matrix decode, CSS
explicit-header algebra) is caller/test-side, not here."""

from __future__ import annotations


def gray_encode(x: int) -> int:
    return x ^ (x >> 1)


def gray_decode(g: int) -> int:
    x = g
    while g := g >> 1:
        x ^= g
    return x


def can_correct(cr: int, data_bits: int) -> bool:
    """Whether ``cr`` parity bits can locate a single error among the
    ``data_bits + cr`` codeword positions (the Hamming bound). Below it the
    code is detect-only."""
    return (1 << cr) >= data_bits + cr + 1
