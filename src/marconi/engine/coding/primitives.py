"""Bit primitives the coding ops consume: Gray codes, the single-error
Hamming-bound test, and the syndrome→flip-mask LUT for t>=2 correction.
Everything protocol-shaped (parity-matrix decode, CSS explicit-header
algebra) is caller/test-side, not here."""

from __future__ import annotations

from functools import lru_cache as _lru_cache
from itertools import combinations as _combinations


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


def _column_signatures(
    parity_masks: list[int], code_bits: int, data_bits: int
) -> list[int]:
    n_parity = code_bits - data_bits
    weights = [1 << (n_parity - 1 - p) for p in range(n_parity)]
    sigs = [
        sum(w for m, w in zip(parity_masks, weights) if (m >> c) & 1)
        for c in range(data_bits)
    ]
    return sigs + weights


@_lru_cache(maxsize=64)
def _syndrome_table_cached(
    parity_masks: tuple[int, ...], code_bits: int, data_bits: int, t: int
) -> dict[int, int]:
    sigs = _column_signatures(list(parity_masks), code_bits, data_bits)
    table: dict[int, int] = {}
    for w in range(1, t + 1):
        for combo in _combinations(range(code_bits), w):
            syn = 0
            for pos in combo:
                syn ^= sigs[pos]
            if syn == 0 or syn in table:
                raise ValueError(
                    f"syndromes collide at weight {w}: the code cannot "
                    f"correct {t} errors"
                )
            table[syn] = sum(1 << pos for pos in combo)
    return table


def syndrome_table(
    parity_masks: list[int], code_bits: int, data_bits: int, t: int
) -> dict[int, int]:
    """syndrome -> codeword-position flip mask (bit j = position j,
    LSB-first) for every error pattern of weight 1..t. Raises ``ValueError``
    on syndrome collision (including a nonzero pattern landing on the zero
    syndrome) — the code cannot correct ``t`` errors. Memoized on
    (parity_masks, code_bits, data_bits, t): a decode call rebuilding the
    same code's table on every word would cost O(2^n_parity) per word. Each
    call returns a fresh copy, never the cached dict itself, so a caller's
    mutation can't corrupt a later caller sharing the same cache key."""
    return dict(_syndrome_table_cached(tuple(parity_masks), code_bits, data_bits, t))
