"""Bit primitives the coding ops consume: Gray codes, the single-error
Hamming-bound test, and the syndrome→flip-mask LUT for t>=2 correction.
Everything protocol-shaped (parity-matrix decode, CSS explicit-header
algebra) is caller/test-side, not here."""

from __future__ import annotations

from functools import lru_cache as _lru_cache
from itertools import combinations as _combinations

import numpy as _np

from marconi.deadline import check_deadline as _check_deadline


def gray_encode(x: int) -> int:
    return x ^ (x >> 1)


def gray_decode(g: int) -> int:
    x = g
    while g := g >> 1:
        x ^= g
    return x


def effective_t(
    n_parity: int, data_bits: int, correct_single: bool | None, correct: int | None
) -> int:
    if correct is not None:
        return correct
    do_single = (
        can_correct(n_parity, data_bits) if correct_single is None else correct_single
    )
    return 1 if do_single else 0


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


# The table enumerates every error pattern of weight 1..t, so its size is
# sum(C(code_bits, w)) — superlinear in the code length and paid inside a
# pydantic validator. Capped so an oversized code is refused with a message
# naming the cost, instead of wedging the caller mid-validation.
MAX_SYNDROME_PATTERNS = 1 << 20


def _pattern_count(code_bits: int, t: int) -> int:
    total = 0
    for w in range(1, t + 1):
        term = 1
        for i in range(w):
            term = term * (code_bits - i) // (i + 1)
        total += term
        if total > MAX_SYNDROME_PATTERNS:
            break
    return total


@_lru_cache(maxsize=64)
def _syndrome_table_cached(
    parity_masks: tuple[int, ...], code_bits: int, data_bits: int, t: int
) -> dict[int, int]:
    patterns = _pattern_count(code_bits, t)
    if patterns > MAX_SYNDROME_PATTERNS:
        raise ValueError(
            f"correcting {t} errors in a {code_bits}-bit code needs a syndrome "
            f"table of over {patterns:,} entries (cap {MAX_SYNDROME_PATTERNS:,}); "
            "lower 'correct' or use a shorter code"
        )
    sigs = _column_signatures(list(parity_masks), code_bits, data_bits)
    table: dict[int, int] = {}
    for w in range(1, t + 1):
        for i, combo in enumerate(_combinations(range(code_bits), w)):
            # this runs inside a pydantic validator, and the cap above still
            # admits ~1M patterns: the caller's timeout has to reach in here
            if not i % 4096:
                _check_deadline()
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


def syndrome_key(value: int, n_parity: int) -> bytes:
    """A syndrome's canonical fixed-width key: equation p occupies bit
    ``n_parity-1-p`` of the integer form (see _column_signatures) and byte
    position p here. Both sides of a lookup pack through this one function, so
    a syndrome is compared as a bit VECTOR — the packed-integer form silently
    dropped every equation past bit 63 of the register holding it."""
    bits = _np.fromiter(
        ((value >> (n_parity - 1 - p)) & 1 for p in range(n_parity)),
        dtype=_np.uint8,
        count=n_parity,
    )
    return bytes(_np.packbits(bits))


def flip_table(
    parity_masks: list[int], code_bits: int, data_bits: int, t: int
) -> dict[bytes, int]:
    """syndrome key -> codeword-position flip mask, for t=1 (each column's own
    signature) and t>=2 (every pattern up to weight t). One table for both, so
    a correction path cannot exist for one t and not the other. A duplicate
    single-error signature resolves to the LOWEST column index.

    Memoized via _flip_table_cached: only syndrome_table (the t>=2 half) was
    cached, so the windowed decode rebuilt this once PER WINDOW - measured
    4.4 ms per rebuild (~100% of a Golay window's cost, 16.8 s over 4000
    windows) and 0.787 s per SYNC HIT at the syndrome-pattern cap."""
    return dict(_flip_table_cached(tuple(parity_masks), code_bits, data_bits, t))


@_lru_cache(maxsize=16)
def _flip_table_cached(
    parity_masks: tuple[int, ...], code_bits: int, data_bits: int, t: int
) -> dict[bytes, int]:
    n_parity = code_bits - data_bits
    if t >= 2:
        return {
            syndrome_key(syn, n_parity): flip
            for syn, flip in syndrome_table(
                list(parity_masks), code_bits, data_bits, t
            ).items()
        }
    table: dict[bytes, int] = {}
    if t == 1:
        for column, sig in enumerate(
            _column_signatures(list(parity_masks), code_bits, data_bits)
        ):
            if sig:
                table.setdefault(syndrome_key(sig, n_parity), 1 << column)
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
