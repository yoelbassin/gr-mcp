from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

try:  # reedsolo's optional Cython build: identical API, 10-50x faster RS. NOT
    # a PyPI package of its own, so it cannot be a declared dependency - only
    # envs that hand-build reedsolo with cythonization get this path.
    import creedsolo as _rs
except ImportError:  # pure-Python fallback, always installed
    import reedsolo as _rs

from marconi.deadline import check_deadline
from marconi.engine.coding.carrier import (
    CarrierInvariantError,
    CodingCarrier,
    StepStats,
    Window,
)
from marconi.engine.coding.primitives import effective_t, flip_table
from marconi.engine.types.enums import DecodeMode, EmitMode


def _chance_valid_rate(n: int, redundancy: int, t: int, q: int) -> float:
    """Probability that a uniformly random q-ary word of length n lands within
    distance t of a codeword (sphere volume over the redundancy space), in log
    space so huge codes cannot overflow. Doubles as the per-position chance of
    matching an n-symbol pattern within t errors (redundancy = n)."""
    log_q = math.log(q)
    total = 0.0
    for i in range(t + 1):
        log_term = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * math.log(q - 1)
            - redundancy * log_q
        )
        total += math.exp(min(log_term, 0.0))
    return min(total, 1.0)


def _bitorder(bit_order: str) -> Literal["little", "big"]:
    return "little" if bit_order == "lsb" else "big"


def bytes_to_bits(data: bytes, bit_order: str = "msb") -> npt.NDArray[np.uint8]:
    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8), bitorder=_bitorder(bit_order)
    )


def bits_to_bytes(bits: npt.ArrayLike, bit_order: str = "msb") -> bytes:
    return np.packbits(
        np.asarray(bits, dtype=np.uint8), bitorder=_bitorder(bit_order)
    ).tobytes()


def segment_rx(c: CodingCarrier, *, frame_body_len: int) -> CodingCarrier:
    check_deadline()
    windows: list[Window] = []
    i = 0
    while i + frame_body_len <= len(c.bits):
        windows.append(Window(start=i, cursor=i, end=i + frame_body_len))
        i += frame_body_len
    return CodingCarrier(bits=c.bits, windows=windows, marks=c.marks)


def _correlate(
    stream: npt.NDArray[np.uint8], pat: npt.NDArray[np.uint8], max_errors: int
) -> list[int]:
    """Non-overlapping positions where pat matches stream within max_errors.
    One O(n) mismatch accumulator per pattern bit, never an (n, m) array: at
    the bits-layer budget an n x m compare is tens of GiB."""
    m = pat.size
    size = stream.size - m + 1
    counts = np.zeros(size, np.min_scalar_type(m))
    for j in range(m):
        check_deadline()
        counts += stream[j : j + size] != pat[j]
    out: list[int] = []
    reach = 0
    for i in np.flatnonzero(counts <= max_errors):
        if i < reach:
            continue
        out.append(int(i))
        reach = int(i) + m
    return out


# Rotations of the pattern used as surrogate nulls, at two different fractions
# of the length so a pattern whose own period divides one shift is still
# measured against the other.
_SURROGATE_SHIFTS = (2, 3)


def _surrogate_chance(
    stream: npt.NDArray[np.uint8], pat: npt.NDArray[np.uint8], max_errors: int
) -> float:
    """How often this stream matches a MEANINGLESS pattern of the same length
    and weight — the pattern rotated off its own phase.

    The uniform-bit expectation is a lower bound that collapses toward zero as
    the pattern lengthens, so on a degenerate stream every long pattern clears
    it: a dead all-zero demod searched for an all-zero sync word scored 6250
    hits against an expectation of 4.66e-05 and read "decoded". A rotation is
    matched by the stream's own structure but not by a real sync, so it
    measures this stream instead of assuming a uniform one."""
    best = 0
    for divisor in _SURROGATE_SHIFTS:
        shift = pat.size // divisor
        if not shift or shift == pat.size:
            continue
        best = max(best, len(_correlate(stream, np.roll(pat, shift), max_errors)))
    return float(best)


def sync_word_rx(
    c: CodingCarrier, *, sync: str = "", bits: str = "", max_errors: int = 0
) -> CodingCarrier:
    pat = (
        bytes_to_bits(bytes.fromhex(sync))
        if sync
        else np.asarray([int(ch) for ch in bits], np.uint8)
    )
    stream = np.asarray(c.bits, dtype=np.uint8)
    m = pat.size
    if m == 0 or stream.size < m:
        # no search ran (the stream cannot contain the pattern): no stats, so
        # the quality extractor treats it as untestable rather than negative
        return CodingCarrier(bits=stream, windows=[], marks=c.marks)
    size = stream.size - m + 1
    uniform = float(size) * _chance_valid_rate(m, m, max_errors, 2)
    hits = _correlate(stream, pat, max_errors)
    chance = max(uniform, _surrogate_chance(stream, pat, max_errors))
    return CodingCarrier(
        bits=stream,
        windows=[Window(start=i + m, cursor=i + m) for i in hits],
        marks=c.marks,
        stats=StepStats(chance_windows=chance),
    )


def mark_frame_rx(c: CodingCarrier, *, offset_bits: int = 0) -> CodingCarrier:
    bits = np.asarray(c.bits, np.uint8)
    windows: list[Window] = []
    for m in c.marks:
        start = int(m) + offset_bits
        if 0 <= start < bits.size:
            windows.append(Window(start=start, cursor=start))
    return CodingCarrier(bits=bits, windows=windows, marks=c.marks)


def _codebook_maps(
    code_bits: int, data_bits: int, table: list[int]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    if len(table) != (1 << data_bits):
        raise ValueError(
            f"codebook table has {len(table)} entries, need {1 << data_bits} "
            f"for data_bits={data_bits}"
        )
    fwd = np.asarray(table, dtype=np.int64)
    _check_table_entries(fwd, code_bits)
    inv = np.zeros(1 << code_bits, dtype=np.int64)  # unknown symbol -> 0
    inv[fwd] = np.arange(fwd.size, dtype=np.int64)
    return fwd, inv


def _check_table_entries(fwd: npt.NDArray[np.int64], code_bits: int) -> None:
    if fwd.size and (int(fwd.min()) < 0 or int(fwd.max()) >= (1 << code_bits)):
        raise ValueError("codebook entries must lie in [0, 2**code_bits)")
    if np.unique(fwd).size != fwd.size:
        raise ValueError(
            "codebook entries must be unique; duplicate codewords make "
            "decode ambiguous"
        )


NEAREST_MAX_CODE_BITS = 63  # nearest decode packs codewords through int64

# The (words, codewords, code_bits) distance cube is the peak allocation of a
# nearest decode, so the word chunk is sized from a BYTE budget rather than a
# fixed word count: at a flat 65536 words a (24,12) spec the validator accepts
# asked numpy for a single 6.44 GB temporary, and the deadline check runs
# before the allocation, so the timeout could not save it either.
_NEAREST_CUBE_BYTES = 64 << 20


def _nearest_chunk(codewords: int, code_bits: int) -> int:
    return max(1, _NEAREST_CUBE_BYTES // max(1, codewords * code_bits))


def _nearest_values(
    grouped: npt.NDArray[np.uint8], table: list[int], code_bits: int
) -> npt.NDArray[np.int64]:
    if code_bits > NEAREST_MAX_CODE_BITS:
        raise ValueError(
            "nearest decode packs codewords through int64; "
            f"{NEAREST_MAX_CODE_BITS} bits is the ceiling"
        )
    fwd = np.asarray(table, dtype=np.int64)
    _check_table_entries(fwd, code_bits)
    tbl = np.zeros((fwd.size, code_bits), np.uint8)
    for j in range(code_bits):
        tbl[:, j] = (fwd >> (code_bits - 1 - j)) & 1
    out = np.empty(grouped.shape[0], np.int64)
    chunk = _nearest_chunk(int(fwd.size), code_bits)
    for lo in range(0, grouped.shape[0], chunk):
        check_deadline()
        seg = grouped[lo : lo + chunk]
        dist = (seg[:, None, :] != tbl[None, :, :]).sum(axis=2)
        out[lo : lo + seg.shape[0]] = dist.argmin(axis=1)
    return out


def _sym_dtype(width: int) -> type[np.signedinteger[Any]]:
    # width <= 31 fits int32 with room for the sign bit; wider needs int64.
    return np.int64 if width > 31 else np.int32


def _unpack_symbols(
    bits: npt.NDArray[np.uint8], width: int
) -> npt.NDArray[np.signedinteger[Any]]:
    n = bits.size // width
    g = np.asarray(bits[: n * width], dtype=np.uint8).reshape(n, width)
    out: npt.NDArray[np.signedinteger[Any]] = np.zeros(n, dtype=_sym_dtype(width))
    for j in range(width):
        out <<= 1
        out |= g[:, j]
    return out


def _pack_symbols(
    values: npt.NDArray[np.signedinteger[Any]], width: int
) -> npt.NDArray[np.uint8]:
    out = np.empty((values.size, width), np.uint8)
    for j in range(width):
        out[:, j] = (values >> (width - 1 - j)) & 1
    return out.reshape(-1)


def codebook_rx(
    c: CodingCarrier,
    *,
    code_bits: int,
    data_bits: int,
    table: list[int],
    symbol_input: bool = False,
    decode: DecodeMode = DecodeMode.EXACT,
) -> CodingCarrier:
    if DecodeMode(decode) is DecodeMode.NEAREST:

        def _values(
            syms: npt.NDArray[np.signedinteger[Any]],
        ) -> npt.NDArray[np.int64]:
            rows = _pack_symbols(syms, code_bits).reshape(-1, code_bits)
            return _nearest_values(rows, table, code_bits)

    else:
        _, inv = _codebook_maps(code_bits, data_bits, table)

        def _values(
            syms: npt.NDArray[np.signedinteger[Any]],
        ) -> npt.NDArray[np.int64]:
            picked: npt.NDArray[np.int64] = inv[syms]
            return picked

    if symbol_input:
        if c.symbols is None:
            raise ValueError("codebook(symbol_input=True) needs a symbols carrier")
        syms = np.asarray(c.symbols, np.int64)
        if syms.size and (int(syms.min()) < 0 or int(syms.max()) >= (1 << code_bits)):
            raise ValueError(
                f"symbol values must lie in [0, {1 << code_bits}) for "
                f"code_bits={code_bits}; got [{int(syms.min())}, {int(syms.max())}]."
                " m_slice levels are the emitted symbol values — label them "
                "0..2**code_bits-1"
            )
        bits = (
            _pack_symbols(_values(syms), data_bits)
            if syms.size
            else np.zeros(0, np.uint8)
        )
        marks = tuple(int(m) * data_bits for m in c.marks)
        return CodingCarrier(bits=bits, marks=marks)

    def _decode(bits: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return _pack_symbols(_values(_unpack_symbols(bits, code_bits)), data_bits)

    return _decode_scoped(c, lambda bits: (_decode(bits), None))


def _syndrome_bits(
    words: npt.NDArray[np.uint8], parity_masks: list[int], data_bits: int
) -> npt.NDArray[np.uint8]:
    """One row per word, ONE COLUMN PER PARITY EQUATION. The equations used to
    be shifted into a single fixed-width integer, which dropped every equation
    past bit 63 of it: measured on a (300,100) code, a wholly corrupt stream
    reported words_valid = 400/400 and the quality layer read "decoded".
    A syndrome is as wide as the code says it is, so it is never a register."""
    syn = np.empty((words.shape[0], len(parity_masks)), np.uint8)
    for p, mask in enumerate(parity_masks):
        col = words[:, data_bits + p].astype(np.uint8)
        for c in range(data_bits):
            if (mask >> c) & 1:
                col = col ^ words[:, c]
        syn[:, p] = col & 1
    return syn


def _correct_words(
    words: npt.NDArray[np.uint8], syn: npt.NDArray[np.uint8], lut: dict[bytes, int]
) -> npt.NDArray[np.bool_]:
    """Flip the error pattern each failing syndrome names; returns which words
    were repaired.

    Iterates the correctable PATTERNS, not the words and not the syndromes the
    data happens to contain: the pattern count is fixed by the spec (one per
    column at t=1), so the numpy call count does not move with input size —
    a per-word python loop is what this shape exists to avoid. Each pattern
    costs one vectorized row compare over the words that failed parity."""
    fixed = np.zeros(words.shape[0], bool)
    bad = np.flatnonzero(syn.any(axis=1))
    if not lut or bad.size == 0:
        return fixed
    keys = np.packbits(syn[bad], axis=1)
    unresolved = np.ones(bad.size, bool)
    for key, flip in lut.items():
        check_deadline()
        want = np.frombuffer(key, np.uint8)
        match = unresolved & (keys == want).all(axis=1)
        rows = bad[match]
        if rows.size == 0:
            continue
        positions = [p for p in range(words.shape[1]) if (flip >> p) & 1]
        if positions:
            words[np.ix_(rows, positions)] ^= 1
        fixed[rows] = True
        unresolved &= ~match
        if not unresolved.any():
            break
    return fixed


def _block_decode(
    bits: npt.NDArray[np.uint8],
    code_bits: int,
    data_bits: int,
    parity_masks: list[int],
    t: int,
    emit: EmitMode,
) -> tuple[npt.NDArray[np.uint8], tuple[int, int]]:
    # Codeword basis is LSB-first: stream bit j of a stride is codeword bit j,
    # so parity_masks and the emitted data bits are LSB-first too. A caller
    # assembling codewords MSB-first must reconcile the basis (supply masks
    # LSB-first, or reverse each stride).
    n_words = bits.size // code_bits
    words = bits[: n_words * code_bits].reshape(n_words, code_bits).copy()
    ok = np.zeros(n_words, bool)
    if n_words and parity_masks:
        check_deadline()
        syn = _syndrome_bits(words, parity_masks, data_bits)
        ok = ~syn.any(axis=1)
        if t >= 1:
            ok |= _correct_words(
                words, syn, flip_table(parity_masks, code_bits, data_bits, t)
            )
    out = words if emit is EmitMode.CODEWORD else words[:, :data_bits]
    return out.reshape(-1), (int(ok.sum()), n_words)


def block_code_rx(
    c: CodingCarrier,
    *,
    code_bits: int,
    data_bits: int,
    parity_masks: list[int],
    correct_single: bool | None = None,
    correct: int | None = None,
    emit: EmitMode = EmitMode.DATA,
) -> CodingCarrier:
    emit = EmitMode(emit)
    t = effective_t(code_bits - data_bits, data_bits, correct_single, correct)
    return _decode_scoped(
        c,
        lambda bits: _block_decode(bits, code_bits, data_bits, parity_masks, t, emit),
        chance_word_rate=_chance_valid_rate(code_bits, code_bits - data_bits, t, 2),
    )


def _word_stats(
    tallies: list[tuple[int, int]], chance_word_rate: float | None
) -> StepStats | None:
    if not tallies:
        return None
    return StepStats(
        words_valid=sum(v for v, _ in tallies),
        words_total=sum(n for _, n in tallies),
        chance_word_rate=chance_word_rate,
    )


def _decode_scoped(
    c: CodingCarrier,
    decode: Callable[
        [npt.NDArray[np.uint8]],
        tuple[npt.NDArray[np.uint8], tuple[int, int] | None],
    ],
    *,
    chance_word_rate: float | None = None,
) -> CodingCarrier:
    tallies: list[tuple[int, int]] = []
    if c.windows is None:
        out, tally = decode(np.asarray(c.bits, np.uint8))
        if tally is not None:
            tallies.append(tally)
        return CodingCarrier(bits=out, stats=_word_stats(tallies, chance_word_rate))
    bits = np.asarray(c.bits, np.uint8)
    pieces: list[npt.NDArray[np.uint8]] = []
    windows: list[Window] = []
    pos = 0
    for lo, hi in c.window_spans():
        check_deadline()
        dec, tally = decode(bits[lo:hi])
        if tally is not None:
            tallies.append(tally)
        windows.append(Window(start=pos, cursor=pos, end=pos + int(dec.size)))
        pieces.append(dec)
        pos += int(dec.size)
    joined = np.concatenate(pieces) if pieces else np.zeros(0, np.uint8)
    return CodingCarrier(
        bits=joined, windows=windows, stats=_word_stats(tallies, chance_word_rate)
    )


def _rs_decode_words(
    bits: npt.NDArray[np.uint8],
    codec: _rs.RSCodec,
    symbol_bits: int,
    n: int,
    k: int,
    emit: EmitMode,
) -> tuple[npt.NDArray[np.uint8], tuple[int, int]]:
    syms = _unpack_symbols(bits, symbol_bits)
    out: list[int] = []
    total = syms.size // n
    valid = 0
    for w in range(total):
        if w & 0xFF == 0:
            check_deadline()
        word = [int(s) for s in syms[w * n : (w + 1) * n]]
        try:
            data, full, _ = codec.decode(word)
            out.extend(full if emit is EmitMode.CODEWORD else data)
            valid += 1
        except _rs.ReedSolomonError:
            # uncorrectable is detectable, never repaired by guessing: emit the
            # received word unchanged so downstream framing keeps its alignment
            out.extend(word if emit is EmitMode.CODEWORD else word[:k])
    return _pack_symbols(np.asarray(out, np.int64), symbol_bits), (valid, total)


def rs_code_rx(
    c: CodingCarrier,
    *,
    symbol_bits: int,
    n: int,
    k: int,
    prim_poly: int,
    fcr: int = 0,
    generator: int = 2,
    emit: EmitMode = EmitMode.DATA,
) -> CodingCarrier:
    emit = EmitMode(emit)
    codec = _rs.RSCodec(
        nsym=n - k,
        nsize=n,
        fcr=fcr,
        prim=prim_poly,
        generator=generator,
        c_exp=symbol_bits,
    )
    return _decode_scoped(
        c,
        lambda bits: _rs_decode_words(bits, codec, symbol_bits, n, k, emit),
        chance_word_rate=_chance_valid_rate(n, n - k, (n - k) // 2, 1 << symbol_bits),
    )


def _perm_span(idx: npt.NDArray[np.int64]) -> int:
    """Input stride a gather consumes: one past its highest index, NOT its
    length. A dropping gather (one that skips per-slot bits, so it spans more
    input than it emits) reads further than it writes, and assuming
    stride == len(perm) indexes out of bounds. Both scopes must agree on this."""
    return int(idx.max()) + 1 if idx.size else 0


def _permute_block(
    bits: npt.NDArray[np.uint8], idx: npt.NDArray[np.int64], span: int
) -> npt.NDArray[np.uint8]:
    n = bits.size // span if span else 0
    return bits[: n * span].reshape(n, span)[:, idx].reshape(-1)


def permute_rx(c: CodingCarrier, *, perm: list[int]) -> CodingCarrier:
    idx = np.asarray(perm, np.int64)
    span = _perm_span(idx)
    return _decode_scoped(c, lambda bits: (_permute_block(bits, idx, span), None))


def realign_rx(c: CodingCarrier, *, bit_offset: int) -> CodingCarrier:
    if c.windows is None:
        marks = tuple(m - bit_offset for m in c.marks if m >= bit_offset)
        return CodingCarrier(
            bits=np.asarray(c.bits, np.uint8)[bit_offset:], marks=marks
        )
    # materialize each window's scope end before shifting: the dropped lead
    # bits belong to this window, never to the previous frame's tail
    nxt = [w.cursor for w in c.windows[1:]] + [None]
    windows = [
        Window(
            start=w.start,
            cursor=w.cursor + bit_offset,
            end=w.end if w.end is not None else nx,
        )
        for w, nx in zip(c.windows, nxt)
    ]
    return replace(c, bits=np.asarray(c.bits, np.uint8), windows=windows)


def differential_rx(c: CodingCarrier, *, invert: bool = False) -> CodingCarrier:
    b = np.asarray(c.bits, dtype=np.uint8)
    if b.size == 0:
        return replace(c, bits=b)
    prev = np.concatenate(([np.uint8(0)], b[:-1]))
    inv = np.uint8(1 if invert else 0)
    return replace(c, bits=np.bitwise_xor(np.bitwise_xor(b, prev), inv))


def _nibble_swap_bits(bits: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    b = np.asarray(bits, dtype=np.uint8).copy()
    n = (b.size // 8) * 8
    if n:
        g = b[:n].reshape(-1, 8)
        b[:n] = np.concatenate([g[:, 4:], g[:, :4]], axis=1).reshape(-1)
    return b


def nibble_swap_rx(c: CodingCarrier) -> CodingCarrier:
    return replace(c, bits=_nibble_swap_bits(c.bits))


def _seq_bits(sequence: str) -> npt.NDArray[np.uint8]:
    return np.unpackbits(np.frombuffer(bytes.fromhex(sequence), dtype=np.uint8))


_LFSR_CHUNK = 1 << 16


def _lfsr_seq(mask: int, seed: int, length: int, n: int) -> npt.NDArray[np.uint8]:
    # The Fibonacci state is a sliding window of the output, so the output
    # stream obeys out[t+L] = XOR of out[t+j] over the mask's tap positions:
    # a whole block resolves as uint64 row-masks AND window + popcount,
    # never a per-bit Python loop (the blind whole-stream case is up to
    # 2^30 bits).
    out = np.empty(n, np.uint8)
    head = min(length, n)
    for i in range(head):
        out[i] = (seed >> i) & 1
    if n <= length:
        return out
    taps = [j for j in range(length) if (mask >> j) & 1]
    chunk = min(_LFSR_CHUNK, n - length)
    # row i = which bits of the L-wide window XOR into block output i
    rows: list[int] = []
    for i in range(chunk):
        if not i % 4096:
            check_deadline()
        acc = 0
        for j in taps:
            m = i - length + j
            acc ^= (1 << (m + length)) if m < 0 else rows[m]
        rows.append(acc)
    rows_arr = np.asarray(rows, np.uint64)
    t = length
    window = seed & ((1 << length) - 1)
    while t < n:
        check_deadline()
        c = min(chunk, n - t)
        vals = np.bitwise_and(rows_arr[:c], np.uint64(window))
        out[t : t + c] = (np.bitwise_count(vals) & 1).astype(np.uint8)
        t += c
        window = 0
        for i, b in enumerate(out[t - length : t]):
            window |= int(b) << i
    return out


def descramble_rx(
    c: CodingCarrier,
    *,
    sequence: str = "",
    lfsr_mask: int | None = None,
    lfsr_seed: int | None = None,
    lfsr_len: int | None = None,
) -> CodingCarrier:
    bits = np.asarray(c.bits, dtype=np.uint8)
    if bits.size == 0:
        return c
    lfsr = (lfsr_mask, lfsr_seed, lfsr_len)
    if lfsr_mask is not None and lfsr_seed is not None and lfsr_len is not None:
        return _descramble_lfsr(c, bits, lfsr_mask, lfsr_seed, lfsr_len)
    if any(v is not None for v in lfsr):
        # DescrambleStep's validator already refuses a partial triple; falling
        # through to the sequence path here would descramble with an EMPTY
        # sequence and return the input unchanged, which reads as a clean run
        raise CarrierInvariantError(
            "descramble needs lfsr_mask, lfsr_seed and lfsr_len together; "
            f"got {lfsr}"
        )
    return _descramble_sequence(c, bits, sequence)


def _descramble_lfsr(
    c: CodingCarrier, bits: npt.NDArray[np.uint8], mask: int, seed: int, length: int
) -> CodingCarrier:
    if c.windows is None:
        lfsr_seq = _lfsr_seq(mask, seed, length, bits.size)
        return replace(c, bits=np.bitwise_xor(bits, lfsr_seq))
    out = bits.copy()
    spans = c.window_spans()
    max_span = max((hi - lo for lo, hi in spans), default=0)
    if max_span > 0:
        lfsr_seq = _lfsr_seq(mask, seed, length, max_span)
        for lo, hi in spans:
            out[lo:hi] = np.bitwise_xor(bits[lo:hi], lfsr_seq[: hi - lo])
    return replace(c, bits=out)


def _descramble_sequence(
    c: CodingCarrier, bits: npt.NDArray[np.uint8], sequence: str
) -> CodingCarrier:
    seq = _seq_bits(sequence)
    if seq.size == 0:
        return c
    if c.windows is None:
        return replace(c, bits=np.bitwise_xor(bits, np.resize(seq, bits.size)))
    out = bits.copy()
    for lo, hi in c.window_spans():
        out[lo:hi] = np.bitwise_xor(bits[lo:hi], np.resize(seq, hi - lo))
    return replace(c, bits=out)
