from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np

try:  # reedsolo's optional Cython build: identical API, 10-50x faster RS. NOT
    # a PyPI package of its own, so it cannot be a declared dependency - only
    # envs that hand-build reedsolo with cythonization get this path.
    import creedsolo as _rs
except ImportError:  # pure-Python fallback, always installed
    import reedsolo as _rs

from marconi.engine.coding.carrier import CodingCarrier, StepStats, Window
from marconi.engine.coding.primitives import can_correct, syndrome_table


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


def bytes_to_bits(data: bytes, bit_order: str = "msb") -> np.ndarray:
    return np.unpackbits(
        np.frombuffer(data, dtype=np.uint8), bitorder=_bitorder(bit_order)
    )


def bits_to_bytes(bits: np.ndarray, bit_order: str = "msb") -> bytes:
    return np.packbits(
        np.asarray(bits, dtype=np.uint8), bitorder=_bitorder(bit_order)
    ).tobytes()


def segment_rx(c: CodingCarrier, *, frame_body_len: int) -> CodingCarrier:
    windows: list[Window] = []
    i = 0
    while i + frame_body_len <= len(c.bits):
        windows.append(Window(start=i, cursor=i))
        i += frame_body_len
    return CodingCarrier(bits=c.bits, windows=windows, marks=c.marks)


def sync_word_rx(
    c: CodingCarrier, *, sync: str = "", bits: str = "", max_errors: int = 0
) -> CodingCarrier:
    """Correlating window seeder: seed a window just past every position where
    the bitstream matches the pattern (``sync`` as hex, or ``bits`` as a
    '0'/'1' string for non-byte lengths) within ``max_errors`` bit flips (a
    sync word is detected by correlation, not equality). Matches are taken
    non-overlapping."""
    pat = (
        bytes_to_bits(bytes.fromhex(sync))
        if sync
        else np.asarray([int(ch) for ch in bits], np.uint8)
    )
    stream = np.asarray(c.bits, dtype=np.uint8)
    m = pat.size
    windows: list[Window] = []
    if m == 0 or stream.size < m:
        # no search ran (the stream cannot contain the pattern): no stats, so
        # the quality extractor treats it as untestable rather than negative
        return CodingCarrier(bits=stream, windows=windows, marks=c.marks)
    # one O(n) mismatch accumulator per pattern bit, never an (n, m) array:
    # at the bits-layer budget an n x m compare is tens of GiB
    size = stream.size - m + 1
    chance = float(size) * _chance_valid_rate(m, m, max_errors, 2)
    counts = np.zeros(size, np.min_scalar_type(m))
    for j in range(m):
        counts += stream[j : j + size] != pat[j]
    hits = np.flatnonzero(counts <= max_errors)
    reach = 0
    for i in hits:
        if i < reach:
            continue
        start = int(i) + m
        windows.append(Window(start=start, cursor=start))
        reach = int(i) + m
    return CodingCarrier(
        bits=stream,
        windows=windows,
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
) -> tuple[np.ndarray, np.ndarray]:
    if len(table) != (1 << data_bits):
        raise ValueError(
            f"codebook table has {len(table)} entries, need {1 << data_bits} "
            f"for data_bits={data_bits}"
        )
    fwd = np.asarray(table, dtype=np.int64)
    if fwd.size and int(fwd.max()) >= (1 << code_bits):
        raise ValueError("codebook symbol exceeds code_bits width")
    inv = np.zeros(1 << code_bits, dtype=np.int64)  # unknown symbol -> 0
    inv[fwd] = np.arange(fwd.size, dtype=np.int64)
    return fwd, inv


def _nearest_values(
    grouped: np.ndarray, table: list[int], code_bits: int
) -> np.ndarray:
    if code_bits > 63:
        raise ValueError(
            "nearest decode packs codewords through int64; 63 bits is the " "ceiling"
        )
    fwd = np.asarray(table, dtype=np.int64)
    if fwd.size and int(fwd.max()) >= (1 << code_bits):
        raise ValueError("codebook symbol exceeds code_bits width")
    tbl = np.zeros((fwd.size, code_bits), np.uint8)
    for j in range(code_bits):
        tbl[:, j] = (fwd >> (code_bits - 1 - j)) & 1
    out = np.empty(grouped.shape[0], np.int64)
    chunk = 1 << 16
    for lo in range(0, grouped.shape[0], chunk):
        seg = grouped[lo : lo + chunk]
        dist = (seg[:, None, :] != tbl[None, :, :]).sum(axis=2)
        out[lo : lo + seg.shape[0]] = dist.argmin(axis=1)
    return out


def _sym_dtype(width: int) -> type:
    # width <= 31 fits int32 with room for the sign bit; wider needs int64.
    return np.int64 if width > 31 else np.int32


def _unpack_symbols(bits: np.ndarray, width: int) -> np.ndarray:
    n = bits.size // width
    g = np.asarray(bits[: n * width], dtype=np.uint8).reshape(n, width)
    out: np.ndarray = np.zeros(n, dtype=_sym_dtype(width))
    for j in range(width):
        out <<= 1
        out |= g[:, j]
    return out


def _pack_symbols(values: np.ndarray, width: int) -> np.ndarray:
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
    decode: str = "exact",
) -> CodingCarrier:
    """Fixed-width symbol substitution (a line/block code): each ``code_bits``
    input symbol maps to its ``data_bits`` table value. One op expresses
    3-of-6, Manchester, PPM chip-pairs — the table and widths are caller data.
    ``decode="exact"`` is an inverse-table lookup (unknown codewords degrade to
    0); ``decode="nearest"`` is min-Hamming-distance against the table, ties
    broken by lowest table index, and never builds the ``2**code_bits``
    inverse table — the only mode safe for wide chip codes."""
    if decode == "nearest":

        def _values(syms: np.ndarray) -> np.ndarray:
            rows = _pack_symbols(syms, code_bits).reshape(-1, code_bits)
            return _nearest_values(rows, table, code_bits)

    else:
        _, inv = _codebook_maps(code_bits, data_bits, table)

        def _values(syms: np.ndarray) -> np.ndarray:
            return inv[syms]

    if symbol_input:
        if c.symbols is None:
            raise ValueError("codebook(symbol_input=True) needs a symbols carrier")
        syms = np.asarray(c.symbols, np.int64)
        bits = (
            _pack_symbols(_values(syms), data_bits)
            if syms.size
            else np.zeros(0, np.uint8)
        )
        marks = tuple(int(m) * data_bits for m in c.marks)
        return CodingCarrier(bits=bits, marks=marks)

    def _decode(bits: np.ndarray) -> np.ndarray:
        return _pack_symbols(_values(_unpack_symbols(bits, code_bits)), data_bits)

    if c.windows is not None:
        bits_arr = np.asarray(c.bits, np.uint8)
        cursors = [w.cursor for w in c.windows]
        if any(b <= a for a, b in zip(cursors, cursors[1:])):
            raise ValueError("codebook needs strictly increasing window cursors")
        pieces: list[np.ndarray] = []
        windows: list[Window] = []
        pos = 0
        for w, hi in zip(c.windows, cursors[1:] + [int(bits_arr.size)]):
            dec = _decode(bits_arr[w.cursor : hi])
            windows.append(Window(start=pos, cursor=pos))
            pieces.append(dec)
            pos += int(dec.size)
        joined = np.concatenate(pieces) if pieces else np.zeros(0, np.uint8)
        return CodingCarrier(bits=joined, windows=windows)
    data = _decode(np.asarray(c.bits, np.uint8))
    return CodingCarrier(bits=data, windows=c.windows)


def _syndromes(
    words: np.ndarray, parity_masks: list[int], code_bits: int, data_bits: int
) -> np.ndarray:
    # syndrome stays uint8 (mod-2 XOR of selected data cols + parity col),
    # then packs to a small per-word int - no (n_words, k) int matrix
    n_parity = code_bits - data_bits
    dtype = _sym_dtype(n_parity)
    rows = [[(m >> b) & 1 for b in range(data_bits)] for m in parity_masks]
    s_int: np.ndarray = np.zeros(words.shape[0], dtype)
    for p, row in enumerate(rows):
        col = words[:, data_bits + p].astype(np.uint8)
        for c, bit in enumerate(row):
            if bit:
                col = col ^ words[:, c]
        s_int = (s_int << 1) | (col & 1)
    return s_int


def _effective_t(
    n_parity: int, data_bits: int, correct_single: bool | None, correct: int | None
) -> int:
    if correct is not None:
        return correct
    do_single = (
        can_correct(n_parity, data_bits) if correct_single is None else correct_single
    )
    return 1 if do_single else 0


def _block_decode(
    bits: np.ndarray,
    code_bits: int,
    data_bits: int,
    parity_masks: list[int],
    t: int,
    emit: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    # Codeword basis is LSB-first: stream bit j of a stride is codeword bit j,
    # so parity_masks and the emitted data bits are LSB-first too. The CSS
    # explicit decoder assembles the same block_fec_decode input MSB-first
    # (bits/symbols.py, _fec_nibbles); re-expressing that path on this stage
    # must reconcile the basis (supply masks LSB-first, or reverse each
    # stride).
    n_parity = code_bits - data_bits
    n_words = bits.size // code_bits
    words = bits[: n_words * code_bits].reshape(n_words, code_bits).copy()
    ok = np.zeros(n_words, bool)
    if n_words and parity_masks:
        s_int = _syndromes(words, parity_masks, code_bits, data_bits)
        ok = s_int == 0
        if t == 1:
            dtype = _sym_dtype(n_parity)
            rows = [[(m >> b) & 1 for b in range(data_bits)] for m in parity_masks]
            # a column's syndrome signature, data columns then the parity
            # identity; first ascending match flips, matching the scalar scan
            weights = np.left_shift(1, np.arange(n_parity - 1, -1, -1, dtype=dtype))
            col_sig = np.concatenate([np.asarray(rows, dtype).T @ weights, weights])
            bad = np.flatnonzero(s_int)
            if bad.size:
                match = s_int[bad, None] == col_sig[None, :]
                col_idx = match.argmax(axis=1)
                fixable = match.any(axis=1)
                words[bad[fixable], col_idx[fixable]] ^= 1
                ok[bad[fixable]] = True
        if t >= 2:
            lut = syndrome_table(parity_masks, code_bits, data_bits, t)
            keys = np.fromiter(lut.keys(), np.int64, len(lut))
            flips = np.fromiter(lut.values(), np.int64, len(lut))
            order_ = np.argsort(keys)
            keys, flips = keys[order_], flips[order_]
            s64 = s_int.astype(np.int64)
            bad = np.flatnonzero(s64)
            if bad.size:
                pos = np.searchsorted(keys, s64[bad])
                pos = np.clip(pos, 0, keys.size - 1)
                hit = keys[pos] == s64[bad]
                flip = flips[pos[hit]]
                fixed = bad[hit]
                flip_bits = ((flip[:, None] >> np.arange(code_bits)) & 1).astype(
                    np.uint8
                )
                words[fixed] ^= flip_bits
                ok[fixed] = True
    out = words if emit == "codeword" else words[:, :data_bits]
    return out.reshape(-1), (int(ok.sum()), n_words)


def block_code_rx(
    c: CodingCarrier,
    *,
    code_bits: int,
    data_bits: int,
    parity_masks: list[int],
    correct_single: bool | None = None,
    correct: int | None = None,
    emit: str = "data",
) -> CodingCarrier:
    t = _effective_t(code_bits - data_bits, data_bits, correct_single, correct)
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
    decode: Callable[[np.ndarray], tuple[np.ndarray, tuple[int, int] | None]],
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
    cursors = [w.cursor for w in c.windows]
    bounds = cursors[1:] + [int(bits.size)]
    pieces: list[np.ndarray] = []
    windows: list[Window] = []
    pos = 0
    for w, hi in zip(c.windows, bounds):
        dec, tally = decode(bits[w.cursor : hi])
        if tally is not None:
            tallies.append(tally)
        windows.append(Window(start=pos, cursor=pos))
        pieces.append(dec)
        pos += int(dec.size)
    joined = np.concatenate(pieces) if pieces else np.zeros(0, np.uint8)
    return CodingCarrier(
        bits=joined, windows=windows, stats=_word_stats(tallies, chance_word_rate)
    )


def _rs_decode_words(
    bits: np.ndarray,
    codec: _rs.RSCodec,
    symbol_bits: int,
    n: int,
    k: int,
    emit: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    syms = _unpack_symbols(bits, symbol_bits)
    out: list[int] = []
    total = syms.size // n
    valid = 0
    for w in range(total):
        word = [int(s) for s in syms[w * n : (w + 1) * n]]
        try:
            data, full, _ = codec.decode(word)
            out.extend(full if emit == "codeword" else data)
            valid += 1
        except _rs.ReedSolomonError:
            # uncorrectable is detectable, never repaired by guessing: emit the
            # received word unchanged so downstream framing keeps its alignment
            out.extend(word if emit == "codeword" else word[:k])
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
    emit: str = "data",
) -> CodingCarrier:
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


def _perm_span(idx: np.ndarray) -> int:
    """Input stride a gather consumes: one past its highest index, NOT its
    length. A dropping gather (one that skips per-slot bits, so it spans more
    input than it emits) reads further than it writes, and assuming
    stride == len(perm) indexes out of bounds. Both scopes must agree on this."""
    return int(idx.max()) + 1 if idx.size else 0


def _permute_block(bits: np.ndarray, idx: np.ndarray, span: int) -> np.ndarray:
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
    windows = [Window(start=w.start, cursor=w.cursor + bit_offset) for w in c.windows]
    return replace(c, bits=np.asarray(c.bits, np.uint8), windows=windows)


def differential_rx(c: CodingCarrier, *, invert: bool = False) -> CodingCarrier:
    b = np.asarray(c.bits, dtype=np.uint8)
    if b.size == 0:
        return replace(c, bits=b)
    prev = np.concatenate(([np.uint8(0)], b[:-1]))
    inv = np.uint8(1 if invert else 0)
    return replace(c, bits=np.bitwise_xor(np.bitwise_xor(b, prev), inv))


def _nibble_swap_bits(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).copy()
    n = (b.size // 8) * 8
    if n:
        g = b[:n].reshape(-1, 8)
        b[:n] = np.concatenate([g[:, 4:], g[:, :4]], axis=1).reshape(-1)
    return b


def nibble_swap_rx(c: CodingCarrier) -> CodingCarrier:
    return replace(c, bits=_nibble_swap_bits(c.bits))


def _seq_bits(sequence: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(sequence), dtype=np.uint8))


_LFSR_CHUNK = 1 << 16


def _lfsr_seq(mask: int, seed: int, length: int, n: int) -> np.ndarray:
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
        acc = 0
        for j in taps:
            m = i - length + j
            acc ^= (1 << (m + length)) if m < 0 else rows[m]
        rows.append(acc)
    rows_arr = np.asarray(rows, np.uint64)
    t = length
    window = seed & ((1 << length) - 1)
    while t < n:
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

    # LFSR mode
    if lfsr_mask is not None:
        assert lfsr_seed is not None and lfsr_len is not None
        if c.windows is None:
            lfsr_seq = _lfsr_seq(lfsr_mask, lfsr_seed, lfsr_len, bits.size)
            return replace(c, bits=np.bitwise_xor(bits, lfsr_seq))
        out = bits.copy()
        cursors = [int(w.cursor) for w in c.windows]
        bounds: list[int] = cursors[1:] + [int(bits.size)]
        spans = [hi - lo for lo, hi in zip(cursors, bounds)]
        max_span = max(spans) if spans else 0
        if max_span > 0:
            lfsr_seq = _lfsr_seq(lfsr_mask, lfsr_seed, lfsr_len, max_span)
            for (lo, hi), span_len in zip(zip(cursors, bounds), spans):
                out[lo:hi] = np.bitwise_xor(bits[lo:hi], lfsr_seq[:span_len])
        return replace(c, bits=out)

    # Sequence mode
    seq = _seq_bits(sequence)
    if seq.size == 0:
        return c
    if c.windows is None:
        return replace(c, bits=np.bitwise_xor(bits, np.resize(seq, bits.size)))
    out = bits.copy()
    cursors = [int(w.cursor) for w in c.windows]
    bounds_seq: list[int] = cursors[1:] + [int(bits.size)]
    for lo, hi in zip(cursors, bounds_seq):
        span: np.ndarray = out[lo:hi]
        out[lo:hi] = np.bitwise_xor(span, np.resize(seq, span.size))
    return replace(c, bits=out)
