from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np
import reedsolo

from marconi.engine.coding.carrier import CodingCarrier, Window
from marconi.engine.coding.primitives import can_correct


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


def sync_word_rx(c: CodingCarrier, *, sync: str, max_errors: int = 0) -> CodingCarrier:
    """Correlating window seeder: seed a window just past every position where
    the bitstream matches ``sync`` within ``max_errors`` bit flips (a sync word
    is detected by correlation, not equality). Matches are taken
    non-overlapping."""
    pat = bytes_to_bits(bytes.fromhex(sync))
    bits = np.asarray(c.bits, dtype=np.uint8)
    m = pat.size
    windows: list[Window] = []
    if m == 0 or bits.size < m:
        return CodingCarrier(bits=bits, windows=windows, marks=c.marks)
    # one O(n) mismatch accumulator per pattern bit, never an (n, m) array:
    # at the bits-layer budget an n x m compare is tens of GiB
    size = bits.size - m + 1
    counts = np.zeros(size, np.min_scalar_type(m))
    for j in range(m):
        counts += bits[j : j + size] != pat[j]
    hits = np.flatnonzero(counts <= max_errors)
    reach = 0
    for i in hits:
        if i < reach:
            continue
        start = int(i) + m
        windows.append(Window(start=start, cursor=start))
        reach = int(i) + m
    return CodingCarrier(bits=bits, windows=windows, marks=c.marks)


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
) -> CodingCarrier:
    """Fixed-width symbol substitution (a line/block code): each ``code_bits``
    input symbol maps to its ``data_bits`` inverse-table value. One op expresses
    3-of-6, Manchester, PPM chip-pairs — the table and widths are caller data."""
    _, inv = _codebook_maps(code_bits, data_bits, table)
    if symbol_input:
        if c.symbols is None:
            raise ValueError("codebook(symbol_input=True) needs a symbols carrier")
        syms = np.asarray(c.symbols, np.int64)
        bits = (
            _pack_symbols(inv[syms], data_bits) if syms.size else np.zeros(0, np.uint8)
        )
        marks = tuple(int(m) * data_bits for m in c.marks)
        return CodingCarrier(bits=bits, marks=marks)
    if c.windows is not None:

        def _decode(bits: np.ndarray) -> np.ndarray:
            return _pack_symbols(inv[_unpack_symbols(bits, code_bits)], data_bits)

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
    data = inv[_unpack_symbols(np.asarray(c.bits, np.uint8), code_bits)]
    return CodingCarrier(bits=_pack_symbols(data, data_bits), windows=c.windows)


def _block_decode(
    bits: np.ndarray,
    code_bits: int,
    data_bits: int,
    parity_masks: list[int],
    correct_single: bool | None,
    emit: str,
) -> np.ndarray:
    # Codeword basis is LSB-first: stream bit j of a stride is codeword bit j,
    # so parity_masks and the emitted data bits are LSB-first too. The CSS
    # explicit decoder assembles the same block_fec_decode input MSB-first
    # (bits/symbols.py, _fec_nibbles); re-expressing that path on this stage
    # must reconcile the basis (supply masks LSB-first, or reverse each
    # stride).
    n_parity = code_bits - data_bits
    do_correct = (
        can_correct(n_parity, data_bits) if correct_single is None else correct_single
    )
    n_words = bits.size // code_bits
    words = bits[: n_words * code_bits].reshape(n_words, code_bits).copy()
    if do_correct and n_words and parity_masks:
        dtype = _sym_dtype(n_parity)
        rows = [[(m >> b) & 1 for b in range(data_bits)] for m in parity_masks]
        # syndrome stays uint8 (mod-2 XOR of selected data cols + parity col),
        # then packs to a small per-word int - no (n_words, k) int matrix
        s_int: np.ndarray = np.zeros(n_words, dtype)
        for p, row in enumerate(rows):
            col = words[:, data_bits + p].astype(np.uint8)
            for c, bit in enumerate(row):
                if bit:
                    col = col ^ words[:, c]
            s_int = (s_int << 1) | (col & 1)
        # a column's syndrome signature, data columns then the parity identity;
        # first ascending match flips, matching the scalar column scan
        weights = np.left_shift(1, np.arange(n_parity - 1, -1, -1, dtype=dtype))
        col_sig = np.concatenate([np.asarray(rows, dtype).T @ weights, weights])
        bad = np.flatnonzero(s_int)
        if bad.size:
            match = s_int[bad, None] == col_sig[None, :]
            col_idx = match.argmax(axis=1)
            fixable = match.any(axis=1)
            words[bad[fixable], col_idx[fixable]] ^= 1
    out = words if emit == "codeword" else words[:, :data_bits]
    return out.reshape(-1)


def block_code_rx(
    c: CodingCarrier,
    *,
    code_bits: int,
    data_bits: int,
    parity_masks: list[int],
    correct_single: bool | None = None,
    emit: str = "data",
) -> CodingCarrier:
    return _decode_scoped(
        c,
        lambda bits: _block_decode(
            bits, code_bits, data_bits, parity_masks, correct_single, emit
        ),
    )


def _decode_scoped(
    c: CodingCarrier, decode: Callable[[np.ndarray], np.ndarray]
) -> CodingCarrier:
    if c.windows is None:
        return CodingCarrier(bits=decode(np.asarray(c.bits, np.uint8)))
    bits = np.asarray(c.bits, np.uint8)
    cursors = [w.cursor for w in c.windows]
    bounds = cursors[1:] + [int(bits.size)]
    pieces: list[np.ndarray] = []
    windows: list[Window] = []
    pos = 0
    for w, hi in zip(c.windows, bounds):
        dec = decode(bits[w.cursor : hi])
        windows.append(Window(start=pos, cursor=pos))
        pieces.append(dec)
        pos += int(dec.size)
    joined = np.concatenate(pieces) if pieces else np.zeros(0, np.uint8)
    return CodingCarrier(bits=joined, windows=windows)


def _rs_decode_words(
    bits: np.ndarray,
    codec: reedsolo.RSCodec,
    symbol_bits: int,
    n: int,
    k: int,
    emit: str,
) -> np.ndarray:
    syms = _unpack_symbols(bits, symbol_bits)
    out: list[int] = []
    for w in range(syms.size // n):
        word = [int(s) for s in syms[w * n : (w + 1) * n]]
        try:
            data, full, _ = codec.decode(word)
            out.extend(full if emit == "codeword" else data)
        except reedsolo.ReedSolomonError:
            # uncorrectable is detectable, never repaired by guessing: emit the
            # received word unchanged so downstream framing keeps its alignment
            out.extend(word if emit == "codeword" else word[:k])
    return _pack_symbols(np.asarray(out, np.int64), symbol_bits)


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
    codec = reedsolo.RSCodec(
        nsym=n - k,
        nsize=n,
        fcr=fcr,
        prim=prim_poly,
        generator=generator,
        c_exp=symbol_bits,
    )
    return _decode_scoped(
        c, lambda bits: _rs_decode_words(bits, codec, symbol_bits, n, k, emit)
    )


def _conv_tables(rate_inv: int, polys: list[int]) -> tuple[int, int, np.ndarray]:
    m = max(int(p).bit_length() for p in polys) - 1
    regs = np.arange(2 << m, dtype=np.int64)
    outs = np.zeros((regs.size, rate_inv), np.uint8)
    for d, poly in enumerate(polys):
        x = regs & int(poly)
        while x.any():
            outs[:, d] ^= (x & 1).astype(np.uint8)
            x >>= 1
    return m, 1 << m, outs


def _viterbi_frame(
    coded: np.ndarray, rate_inv: int, polys: list[int], n_steps: int, tail: int
) -> np.ndarray:
    # GR fsm(1, n, polys) trellis: reg = (u << m) | state, next = reg >> 1,
    # poly 0 = first output bit; regs (2s, 2s+1) are exactly the two branches
    # into next-state s, so the ACS pairs by reshape.
    m, n_states, outs = _conv_tables(rate_inv, polys)
    obs = coded[: n_steps * rate_inv].reshape(n_steps, rate_inv)
    pm = np.zeros(n_states, np.int64)
    if tail:
        pm += np.iinfo(np.int32).max
        pm[0] = 0
    tb = np.empty((n_steps, n_states), np.uint8)
    src_state = np.arange(2 << m, dtype=np.int64) & (n_states - 1)
    for t in range(n_steps):
        bm = (outs != obs[t][None, :]).sum(axis=1)
        cand = (pm[src_state] + bm).reshape(n_states, 2)
        tb[t] = cand.argmin(axis=1)
        pm = cand[np.arange(n_states), tb[t]]
    state = 0 if tail else int(pm.argmin())
    decoded = np.empty(n_steps, np.uint8)
    for t in range(n_steps - 1, -1, -1):
        reg = 2 * state + int(tb[t, state])
        decoded[t] = reg >> m
        state = reg & (n_states - 1)
    return decoded


def _conv_decode(
    bits: np.ndarray, rate_inv: int, polys: list[int], frame_bits: int, tail: int
) -> np.ndarray:
    n_steps = frame_bits + tail
    need = n_steps * rate_inv
    frames = []
    for f in range(bits.size // need):
        frame = _viterbi_frame(bits[f * need :], rate_inv, polys, n_steps, tail)
        frames.append(frame[:frame_bits])
    return np.concatenate(frames) if frames else np.zeros(0, np.uint8)


def conv_code_rx(
    c: CodingCarrier,
    *,
    rate_inv: int,
    polys: list[int],
    frame_bits: int,
    tail: int = 0,
) -> CodingCarrier:
    return _decode_scoped(
        c, lambda bits: _conv_decode(bits, rate_inv, polys, frame_bits, tail)
    )


def _perm_span(idx: np.ndarray) -> int:
    """Input stride a gather consumes: one past its highest index, NOT its
    length. A dropping gather (one that skips per-slot bits, so it spans more
    input than it emits) reads further than it writes, and assuming
    stride == len(perm) indexes out of bounds. Both scopes must agree on this."""
    return int(idx.max()) + 1 if idx.size else 0


def permute_rx(c: CodingCarrier, *, perm: list[int]) -> CodingCarrier:
    idx = np.asarray(perm, np.int64)
    span = _perm_span(idx)
    if c.windows is None:
        bits = np.asarray(c.bits, np.uint8)
        n = bits.size // span if span else 0
        out = bits[: n * span].reshape(n, span)[:, idx].reshape(-1)
        return CodingCarrier(bits=out)
    bits = np.asarray(c.bits, np.uint8)
    pieces, windows, pos = [], [], 0
    for w in c.windows:
        if w.cursor + span > bits.size:
            continue
        dec = bits[w.cursor + idx]
        windows.append(Window(start=pos, cursor=pos))
        pieces.append(dec)
        pos += int(dec.size)
    joined = np.concatenate(pieces) if pieces else np.zeros(0, np.uint8)
    return CodingCarrier(bits=joined, windows=windows)


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


def descramble_rx(c: CodingCarrier, *, sequence: str = "") -> CodingCarrier:
    seq = _seq_bits(sequence)
    bits = np.asarray(c.bits, dtype=np.uint8)
    if seq.size == 0 or bits.size == 0:
        return c
    if c.windows is None:
        return replace(c, bits=np.bitwise_xor(bits, np.resize(seq, bits.size)))
    out = bits.copy()
    cursors = [w.cursor for w in c.windows]
    bounds = cursors[1:] + [int(bits.size)]
    for lo, hi in zip(cursors, bounds):
        span = out[lo:hi]
        out[lo:hi] = np.bitwise_xor(span, np.resize(seq, span.size))
    return replace(c, bits=out)
