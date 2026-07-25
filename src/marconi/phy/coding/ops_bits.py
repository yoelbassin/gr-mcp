from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from marconi.core.coding import can_correct
from marconi.phy.coding.carrier import CodingCarrier, Window


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


def _unpack_symbols(bits: np.ndarray, width: int) -> np.ndarray:
    n = bits.size // width
    g = np.asarray(bits[: n * width], dtype=np.int64).reshape(n, width)
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    return g @ weights


def _pack_symbols(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((values[:, None] >> shifts) & 1).reshape(-1).astype(np.uint8)


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
    correct: bool | None,
    emit: str,
) -> np.ndarray:
    # Codeword basis is LSB-first: stream bit j of a stride is codeword bit j,
    # so parity_masks and the emitted data bits are LSB-first too. The CSS
    # explicit decoder assembles the same block_fec_decode input MSB-first
    # (bits/symbols.py, _fec_nibbles); re-expressing that path on this stage
    # must reconcile the basis (supply masks LSB-first, or reverse each
    # stride).
    n_parity = code_bits - data_bits
    do_correct = can_correct(n_parity, data_bits) if correct is None else correct
    n_words = bits.size // code_bits
    words = bits[: n_words * code_bits].reshape(n_words, code_bits).copy()
    if do_correct and n_words and parity_masks:
        rows = np.array(
            [[(m >> b) & 1 for b in range(data_bits)] for m in parity_masks],
            np.int64,
        )
        data64 = words[:, :data_bits].astype(np.int64)
        syndrome = (data64 @ rows.T + words[:, data_bits:]) & 1
        weights = np.left_shift(1, np.arange(n_parity, dtype=np.int64))
        s_int = syndrome @ weights
        # a column's syndrome signature, data columns then the parity identity;
        # first ascending match flips, matching the scalar column scan
        col_sig = np.concatenate([rows.T @ weights, weights])
        bad = np.flatnonzero(s_int)
        if bad.size:
            match = s_int[bad, None] == col_sig[None, :]
            col = match.argmax(axis=1)
            fixable = match.any(axis=1)
            words[bad[fixable], col[fixable]] ^= 1
    out = words if emit == "codeword" else words[:, :data_bits]
    return out.reshape(-1)


def block_code_rx(
    c: CodingCarrier,
    *,
    code_bits: int,
    data_bits: int,
    parity_masks: list[int],
    correct: bool | None = None,
    emit: str = "data",
) -> CodingCarrier:
    if c.windows is None:
        decoded = _block_decode(
            np.asarray(c.bits, np.uint8),
            code_bits,
            data_bits,
            parity_masks,
            correct,
            emit,
        )
        return CodingCarrier(bits=decoded)
    bits = np.asarray(c.bits, np.uint8)
    cursors = [w.cursor for w in c.windows]
    bounds = cursors[1:] + [int(bits.size)]
    pieces: list[np.ndarray] = []
    windows: list[Window] = []
    pos = 0
    for w, hi in zip(c.windows, bounds):
        window_bits = bits[w.cursor : hi]
        dec = _block_decode(
            window_bits, code_bits, data_bits, parity_masks, correct, emit
        )
        windows.append(Window(start=pos, cursor=pos))
        pieces.append(dec)
        pos += int(dec.size)
    joined = np.concatenate(pieces) if pieces else np.zeros(0, np.uint8)
    return CodingCarrier(bits=joined, windows=windows)


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
