from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from functools import lru_cache
from typing import Literal

import numpy as np
from construct import BitsInteger, BitStruct, ConstructError, Padding
from crc import Calculator, Configuration

from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame
from marconi.bits.models import ParseField

_HDLC_FLAG = np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8)
_BITREV = bytes(int(format(i, "08b")[::-1], 2) for i in range(256))


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


def _lsb(data: bytes, bit_order: str) -> bytes:
    return data.translate(_BITREV) if bit_order == "lsb" else data


@lru_cache(maxsize=None)
def _calculator(
    poly: int, bits: int, init: int, reflected: bool, xorout: int
) -> Calculator:
    return Calculator(
        Configuration(
            width=bits,
            polynomial=poly,
            init_value=init,
            final_xor_value=xorout,
            reverse_input=reflected,
            reverse_output=reflected,
        ),
        optimized=True,  # table engine (~2.1 MB/s) not the bit-by-bit one (~0.30)
    )


def crc_value(
    data: bytes,
    *,
    poly: int,
    bits: int,
    init: int,
    reflected: bool = False,
    xorout: int = 0,
) -> int:
    return _calculator(poly, bits, init, reflected, xorout).checksum(bytes(data))


def _fold(
    body: bytes,
    *,
    poly: int,
    bits: int,
    init: int,
    reflected: bool,
    xorout: int,
    fold_tail: int,
) -> int:
    if fold_tail <= 0:
        return crc_value(
            body, poly=poly, bits=bits, init=init, reflected=reflected, xorout=xorout
        )
    val = crc_value(
        body[:-fold_tail],
        poly=poly,
        bits=bits,
        init=init,
        reflected=reflected,
        xorout=xorout,
    )
    return (val ^ int.from_bytes(body[-fold_tail:], "big")) & ((1 << bits) - 1)


def crc_rx(
    c: RxCarrier,
    *,
    poly: int,
    bits: int,
    init: int,
    reflected: bool = False,
    xorout: int = 0,
    bit_order: str = "msb",
    fold_tail: int = 0,
    checksum_le: bool = False,
) -> RxCarrier:
    """Trailing check: the last `bits` of each frame payload are the FCS."""
    endian = "little" if checksum_le else _bitorder(bit_order)
    n = bits // 8
    out: list[_Frame] = []
    for f in c.frames:
        if len(f.payload) < n:
            continue
        body, checksum = f.payload[:-n], f.payload[-n:]
        rx = int.from_bytes(checksum, endian)
        crc_ok = (
            _fold(
                body,
                poly=poly,
                bits=bits,
                init=init,
                reflected=reflected,
                xorout=xorout,
                fold_tail=fold_tail,
            )
            == rx
        )
        out.append(replace(f, crc_ok=crc_ok, payload=body))
    return RxCarrier(bits=c.bits, frames=out)


def crc_tx(
    c: TxCarrier,
    *,
    poly: int,
    bits: int,
    init: int,
    reflected: bool = False,
    xorout: int = 0,
    bit_order: str = "msb",
    fold_tail: int = 0,
    checksum_le: bool = False,
) -> TxCarrier:
    endian = "little" if checksum_le else _bitorder(bit_order)
    n = bits // 8
    return TxCarrier(
        [
            p
            + _fold(
                p,
                poly=poly,
                bits=bits,
                init=init,
                reflected=reflected,
                xorout=xorout,
                fold_tail=fold_tail,
            ).to_bytes(n, endian)
            for p in c.items
        ]
    )


def parse_fields(raw: Iterable[object]) -> list[ParseField]:
    return [ParseField.model_validate(f) for f in raw]


def build_struct(fields: list[ParseField]) -> BitStruct:
    items: list = [f.name / BitsInteger(f.bits, signed=f.signed) for f in fields]
    pad = (-sum(f.bits for f in fields)) % 8
    if pad:
        items.append(Padding(pad))
    return BitStruct(*items)


def _decode_charset(value: int, bits: int, table: str) -> str:
    if len(table) < 64:
        raise ValueError(f"charset table has {len(table)} entries, need >= 64")
    n = bits // 6
    return "".join(table[(value >> (6 * (n - 1 - i))) & 0x3F] for i in range(n)).rstrip(
        " "
    )


def _encode_charset(text: str, bits: int, table: str) -> int:
    if len(table) < 64 or " " not in table:
        raise ValueError("charset table needs >= 64 entries including a space")
    n = bits // 6
    text = text.ljust(n, " ")[:n]
    value = 0
    for ch in text:
        value = (value << 6) | (table.index(ch) if ch in table else table.index(" "))
    return value


def _apply_fields_rx(
    message: dict[str, int | str], fields: list[ParseField]
) -> dict[str, int | str]:
    for f in fields:
        if f.charset is not None:
            message[f.name] = _decode_charset(int(message[f.name]), f.bits, f.charset)
        elif f.enum is not None:
            message[f.name] = f.enum.get(int(message[f.name]), int(message[f.name]))
    return message


def _apply_fields_tx(message: dict, fields: list[ParseField]) -> dict:
    out = dict(message)
    for f in fields:
        if f.charset is not None:
            out[f.name] = _encode_charset(str(out[f.name]), f.bits, f.charset)
        elif f.enum is not None:
            rev = {label: code for code, label in f.enum.items()}
            out[f.name] = rev.get(out[f.name], out[f.name])
    return out


_Case = tuple[BitStruct, list[ParseField]]


def _need_bytes(fields: list[ParseField]) -> int:
    return (sum(f.bits for f in fields) + 7) // 8


def _decode_struct(
    struct: BitStruct, payload: bytes, bit_order: str, fields: list[ParseField]
) -> dict[str, int | str]:
    parsed = struct.parse(_lsb(payload, bit_order))
    msg: dict[str, int | str] = {
        k: int(v) for k, v in parsed.items() if not k.startswith("_")
    }
    return _apply_fields_rx(msg, fields)


def parse_rx(
    c: RxCarrier,
    *,
    struct: BitStruct,
    fields: list[ParseField],
    bit_order: str = "msb",
    discriminator: str | None = None,
    cases: dict[int, _Case] | None = None,
) -> RxCarrier:
    out: list[_Frame] = []
    for f in c.frames:
        if f.crc_ok is False or len(f.payload) < _need_bytes(fields):
            out.append(f)
            continue
        msg = _decode_struct(struct, f.payload, bit_order, fields)
        # A per-type body is applied only when its discriminator matches AND the
        # payload can hold it; otherwise the shared header stands alone — a
        # non-matching type is never re-read with a wrong-type struct.
        if discriminator is not None and cases:
            selected = cases.get(int(msg.get(discriminator, -1)))
            if selected is not None and len(f.payload) >= _need_bytes(selected[1]):
                msg = _decode_struct(selected[0], f.payload, bit_order, selected[1])
        out.append(replace(f, message=msg))
    return RxCarrier(bits=c.bits, frames=out)


def parse_tx(
    c: TxCarrier,
    *,
    struct: BitStruct,
    fields: list[ParseField],
    bit_order: str = "msb",
    discriminator: str | None = None,
    cases: dict[int, _Case] | None = None,
) -> TxCarrier:
    out = []
    for m in c.items:
        sel_struct, sel_fields = struct, fields
        if discriminator is not None and cases and discriminator in m:
            case = cases.get(int(m[discriminator]))
            if case is not None:
                sel_struct, sel_fields = case
        try:
            built = sel_struct.build(_apply_fields_tx(dict(m), sel_fields))
        except (ConstructError, KeyError, TypeError) as e:
            raise ValueError(f"message {m} does not fit the parse fields: {e}") from e
        out.append(_lsb(built, bit_order))
    return TxCarrier(out)


def segment_rx(c: RxCarrier, *, frame_body_len: int) -> RxCarrier:
    frames: list[_Frame] = []
    i = 0
    while i + frame_body_len <= len(c.bits):
        frames.append(_Frame(start=i, cursor=i))
        i += frame_body_len
    return RxCarrier(bits=c.bits, frames=frames)


def segment_tx(c: TxCarrier) -> TxCarrier:
    return c


def fixed_frame_rx(c: RxCarrier, *, payload_bits: int) -> RxCarrier:
    out: list[_Frame] = []
    for f in c.frames:
        if f.cursor + payload_bits <= len(c.bits):
            payload = bits_to_bytes(c.bits[f.cursor : f.cursor + payload_bits])
            out.append(replace(f, payload=payload, cursor=f.cursor + payload_bits))
    return RxCarrier(bits=c.bits, frames=out)


def fixed_frame_tx(c: TxCarrier, *, payload_bits: int) -> TxCarrier:
    return TxCarrier([bytes_to_bits(b) for b in c.items])


def _bitstuff(bits: np.ndarray) -> np.ndarray:
    out: list[int] = []
    ones = 0
    for bit in bits:
        out.append(int(bit))
        if bit:
            ones += 1
            if ones == 5:
                out.append(0)
                ones = 0
        else:
            ones = 0
    return np.array(out, dtype=np.uint8)


def _destuff(bits: np.ndarray) -> np.ndarray | None:
    out: list[int] = []
    ones = 0
    i = 0
    n = len(bits)
    while i < n:
        bit = int(bits[i])
        out.append(bit)
        if bit:
            ones += 1
            if ones == 5:
                if i + 1 >= n or int(bits[i + 1]) == 1:
                    return None
                i += 1
                ones = 0
        else:
            ones = 0
        i += 1
    return np.array(out, dtype=np.uint8)


def _find_flags(bits: np.ndarray) -> list[int]:
    # A flag 01111110 is a 0, six 1s (a length-6 window summing to 6), then a 0.
    # One cumsum + slice compares finds every one in a single pass -- no per-bit
    # loop, no O(8n) window materialization. The 0-bookends cannot fall inside
    # another flag's ones-run, so matches are always >=7 apart, exactly what the
    # old greedy i+=7 scan reported. Kept in the GR-free bits layer; stock
    # digital.hdlc_deframer_bp would drag GR into it for no throughput need.
    n = bits.size
    if n < 8:
        return []
    cs = np.concatenate(([0], np.cumsum(bits, dtype=np.int32)))
    six_ones = (cs[7:n] - cs[1 : n - 6]) == 6
    match = six_ones & (bits[0 : n - 7] == 0) & (bits[7:n] == 0)
    return np.flatnonzero(match).tolist()


def hdlc_deframe_rx(c: RxCarrier, *, bit_order: str = "msb") -> RxCarrier:
    bits = np.asarray(c.bits, dtype=np.uint8)
    flags = _find_flags(bits)
    frames: list[_Frame] = []
    off = 0
    for a, b in zip(flags, flags[1:]):
        region = bits[a + 8 : b]
        if region.size == 0:
            continue
        content = _destuff(region)
        if content is None or content.size == 0 or content.size % 8 != 0:
            continue
        frames.append(
            _Frame(
                start=off,
                cursor=off + content.size,
                payload=bits_to_bytes(content, bit_order),
            )
        )
        off += content.size
    return RxCarrier(bits=bits, frames=frames)


def hdlc_deframe_tx(
    c: TxCarrier, *, bit_order: str = "msb", training: str = ""
) -> TxCarrier:
    """Flag-delimit each frame with bit-stuffing. ``training`` is an optional
    bit-string ("0101...") wrapped around the whole burst (a link's preamble /
    ramp-up sequence); empty by default — the sequence belongs to a protocol,
    not to HDLC."""
    train = np.array([int(ch) for ch in training], dtype=np.uint8)
    framed: list[np.ndarray] = []
    for content in c.items:
        body = _bitstuff(bytes_to_bits(content, bit_order))
        framed.append(np.concatenate([_HDLC_FLAG, body, _HDLC_FLAG]).astype(np.uint8))
    if framed and train.size:
        framed[0] = np.concatenate([train, framed[0]]).astype(np.uint8)
        framed[-1] = np.concatenate([framed[-1], train]).astype(np.uint8)
    return TxCarrier(framed)


def _seq_bits(sequence: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(sequence), dtype=np.uint8))


def descramble_bits_rx(c: RxCarrier, *, sequence: str = "") -> RxCarrier:
    seq = _seq_bits(sequence)
    bits = np.asarray(c.bits, dtype=np.uint8)
    if seq.size == 0 or bits.size == 0:
        return c
    return RxCarrier(
        bits=np.bitwise_xor(bits, np.resize(seq, bits.size)), frames=c.frames
    )


def descramble_bits_tx(c: TxCarrier, *, sequence: str = "") -> TxCarrier:
    if not c.items:
        return c
    seq = _seq_bits(sequence)
    stream = np.concatenate([np.asarray(it, dtype=np.uint8) for it in c.items])
    if seq.size == 0 or stream.size == 0:
        return c
    return TxCarrier(items=[np.bitwise_xor(stream, np.resize(seq, stream.size))])


def _xor_seq(data: bytes, seq: bytes) -> bytes:
    if len(seq) < len(data):
        raise ValueError(
            f"descramble sequence ({len(seq)} bytes) shorter than payload "
            f"({len(data)} bytes)"
        )
    return bytes(b ^ seq[i] for i, b in enumerate(data))


def descramble_rx(c: RxCarrier, *, sequence: str) -> RxCarrier:
    seq = bytes.fromhex(sequence)
    frames = [replace(f, payload=_xor_seq(f.payload, seq)) for f in c.frames]
    return RxCarrier(bits=c.bits, frames=frames)


def descramble_tx(c: TxCarrier, *, sequence: str) -> TxCarrier:
    seq = bytes.fromhex(sequence)
    return TxCarrier([_xor_seq(it, seq) for it in c.items])


def _nibble_swap_bits(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).copy()
    n = (b.size // 8) * 8
    if n:
        g = b[:n].reshape(-1, 8)
        b[:n] = np.concatenate([g[:, 4:], g[:, :4]], axis=1).reshape(-1)
    return b


def nibble_swap_rx(c: RxCarrier) -> RxCarrier:
    return RxCarrier(bits=_nibble_swap_bits(c.bits), frames=c.frames)


def nibble_swap_tx(c: TxCarrier) -> TxCarrier:
    return TxCarrier([_nibble_swap_bits(it) for it in c.items])


def realign_rx(c: RxCarrier, *, bit_offset: int) -> RxCarrier:
    if c.frames:
        raise ValueError("realign must run before any frame seeder")
    return RxCarrier(bits=np.asarray(c.bits, np.uint8)[bit_offset:], frames=[])


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
    c: RxCarrier, *, code_bits: int, data_bits: int, table: list[int]
) -> RxCarrier:
    """Fixed-width symbol substitution (a line/block code): each ``code_bits``
    input symbol maps to its ``data_bits`` inverse-table value. One op expresses
    3-of-6, Manchester, PPM chip-pairs — the table and widths are caller data."""
    _, inv = _codebook_maps(code_bits, data_bits, table)
    data = inv[_unpack_symbols(np.asarray(c.bits, np.uint8), code_bits)]
    return RxCarrier(bits=_pack_symbols(data, data_bits), frames=c.frames)


def codebook_tx(
    c: TxCarrier, *, code_bits: int, data_bits: int, table: list[int]
) -> TxCarrier:
    fwd, _ = _codebook_maps(code_bits, data_bits, table)
    out = [
        _pack_symbols(
            fwd[_unpack_symbols(np.asarray(it, np.uint8), data_bits)], code_bits
        )
        for it in c.items
    ]
    return TxCarrier(out)


def frame_codebook_rx(
    c: RxCarrier, *, code_bits: int, data_bits: int, table: list[int]
) -> RxCarrier:
    _, inv = _codebook_maps(code_bits, data_bits, table)

    def _decode(bits: np.ndarray) -> np.ndarray:
        return _pack_symbols(inv[_unpack_symbols(bits, code_bits)], data_bits)

    if any(len(f.payload) for f in c.frames):
        done = [
            replace(f, payload=bits_to_bytes(_decode(bytes_to_bits(f.payload))))
            for f in c.frames
        ]
        return RxCarrier(bits=c.bits, frames=done)
    bits = np.asarray(c.bits, np.uint8)
    cursors = [f.cursor for f in c.frames]
    if any(b <= a for a, b in zip(cursors, cursors[1:])):
        raise ValueError("frame_codebook needs strictly increasing frame cursors")
    pieces: list[np.ndarray] = []
    frames: list[_Frame] = []
    pos = 0
    for f, hi in zip(c.frames, cursors[1:] + [int(bits.size)]):
        dec = _decode(bits[f.cursor : hi])
        frames.append(replace(f, start=pos, cursor=pos))
        pieces.append(dec)
        pos += int(dec.size)
    joined = np.concatenate(pieces) if pieces else np.zeros(0, np.uint8)
    return RxCarrier(bits=joined, frames=frames)


def frame_codebook_tx(
    c: TxCarrier, *, code_bits: int, data_bits: int, table: list[int]
) -> TxCarrier:
    fwd, _ = _codebook_maps(code_bits, data_bits, table)
    out: list[np.ndarray] = []
    for it in c.items:
        bits = it if isinstance(it, np.ndarray) else bytes_to_bits(bytes(it))
        enc = _pack_symbols(
            fwd[_unpack_symbols(np.asarray(bits, np.uint8), data_bits)], code_bits
        )
        out.append(enc)
    return TxCarrier(out)


def _read_uint(bits: np.ndarray, start: int, width: int) -> int:
    field = bits[start : start + width]
    return int(field.dot(1 << np.arange(width - 1, -1, -1, dtype=np.int64)))


def length_frame_rx(
    c: RxCarrier,
    *,
    length_bits: int,
    base_bytes: int,
    unit_bytes: int,
    bit_order: str = "msb",
) -> RxCarrier:
    """Carve each seeded frame's body using a length field it carries: read a
    ``length_bits`` integer L at the cursor, then take ``base_bytes + unit_bytes*L``
    bytes as the payload (the length field and any trailing checkword included).
    Closes the header->length plumbing gap: a decoded length drives framing
    instead of a hardcoded stride."""
    bits = np.asarray(c.bits, dtype=np.uint8)
    out: list[_Frame] = []
    for f in c.frames:
        if f.cursor + length_bits > bits.size:
            continue
        length = _read_uint(bits, f.cursor, length_bits)
        nbits = (base_bytes + unit_bytes * length) * 8
        if nbits <= 0 or f.cursor + nbits > bits.size:
            continue
        payload = bits_to_bytes(bits[f.cursor : f.cursor + nbits], bit_order)
        out.append(replace(f, payload=payload, cursor=f.cursor + nbits))
    return RxCarrier(bits=bits, frames=out)


def length_frame_tx(
    c: TxCarrier,
    *,
    length_bits: int,
    base_bytes: int,
    unit_bytes: int,
    bit_order: str = "msb",
) -> TxCarrier:
    return c  # the length field is assembled upstream (parse/crc); nothing to add


def delimiter_frame_rx(
    c: RxCarrier, *, delimiter: str, tail_bits: int = 0
) -> RxCarrier:
    """Carve each seeded frame's body up to the first occurrence of a delimiter
    bit-pattern, inclusive, plus ``tail_bits`` trailing bits. Frames with no
    in-bounds delimiter or whose tail runs past stream end are dropped."""
    pat = bytes_to_bits(bytes.fromhex(delimiter))
    bits = np.asarray(c.bits, dtype=np.uint8)
    m = pat.size
    out: list[_Frame] = []
    for f in c.frames:
        seg = bits[f.cursor :]
        if seg.size < m:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(seg, m)
        hits = np.flatnonzero((windows != pat).sum(axis=1) == 0)
        if hits.size == 0:
            continue
        end = f.cursor + int(hits[0]) + m + tail_bits
        if end > bits.size:
            continue
        out.append(replace(f, payload=bits_to_bytes(bits[f.cursor : end]), cursor=end))
    return RxCarrier(bits=bits, frames=out)


def sync_word_rx(c: RxCarrier, *, sync: str, max_errors: int = 0) -> RxCarrier:
    """Correlating frame seeder: seed a frame just past every position where the
    bitstream matches ``sync`` within ``max_errors`` bit flips (a sync word is
    detected by correlation, not equality). Matches are taken non-overlapping."""
    pat = bytes_to_bits(bytes.fromhex(sync))
    bits = np.asarray(c.bits, dtype=np.uint8)
    m = pat.size
    frames: list[_Frame] = []
    if m == 0 or bits.size < m:
        return RxCarrier(bits=bits, frames=frames)
    windows = np.lib.stride_tricks.sliding_window_view(bits, m)
    hits = np.flatnonzero((windows != pat).sum(axis=1) <= max_errors)
    reach = 0
    for i in hits:
        if i < reach:
            continue
        start = int(i) + m
        frames.append(_Frame(start=start, cursor=start))
        reach = int(i) + m
    return RxCarrier(bits=bits, frames=frames)


def sync_word_tx(c: TxCarrier, *, sync: str, max_errors: int = 0) -> TxCarrier:
    pat = bytes_to_bits(bytes.fromhex(sync))
    out: list[np.ndarray] = []
    for it in c.items:
        body = it if isinstance(it, np.ndarray) else bytes_to_bits(bytes(it))
        out.append(np.concatenate([pat, np.asarray(body, np.uint8)]).astype(np.uint8))
    return TxCarrier(out)


def differential_rx(c: RxCarrier, *, invert: bool = False) -> RxCarrier:
    b = np.asarray(c.bits, dtype=np.uint8)
    if b.size == 0:
        return RxCarrier(bits=b, frames=c.frames)
    prev = np.concatenate(([np.uint8(0)], b[:-1]))
    inv = np.uint8(1 if invert else 0)
    return RxCarrier(bits=np.bitwise_xor(np.bitwise_xor(b, prev), inv), frames=c.frames)


def differential_tx(c: TxCarrier, *, invert: bool = False) -> TxCarrier:
    if not c.items:
        return c
    stream = np.concatenate([np.asarray(it, dtype=np.uint8) for it in c.items])
    inv = np.uint8(1 if invert else 0)
    enc = np.bitwise_xor.accumulate(np.bitwise_xor(stream, inv).astype(np.uint8))
    return TxCarrier(items=[enc])
