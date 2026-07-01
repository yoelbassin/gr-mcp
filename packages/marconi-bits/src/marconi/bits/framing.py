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
_HDLC_TRAINING = np.array([0, 1] * 12, dtype=np.uint8)
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
        )
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
    n = bits // 6
    return "".join(table[(value >> (6 * (n - 1 - i))) & 0x3F] for i in range(n)).rstrip(
        " "
    )


def _encode_charset(text: str, bits: int, table: str) -> int:
    n = bits // 6
    text = text.ljust(n, " ")[:n]
    value = 0
    for ch in text:
        value = (value << 6) | (table.index(ch) if ch in table else table.index(" "))
    return value


def _apply_fields_rx(message: dict, fields: list[ParseField]) -> dict:
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


def parse_rx(
    c: RxCarrier, *, struct: BitStruct, fields: list[ParseField], bit_order: str = "msb"
) -> RxCarrier:
    need = (sum(f.bits for f in fields) + 7) // 8
    out: list[_Frame] = []
    for f in c.frames:
        if f.crc_ok is not False and len(f.payload) >= need:
            parsed = struct.parse(_lsb(f.payload, bit_order))
            msg = {k: int(v) for k, v in parsed.items() if not k.startswith("_")}
            out.append(replace(f, message=_apply_fields_rx(msg, fields)))
        else:
            out.append(f)
    return RxCarrier(bits=c.bits, frames=out)


def parse_tx(
    c: TxCarrier, *, struct: BitStruct, fields: list[ParseField], bit_order: str = "msb"
) -> TxCarrier:
    out = []
    for m in c.items:
        try:
            built = struct.build(_apply_fields_tx(dict(m), fields))
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
    flags: list[int] = []
    i = 0
    n = len(bits)
    while i <= n - 8:
        if np.array_equal(bits[i : i + 8], _HDLC_FLAG):
            flags.append(i)
            i += 7
        else:
            i += 1
    return flags


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


def hdlc_deframe_tx(c: TxCarrier, *, bit_order: str = "msb") -> TxCarrier:
    framed: list[np.ndarray] = []
    for content in c.items:
        body = _bitstuff(bytes_to_bits(content, bit_order))
        framed.append(np.concatenate([_HDLC_FLAG, body, _HDLC_FLAG]).astype(np.uint8))
    if framed:
        framed[0] = np.concatenate([_HDLC_TRAINING, framed[0]]).astype(np.uint8)
        framed[-1] = np.concatenate([framed[-1], _HDLC_TRAINING]).astype(np.uint8)
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
