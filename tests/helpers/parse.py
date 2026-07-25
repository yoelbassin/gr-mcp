from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from construct import BitsInteger, BitStruct, ConstructError, Padding
from helpers.bitops import bits_to_bytes, bytes_to_bits
from pydantic import BaseModel, model_validator

_BITREV = bytes(int(format(i, "08b")[::-1], 2) for i in range(256))


class ParseField(BaseModel):
    name: str
    bits: int
    signed: bool = False
    charset: str | None = None
    enum: dict[int, str] | None = None
    char_bits: int = 6
    rest: bool = False

    @model_validator(mode="after")
    def _rest_shape(self) -> "ParseField":
        if self.char_bits < 1:
            raise ValueError("char_bits must be >= 1")
        if self.rest and (self.charset is None or self.bits != 0):
            raise ValueError("a rest field needs charset set and bits == 0")
        return self


def _lsb(data: bytes, bit_order: str) -> bytes:
    return data.translate(_BITREV) if bit_order == "lsb" else data


def parse_fields(raw: Iterable[object]) -> list[ParseField]:
    return [ParseField.model_validate(f) for f in raw]


def _fixed_bits(fields: list[ParseField]) -> int:
    return sum(f.bits for f in fields if not f.rest)


def build_struct(fields: list[ParseField]) -> BitStruct:
    fixed = [f for f in fields if not f.rest]
    items: list = [f.name / BitsInteger(f.bits, signed=f.signed) for f in fixed]
    pad = (-_fixed_bits(fields)) % 8
    if pad:
        items.append(Padding(pad))
    return BitStruct(*items)


def _require_charset_table(table: str, char_bits: int) -> None:
    if len(table) < (1 << char_bits):
        raise ValueError(
            f"charset table has {len(table)} entries, need >= {1 << char_bits}"
        )


def _indices_to_text(indices: Iterable[int], table: str) -> str:
    return "".join(table[int(i)] for i in indices).rstrip(" ")


def _charset_indices(text: str, table: str) -> list[int]:
    fill = table.index(" ") if " " in table else 0
    return [table.index(ch) if ch in table else fill for ch in text]


def _decode_charset(value: int, bits: int, table: str, char_bits: int) -> str:
    _require_charset_table(table, char_bits)
    mask = (1 << char_bits) - 1
    n = bits // char_bits
    return _indices_to_text(
        ((value >> (char_bits * (n - 1 - i))) & mask for i in range(n)), table
    )


def _encode_charset(text: str, bits: int, table: str, char_bits: int) -> int:
    if len(table) < (1 << char_bits) or " " not in table:
        raise ValueError("charset table too small or missing a space")
    n = bits // char_bits
    value = 0
    for idx in _charset_indices(text.ljust(n, " ")[:n], table):
        value = (value << char_bits) | idx
    return value


def _apply_fields_rx(
    message: dict[str, int | str], fields: list[ParseField]
) -> dict[str, int | str]:
    for f in fields:
        if f.rest:
            continue
        if f.charset is not None:
            message[f.name] = _decode_charset(
                int(message[f.name]), f.bits, f.charset, f.char_bits
            )
        elif f.enum is not None:
            message[f.name] = f.enum.get(int(message[f.name]), int(message[f.name]))
    return message


def _apply_fields_tx(message: dict, fields: list[ParseField]) -> dict:
    out = dict(message)
    for f in fields:
        if f.rest:
            continue
        if f.charset is not None:
            out[f.name] = _encode_charset(
                str(out[f.name]), f.bits, f.charset, f.char_bits
            )
        elif f.enum is not None:
            rev = {label: code for code, label in f.enum.items()}
            out[f.name] = rev.get(out[f.name], out[f.name])
    return out


_Case = tuple[BitStruct, list[ParseField]]


def _need_bytes(fields: list[ParseField]) -> int:
    return (_fixed_bits(fields) + 7) // 8


def _decode_struct(
    struct: BitStruct, payload: bytes, bit_order: str, fields: list[ParseField]
) -> dict[str, int | str]:
    parsed = struct.parse(_lsb(payload, bit_order))
    msg: dict[str, int | str] = {
        k: int(v) for k, v in parsed.items() if not k.startswith("_")
    }
    return _apply_fields_rx(msg, fields)


def _unpack_symbols(bits: np.ndarray, width: int) -> np.ndarray:
    n = bits.size // width
    g = np.asarray(bits[: n * width], dtype=np.int64).reshape(n, width)
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    return g @ weights


def _pack_symbols(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((values[:, None] >> shifts) & 1).reshape(-1).astype(np.uint8)


def _apply_rest(
    msg: dict[str, int | str],
    payload: bytes,
    fields: list[ParseField],
    bit_order: str,
) -> dict[str, int | str]:
    # Rest recovers trailing chars only up to byte-packing granularity: the char
    # count comes from the payload's *byte* length, so for an externally-produced
    # payload whose true bit length isn't a byte multiple the final partial group
    # is byte-fill, cleanly dropped only when the charset maps that residue to an
    # rstrip-able char. Marconi's own TX byte-aligns (see _append_rest) so its
    # round-trip is exact; a bit-exact rest for foreign payloads needs a
    # carrier-level payload bit-length (tracked follow-up).
    rest = [f for f in fields if f.rest]
    if not rest:
        return msg
    (rf,) = rest
    assert rf.charset is not None  # ParseField validator guarantees this
    _require_charset_table(rf.charset, rf.char_bits)
    fixed_bits = _fixed_bits(fields)
    fbits = bytes_to_bits(_lsb(payload, bit_order))
    n_chars = max(0, fbits.size - fixed_bits) // rf.char_bits
    if n_chars == 0:
        msg[rf.name] = ""
        return msg
    span = fbits[fixed_bits : fixed_bits + n_chars * rf.char_bits]
    msg[rf.name] = _indices_to_text(_unpack_symbols(span, rf.char_bits), rf.charset)
    return msg


def _rest_char_count(fixed_bits: int, char_bits: int, text_len: int) -> int:
    # Smallest char count >= text_len that byte-aligns the payload, so packbits
    # adds no zero-fill and RX reads back exactly these chars (the spare ones are
    # spaces, which rstrip drops -> exact round-trip). No aligned count exists for
    # odd fixed_bits with even char_bits; then emit the raw text and accept the
    # byte-fill approximation rather than corrupt it.
    for n in range(text_len, text_len + 8):
        if (fixed_bits + n * char_bits) % 8 == 0:
            return n
    return text_len


def _append_rest(built: bytes, message: dict, fields: list[ParseField]) -> bytes:
    rest = [f for f in fields if f.rest]
    if not rest:
        return built
    (rf,) = rest
    assert rf.charset is not None  # ParseField validator guarantees this
    fixed_bits = _fixed_bits(fields)
    fixed = bytes_to_bits(built)[:fixed_bits]
    text = str(message[rf.name])
    n = len(text)
    if " " in rf.charset:  # space-pad to a byte boundary only if a space exists
        n = _rest_char_count(fixed_bits, rf.char_bits, len(text))
    if n == 0:
        return bits_to_bytes(fixed)
    idxs = np.array(
        _charset_indices(text.ljust(n, " ")[:n], rf.charset), dtype=np.int64
    )
    rest_bits = _pack_symbols(idxs, rf.char_bits)
    return bits_to_bytes(np.concatenate([fixed, rest_bits]).astype(np.uint8))


def _reject_misplaced_rest(fields: list[ParseField]) -> None:
    positions = [i for i, f in enumerate(fields) if f.rest]
    if positions and positions != [len(fields) - 1]:
        raise ValueError("only one rest field is allowed and it must be last")


def _build_cases(
    fields: list[ParseField], cases: list[dict] | None
) -> dict[int, _Case]:
    out: dict[int, _Case] = {}
    for case in cases or []:
        case_fields = parse_fields(case["fields"])
        if any(f.rest for f in case_fields):
            raise ValueError("rest fields are not allowed inside cases")
        body = fields + case_fields
        _reject_misplaced_rest(body)
        out[int(case["when"])] = (build_struct(body), body)
    return out


def parse_message(
    payload: bytes,
    fields: list[dict],
    *,
    bit_order: str = "msb",
    discriminator: str | None = None,
    cases: list[dict] | None = None,
) -> dict[str, int | str] | None:
    pf = parse_fields(fields)
    _reject_misplaced_rest(pf)
    case_map = _build_cases(pf, cases)
    if len(payload) < _need_bytes(pf):
        return None
    struct = build_struct(pf)
    sel_fields = pf
    try:
        msg = _decode_struct(struct, payload, bit_order, pf)
        # A per-type body is applied only when its discriminator matches AND
        # the payload can hold it; otherwise the shared header stands alone —
        # a non-matching type is never re-read with a wrong-type struct.
        if discriminator is not None and case_map:
            selected = case_map.get(int(msg.get(discriminator, -1)))
            if selected is not None and len(payload) >= _need_bytes(selected[1]):
                msg = _decode_struct(selected[0], payload, bit_order, selected[1])
                sel_fields = selected[1]
    except (ConstructError, KeyError, TypeError):
        return None
    return _apply_rest(msg, payload, sel_fields, bit_order)


def build_message(
    message: dict, fields: list[dict], *, bit_order: str = "msb"
) -> bytes:
    pf = parse_fields(fields)
    _reject_misplaced_rest(pf)
    struct = build_struct(pf)
    try:
        built = struct.build(_apply_fields_tx(dict(message), pf))
        built = _append_rest(built, message, pf)
    except (ConstructError, KeyError, TypeError) as e:
        raise ValueError(f"message {message} does not fit the parse fields: {e}") from e
    return _lsb(built, bit_order)
