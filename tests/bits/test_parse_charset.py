from __future__ import annotations

import numpy as np
import pytest

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame
from marconi.bits.models import ParseField
from marconi.bits.stages.message import Parse

_ASCII7 = "".join(chr(i) if 32 <= i < 127 else " " for i in range(128))
# index 0 is '@' (NOT space): a phantom trailing char cannot hide behind rstrip
_AIS6 = "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_ !\"#$%&'()*+,-./0123456789:;<=>?"


def _parse(payload: bytes, fields: list[dict]) -> dict:
    pf = framing.parse_fields(fields)
    fixed = [f for f in pf if not f.rest]
    out = framing.parse_rx(
        RxCarrier(
            bits=np.zeros(0, np.uint8),
            frames=[_Frame(start=0, cursor=0, payload=payload, crc_ok=True)],
        ),
        struct=framing.build_struct(fixed),
        fields=pf,
        bit_order="msb",
        discriminator=None,
        cases=None,
    )
    assert out.frames[0].message is not None
    return out.frames[0].message


def test_seven_bit_charset_field() -> None:
    bits = np.concatenate(
        [[(ord(ch) >> (6 - j)) & 1 for j in range(7)] for ch in "AB"]
    ).astype(np.uint8)
    payload = framing.bits_to_bytes(bits)
    msg = _parse(
        payload,
        [{"name": "tag", "bits": 14, "charset": _ASCII7, "char_bits": 7}],
    )
    assert msg["tag"] == "AB"


def test_rest_consumes_remaining_payload() -> None:
    head = [(0xA5 >> (7 - j)) & 1 for j in range(8)]
    text = np.concatenate(
        [[(ord(ch) >> (6 - j)) & 1 for j in range(7)] for ch in "HELLO"]
    )
    payload = framing.bits_to_bytes(np.concatenate([head, text]).astype(np.uint8))
    msg = _parse(
        payload,
        [
            {"name": "kind", "bits": 8},
            {
                "name": "text",
                "bits": 0,
                "charset": _ASCII7,
                "char_bits": 7,
                "rest": True,
            },
        ],
    )
    assert msg["kind"] == 0xA5
    assert msg["text"] == "HELLO"


def test_default_six_bit_charset_unchanged() -> None:
    f = ParseField(
        name="x",
        bits=12,
        charset="@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_ !\"#$%&'()*+,-./0123456789:;<=>?",
    )
    assert f.char_bits == 6 and f.rest is False


def _rest_roundtrip(charset: str, char_bits: int, header_bits: int, msg: dict) -> dict:
    fields = [
        {"name": "kind", "bits": header_bits},
        {
            "name": "text",
            "bits": 0,
            "charset": charset,
            "char_bits": char_bits,
            "rest": True,
        },
    ]
    pf = framing.parse_fields(fields)
    struct = framing.build_struct(pf)
    tx = framing.parse_tx(TxCarrier([msg]), struct=struct, fields=pf, bit_order="msb")
    frame = _Frame(start=0, cursor=0, payload=tx.items[0], crc_ok=True)
    rx = framing.parse_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), frames=[frame]),
        struct=struct,
        fields=pf,
        bit_order="msb",
        discriminator=None,
        cases=None,
    )
    return dict(rx.frames[0].message or {})


def test_rest_round_trips_through_tx_then_rx() -> None:
    msg = {"kind": 0x7E, "text": "MARCONI!"}
    assert _rest_roundtrip(_ASCII7, 7, 8, msg) == msg


# The byte-packing pad is 0..7 bits; whenever it is >= char_bits the naive
# boundary formula would append a phantom char (e.g. cb=6 lengths 3 and 7 with a
# non-space index-0 charset). TX byte-aligns so RX recovers EVERY length exactly.
@pytest.mark.parametrize("charset,char_bits", [(_ASCII7, 7), (_AIS6, 6)])
def test_rest_round_trips_exactly_across_lengths(charset: str, char_bits: int) -> None:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # in both charsets, no trailing space
    for length in range(1, 11):
        text = alphabet[:length]
        msg = {"kind": 0x5A, "text": text}
        got = _rest_roundtrip(charset, char_bits, 8, msg)
        assert got == msg, f"char_bits={char_bits} length={length}: {got} != {msg}"


def test_rest_rejected_when_a_case_appends_fields_after_it() -> None:
    # rest is "last" in the base list but NOT last once a case appends fields:
    # the merged body must be re-checked so the rest never absorbs case fields.
    with pytest.raises(ValueError, match="must be last"):
        Parse()._kw(
            {
                "fields": [
                    {"name": "kind", "bits": 8},
                    {
                        "name": "text",
                        "bits": 0,
                        "charset": _ASCII7,
                        "char_bits": 7,
                        "rest": True,
                    },
                ],
                "discriminator": "kind",
                "cases": [{"when": 1, "fields": [{"name": "extra", "bits": 4}]}],
            }
        )


def test_rest_rejected_inside_a_case_body() -> None:
    with pytest.raises(ValueError, match="not allowed inside cases"):
        Parse()._kw(
            {
                "fields": [{"name": "kind", "bits": 8}],
                "discriminator": "kind",
                "cases": [
                    {
                        "when": 1,
                        "fields": [
                            {
                                "name": "t",
                                "bits": 0,
                                "charset": _ASCII7,
                                "char_bits": 7,
                                "rest": True,
                            },
                        ],
                    },
                ],
            }
        )
