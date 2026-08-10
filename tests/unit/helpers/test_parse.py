from __future__ import annotations

import numpy as np
import pytest
from helpers import bitops, parse

_ASCII7 = "".join(chr(i) if 32 <= i < 127 else " " for i in range(128))
# index 0 is '@' (NOT space): a phantom trailing char cannot hide behind rstrip
_AIS6 = "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_ !\"#$%&'()*+,-./0123456789:;<=>?"


def test_seven_bit_charset_field() -> None:
    bits = np.concatenate(
        [[(ord(ch) >> (6 - j)) & 1 for j in range(7)] for ch in "AB"]
    ).astype(np.uint8)
    payload = bitops.bits_to_bytes(bits)
    msg = parse.parse_message(
        payload, [{"name": "tag", "bits": 14, "charset": _ASCII7, "char_bits": 7}]
    )
    assert msg is not None
    assert msg["tag"] == "AB"


def test_rest_consumes_remaining_payload() -> None:
    head = [(0xA5 >> (7 - j)) & 1 for j in range(8)]
    text = np.concatenate(
        [[(ord(ch) >> (6 - j)) & 1 for j in range(7)] for ch in "HELLO"]
    )
    payload = bitops.bits_to_bytes(np.concatenate([head, text]).astype(np.uint8))
    msg = parse.parse_message(
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
    assert msg is not None
    assert msg["kind"] == 0xA5
    assert msg["text"] == "HELLO"


def test_default_six_bit_charset_unchanged() -> None:
    f = parse.ParseField(
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
    payload = parse.build_message(msg, fields)
    out = parse.parse_message(payload, fields)
    assert out is not None
    return out


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
        parse.parse_message(
            b"",
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
            discriminator="kind",
            cases=[{"when": 1, "fields": [{"name": "extra", "bits": 4}]}],
        )


def test_rest_rejected_inside_a_case_body() -> None:
    with pytest.raises(ValueError, match="not allowed inside cases"):
        parse.parse_message(
            b"",
            [{"name": "kind", "bits": 8}],
            discriminator="kind",
            cases=[
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
        )


def test_decode_charset_rejects_short_table() -> None:
    with pytest.raises(ValueError):
        parse._decode_charset(0, 6, "ABC", 6)


def test_parse_roundtrip_signed() -> None:
    fields = [
        {"name": "a", "bits": 6},
        {"name": "b", "bits": 28, "signed": True},
        {"name": "c", "bits": 2},
    ]
    msg = {"a": 5, "b": -1234, "c": 1}
    built = parse.build_message(msg, fields)
    out = parse.parse_message(built, fields)
    assert out == msg


# parse dispatches on a discriminator field: different message
# types carry different field layouts, so the wrong-type struct must never be
# applied. Covers AIS msg_type (types 1/2/3 vs 5 vs others) and the general
# mechanism (which also un-blobs DAB's messages rung).
_COMMON = [{"name": "kind", "bits": 4}, {"name": "id", "bits": 12}]
_TYPE1_FIELDS = [{"name": "speed", "bits": 10}, {"name": "course", "bits": 12}]
_TYPE5_FIELDS = [{"name": "callsign", "bits": 42}, {"name": "draught", "bits": 8}]
_CASES = [
    {"when": 1, "fields": _TYPE1_FIELDS},
    {"when": 5, "fields": _TYPE5_FIELDS},
]


def test_type1_parses_its_own_body() -> None:
    msg = {"kind": 1, "id": 7, "speed": 300, "course": 900}
    payload = parse.build_message(msg, _COMMON + _TYPE1_FIELDS)
    out = parse.parse_message(payload, _COMMON, discriminator="kind", cases=_CASES)
    assert out == msg


def test_type5_parses_its_own_body_not_type1() -> None:
    msg = {"kind": 5, "id": 9, "callsign": 0x1FF, "draught": 42}
    payload = parse.build_message(msg, _COMMON + _TYPE5_FIELDS)
    out = parse.parse_message(payload, _COMMON, discriminator="kind", cases=_CASES)
    assert out == msg
    assert out is not None
    assert "speed" not in out and "course" not in out  # not the type-1 struct


def test_unknown_type_yields_common_only_never_misparsed() -> None:
    # kind=9 has no case: the shared header is parsed, no wrong-type body applied
    payload = bytes([0x90, 0x03, 0xFF, 0xFF, 0xFF])  # kind=9, id=3, junk body
    out = parse.parse_message(payload, _COMMON, discriminator="kind", cases=_CASES)
    assert out == {"kind": 9, "id": 3}


def test_empty_payload_returns_none() -> None:
    assert parse.parse_message(b"", [{"name": "a", "bits": 8}]) is None


def test_payload_shorter_than_fixed_fields_returns_none() -> None:
    fields = [{"name": "a", "bits": 8}, {"name": "b", "bits": 8}]
    assert parse.parse_message(b"\xa5", fields) is None


def test_case_body_longer_than_payload_falls_back_to_shared_header() -> None:
    # kind=1 selects the type-1 case, but the payload only holds the 2-byte
    # header: the case must not be applied, the shared header stands alone.
    payload = bytes([0x10, 0x07])  # kind=1, id=7, no body
    out = parse.parse_message(payload, _COMMON, discriminator="kind", cases=_CASES)
    assert out == {"kind": 1, "id": 7}
