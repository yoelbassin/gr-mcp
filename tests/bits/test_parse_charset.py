from __future__ import annotations

import numpy as np

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame
from marconi.bits.models import ParseField

_ASCII7 = "".join(chr(i) if 32 <= i < 127 else " " for i in range(128))


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


def test_rest_round_trips_through_tx_then_rx() -> None:
    fields = [
        {"name": "kind", "bits": 8},
        {"name": "text", "bits": 0, "charset": _ASCII7, "char_bits": 7, "rest": True},
    ]
    pf = framing.parse_fields(fields)
    struct = framing.build_struct(pf)
    msg = {"kind": 0x7E, "text": "MARCONI!"}

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
    assert rx.frames[0].message == msg
