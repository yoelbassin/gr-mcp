"""parse dispatches on a discriminator field (issue 07): different message
types carry different field layouts, so the wrong-type struct must never be
applied. Covers AIS msg_type (types 1/2/3 vs 5 vs others) and the general
mechanism (which also un-blobs DAB's messages rung).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.program import run_program
from marconi.bits.registry import registry

_COMMON = [{"name": "kind", "bits": 4}, {"name": "id", "bits": 12}]
_CASES: list[dict[str, Any]] = [
    {
        "when": 1,
        "fields": [{"name": "speed", "bits": 10}, {"name": "course", "bits": 12}],
    },
    {
        "when": 5,
        "fields": [{"name": "callsign", "bits": 42}, {"name": "draught", "bits": 8}],
    },
]
_COMMON_F = framing.parse_fields(_COMMON)
_STRUCT = framing.build_struct(_COMMON_F)


def _cases_map() -> dict:
    out = {}
    for case in _CASES:
        body = _COMMON_F + framing.parse_fields(case["fields"])
        out[case["when"]] = (framing.build_struct(body), body)
    return out


def _roundtrip(msg: dict) -> dict:
    cases = _cases_map()
    tx = framing.parse_tx(
        TxCarrier([msg]),
        struct=_STRUCT,
        fields=_COMMON_F,
        discriminator="kind",
        cases=cases,
    )
    frame = _Frame(start=0, cursor=0, payload=tx.items[0])
    rx = framing.parse_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), frames=[frame]),
        struct=_STRUCT,
        fields=_COMMON_F,
        discriminator="kind",
        cases=cases,
    )
    return dict(rx.frames[0].message or {})


def test_type1_parses_its_own_body() -> None:
    assert _roundtrip({"kind": 1, "id": 7, "speed": 300, "course": 900}) == {
        "kind": 1,
        "id": 7,
        "speed": 300,
        "course": 900,
    }


def test_type5_parses_its_own_body_not_type1() -> None:
    got = _roundtrip({"kind": 5, "id": 9, "callsign": 0x1FF, "draught": 42})
    assert got == {"kind": 5, "id": 9, "callsign": 0x1FF, "draught": 42}
    assert "speed" not in got and "course" not in got  # not the type-1 struct


def test_unknown_type_yields_common_only_never_misparsed() -> None:
    # kind=9 has no case: the shared header is parsed, no wrong-type body applied
    payload = bytes([0x90, 0x03, 0xFF, 0xFF, 0xFF])  # kind=9, id=3, junk body
    frame = _Frame(start=0, cursor=0, payload=payload)
    rx = framing.parse_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), frames=[frame]),
        struct=_STRUCT,
        fields=_COMMON_F,
        discriminator="kind",
        cases=_cases_map(),
    )
    assert dict(rx.frames[0].message or {}) == {"kind": 9, "id": 3}


def test_dispatch_composes_in_a_full_variable_length_codec() -> None:
    # HDLC delimits variable-length frames (as AIS does); two different types
    # ride the same codec and each parses its own body.
    codec = CodecSpec(
        name="typed",
        path=[
            CodecStep(conv="hdlc_deframe", params={}),
            CodecStep(
                conv="parse",
                params={"fields": _COMMON, "discriminator": "kind", "cases": _CASES},
            ),
        ],
    )
    reg = registry()
    msgs = [
        {"kind": 1, "id": 1, "speed": 111, "course": 222},
        {"kind": 5, "id": 2, "callsign": 0x2AA, "draught": 7},
    ]
    tx = run_program(compile_codec(codec, reg, "tx"), TxCarrier(items=msgs))
    bits = np.concatenate(tx.items).astype(np.uint8)
    rx = run_program(compile_codec(codec, reg, "rx"), RxCarrier(bits=bits))
    decoded = [dict(f.message or {}) for f in rx.frames]
    assert msgs[0] in decoded and msgs[1] in decoded
