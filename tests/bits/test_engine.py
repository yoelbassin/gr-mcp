from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from marconi.bits.carriers import RxCarrier, TxCarrier
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.program import run_program
from marconi.bits.registry import registry

AIS_PATH = [
    CodecStep(conv="differential", params={"invert": True}),
    CodecStep(conv="hdlc_deframe", params={"bit_order": "lsb"}),
    CodecStep(
        conv="crc",
        params={
            "poly": 0x1021,
            "bits": 16,
            "init": 0xFFFF,
            "reflected": True,
            "xorout": 0xFFFF,
            "bit_order": "lsb",
        },
    ),
    CodecStep(
        conv="parse",
        params={
            "bit_order": "msb",
            "fields": [{"name": "a", "bits": 6}, {"name": "b", "bits": 162}],
        },
    ),
]


def test_ais_codec_roundtrip() -> None:
    codec = CodecSpec(name="ais", path=AIS_PATH)
    reg = registry()
    msg = {"a": 1, "b": 0x123456789}
    tx = run_program(compile_codec(codec, reg, "tx"), TxCarrier(items=[msg]))
    bits = np.concatenate(tx.items).astype(np.uint8)
    rx = run_program(compile_codec(codec, reg, "rx"), RxCarrier(bits=bits))
    assert len(rx.frames) == 1
    assert rx.frames[0].crc_ok is True
    assert rx.frames[0].message == msg


def test_compiler_is_name_blind() -> None:
    compiler_py = (
        Path(__file__).resolve().parents[2]
        / "packages"
        / "marconi-bits"
        / "src"
        / "marconi"
        / "bits"
        / "compiler.py"
    )
    names = set(registry())
    literals = {
        n.value
        for n in ast.walk(ast.parse(compiler_py.read_text()))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert not (literals & names), literals & names
