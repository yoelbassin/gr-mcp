from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.bits.carriers import TxCarrier
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.program import run_program
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.bits.validate import validate_codec
from marconi.core.bitfile import write_bits, write_llrs
from marconi.core.models import Bitstream, SoftBitstream
from marconi.core.stages import SpecValidationError

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


def test_validate_flags_missing_parse() -> None:
    codec = CodecSpec(name="x", path=[CodecStep(conv="hdlc_deframe", params={})])
    issues = validate_codec(codec, registry())
    assert any("parse" in i.message for i in issues)


def test_validate_flags_unknown_conv() -> None:
    codec = CodecSpec(name="x", path=[CodecStep(conv="bogus", params={})])
    issues = validate_codec(codec, registry())
    assert any("bogus" in i.message for i in issues)


def test_parse_bitstream_decodes(tmp_path: Path) -> None:
    codec = CodecSpec(name="ais", path=AIS_PATH)
    reg = registry()
    msg = {"a": 1, "b": 0x123456789}
    tx = run_program(compile_codec(codec, reg, "tx"), TxCarrier(items=[msg]))
    bits = np.concatenate(tx.items).astype(np.uint8)
    p = tmp_path / "f.u8"
    write_bits(p, bits)
    res = parse_bitstream(Bitstream(path=p, num_bits=int(bits.size)), codec, reg)
    assert res.num_crc_ok == 1
    assert res.messages == [msg]


def test_soft_bitstream_rejected(tmp_path: Path) -> None:
    p = tmp_path / "l.f32"
    write_llrs(p, np.array([1.0, -1.0], dtype=np.float32))
    with pytest.raises(SpecValidationError):
        parse_bitstream(
            SoftBitstream(path=p, num_bits=2),
            CodecSpec(name="ais", path=AIS_PATH),
            registry(),
        )
