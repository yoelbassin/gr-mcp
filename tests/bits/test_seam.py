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


def test_parseless_deframe_crc_is_valid() -> None:
    # parse is optional: a deframe+integrity codec of an unknown protocol (the
    # reverse-engineering entry state) validates and decodes to frames (08).
    codec = CodecSpec(name="x", path=AIS_PATH[:3])  # differential, hdlc, crc
    assert not validate_codec(codec, registry())


def test_parseless_codec_decodes_to_frames(tmp_path: Path) -> None:
    reg = registry()
    full = CodecSpec(name="ais", path=AIS_PATH)
    msg = {"a": 1, "b": 0x123456789}
    tx = run_program(compile_codec(full, reg, "tx"), TxCarrier(items=[msg]))
    bits = np.concatenate(tx.items).astype(np.uint8)
    p = tmp_path / "f.u8"
    write_bits(p, bits)
    parseless = CodecSpec(name="reveng", path=AIS_PATH[:3])
    res = parse_bitstream(Bitstream(path=p, num_bits=int(bits.size)), parseless, reg)
    assert res.num_frames == 1 and res.num_crc_ok == 1
    assert res.messages == []  # frames without messages is a first-class result


def test_validate_flags_missing_seeder() -> None:
    codec = CodecSpec(name="x", path=[CodecStep(conv="differential", params={})])
    issues = validate_codec(codec, registry())
    assert any("seeder" in i.message for i in issues)


def test_validate_flags_unknown_conv() -> None:
    codec = CodecSpec(name="x", path=[CodecStep(conv="bogus", params={})])
    issues = validate_codec(codec, registry())
    assert any("bogus" in i.message for i in issues)


def test_user_registered_seeder_compiles_through_seam(tmp_path: Path) -> None:
    # A novel seeder registered into the injectable registry compiles and runs
    # through the public seam WITHOUT editing packages/ — capability travels on
    # the stage's flags, not a validator name-set (08).
    from collections.abc import Mapping
    from typing import Any

    from marconi.bits.builder import ProgramBuilder
    from marconi.bits.carriers import RxCarrier, _Frame
    from marconi.core.levels import Level
    from marconi.core.params import StageParams
    from marconi.core.stages import DuplexStage

    _MARK = 0xA5

    def _marker_deframe_rx(c: RxCarrier) -> RxCarrier:
        raw = np.packbits(c.bits)
        frames, off = [], 0
        for chunk in bytes(raw).split(bytes([_MARK])):
            if chunk:
                frames.append(
                    _Frame(start=off, cursor=off + len(chunk) * 8, payload=chunk)
                )
                off += len(chunk) * 8
        return RxCarrier(bits=c.bits, frames=frames)

    class MarkerDeframe(DuplexStage[ProgramBuilder]):
        name = "marker_deframe"
        from_level = Level.BITS
        to_level = Level.FRAMES
        family = "framing"
        seeds_frames = True
        self_slicing = True
        params_model = StageParams

        def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
            b.add(_marker_deframe_rx)

        def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
            raise NotImplementedError

    reg = {**registry(), "marker_deframe": MarkerDeframe()}
    codec = CodecSpec(name="novel", path=[CodecStep(conv="marker_deframe", params={})])
    assert not validate_codec(codec, reg)

    payload = bytes([0x11, 0x22, 0x33])
    framed = bytes([_MARK]) + payload + bytes([_MARK])
    bits = np.unpackbits(np.frombuffer(framed, np.uint8)).astype(np.uint8)
    p = tmp_path / "novel.u8"
    write_bits(p, bits)
    res = parse_bitstream(Bitstream(path=p, num_bits=int(bits.size)), codec, reg)
    assert any(bytes.fromhex(f.payload_hex) == payload for f in res.frames)


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


def test_compile_codec_rejects_bad_params_with_named_field() -> None:
    # Direct compile_codec (not just the seam) must validate params up front and
    # raise a structured, field-named error — never let a raw pydantic
    # ValidationError escape from inside a stage's emit (issue 07).
    codec = CodecSpec(
        name="x",
        path=[CodecStep(conv="crc", params={"poly": 0x1021})],  # 'bits' missing
    )
    with pytest.raises(SpecValidationError) as e:
        compile_codec(codec, registry(), "rx")
    assert "bits" in str(e.value)


def test_soft_bitstream_rejected(tmp_path: Path) -> None:
    p = tmp_path / "l.f32"
    write_llrs(p, np.array([1.0, -1.0], dtype=np.float32))
    with pytest.raises(SpecValidationError):
        parse_bitstream(
            SoftBitstream(path=p, num_bits=2),
            CodecSpec(name="ais", path=AIS_PATH),
            registry(),
        )
