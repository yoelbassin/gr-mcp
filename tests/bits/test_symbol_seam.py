from pathlib import Path

import numpy as np
import pytest

from marconi.bits.builder import ProgramBuilder
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.bitfile import write_bits, write_symbols
from marconi.core.levels import Level
from marconi.core.models import Bitstream, Symbolstream
from marconi.core.params import StageParams
from marconi.core.stages import RxStage, SpecValidationError


class _StubCarve(RxStage[ProgramBuilder]):
    """Test-only SYMBOLS->BITS stage: bits = symbols LSBs; claims the frame
    flags so a single-stage codec passes validate_codec."""

    name = "stub_carve"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "test"
    seeds_frames = True
    self_slicing = True

    class _P(StageParams):
        pass

    params_model = _P

    def emit_rx(self, b, params) -> None:  # type: ignore[no-untyped-def]
        from dataclasses import replace

        b.add(
            lambda c: replace(
                c, bits=np.asarray([int(s) & 1 for s in c.symbols], np.uint8)
            )
        )


def _registry():
    r = dict(registry())
    r["stub_carve"] = _StubCarve()
    return r


def _symbol_codec() -> CodecSpec:
    return CodecSpec(path=[CodecStep(conv="stub_carve", params={})])


def test_symbol_codec_sets_requires_symbol_input() -> None:
    prog = compile_codec(_symbol_codec(), _registry(), "rx")
    assert prog.requires_symbol_input is True
    bits_codec = CodecSpec(path=[CodecStep(conv="hdlc_deframe", params={})])
    assert compile_codec(bits_codec, registry(), "rx").requires_symbol_input is False


def test_symbolstream_reaches_carrier_with_marks(tmp_path: Path) -> None:
    p = tmp_path / "s.i16"
    write_symbols(p, np.array([3, 2, 5], dtype=np.int16))
    res = parse_bitstream(
        Symbolstream(path=p, num_symbols=3, marks=[0, 2]),
        _symbol_codec(),
        _registry(),
    )
    assert res.num_frames >= 0  # ran end-to-end without raising


def test_bitstream_into_symbol_codec_raises(tmp_path: Path) -> None:
    p = tmp_path / "b.bits"
    write_bits(p, np.zeros(8, np.uint8))
    with pytest.raises(SpecValidationError):
        parse_bitstream(Bitstream(path=p, num_bits=8), _symbol_codec(), _registry())


def test_symbolstream_into_bits_codec_raises(tmp_path: Path) -> None:
    p = tmp_path / "s.i16"
    write_symbols(p, np.zeros(4, np.int16))
    codec = CodecSpec(path=[CodecStep(conv="hdlc_deframe", params={})])
    with pytest.raises(SpecValidationError):
        parse_bitstream(Symbolstream(path=p, num_symbols=4), codec, registry())


def test_soft_symbolstream_reaches_carrier(tmp_path: Path) -> None:
    import numpy as np

    from marconi.core.bitfile import write_llrs

    p = tmp_path / "s.f32"
    write_llrs(p, np.array([0.9, -0.3, 0.1, -0.8], dtype=np.float32))
    res = parse_bitstream(
        Symbolstream(path=p, num_symbols=4, item_type="f", marks=[0]),
        _symbol_codec(),
        _registry(),
    )
    assert res.num_frames >= 0
