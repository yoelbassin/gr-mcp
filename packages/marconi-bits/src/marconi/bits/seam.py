from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from marconi.bits.carriers import RxCarrier
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, DecodeResult, FrameResult
from marconi.bits.program import run_program
from marconi.bits.validate import validate_codec
from marconi.core.bitfile import read_bits, read_symbols
from marconi.core.models import Bitstream, SoftBitstream, Symbolstream, ValidationIssue
from marconi.core.stages import SpecValidationError, Stage


def parse_bitstream(
    bitstream: Bitstream | SoftBitstream | Symbolstream,
    codec: CodecSpec,
    registry: Mapping[str, Stage[Any]],
) -> DecodeResult:
    if isinstance(bitstream, SoftBitstream):
        raise SpecValidationError(
            [
                ValidationIssue(
                    message="no bits stage consumes soft input; harden the LLRs"
                    " into a Bitstream first (marconi.core.bitfile.harden)"
                )
            ],
            kind="codec",
        )
    issues = validate_codec(codec, registry)
    if issues:
        raise SpecValidationError(issues, kind="codec")
    program = compile_codec(codec, registry, "rx")
    is_symbols = isinstance(bitstream, Symbolstream)
    if is_symbols != program.requires_symbol_input:
        msg = (
            "codec starts at SYMBOLS but no Symbolstream was given"
            if program.requires_symbol_input
            else "a Symbolstream was given but this codec starts at BITS"
        )
        raise SpecValidationError([ValidationIssue(message=msg)], kind="codec")
    if is_symbols:
        symbolstream = cast(Symbolstream, bitstream)
        carrier = RxCarrier(
            bits=np.zeros(0, np.uint8),
            symbols=read_symbols(symbolstream.path),
            marks=tuple(int(m) for m in symbolstream.marks),
        )
    else:
        carrier = RxCarrier(bits=read_bits(bitstream.path))
    out = run_program(program, carrier)
    frames = [
        FrameResult(
            bit_offset=f.start,
            payload_hex=f.payload.hex(),
            crc_ok=f.crc_ok,
            message=f.message,
        )
        for f in out.frames
    ]
    return DecodeResult(
        messages=[f.message for f in out.frames if f.message is not None],
        frames=frames,
        num_frames=len(frames),
        num_crc_ok=sum(1 for f in frames if f.crc_ok is True),
        num_unchecked=sum(1 for f in frames if f.crc_ok is None),
    )
