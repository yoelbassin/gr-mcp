from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from marconi.bits.carriers import RxCarrier
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, DecodeResult, FrameResult
from marconi.bits.program import run_program
from marconi.bits.validate import validate_codec
from marconi.core.bitfile import read_bits, read_llrs
from marconi.core.models import Bitstream, SoftBitstream, ValidationIssue
from marconi.core.stages import SpecValidationError, Stage


def parse_bitstream(
    bitstream: Bitstream | SoftBitstream,
    codec: CodecSpec,
    registry: Mapping[str, Stage[Any]],
) -> DecodeResult:
    issues = validate_codec(codec, registry)
    if issues:
        raise SpecValidationError(issues, kind="codec")
    program = compile_codec(codec, registry, "rx")
    is_soft = isinstance(bitstream, SoftBitstream)
    if is_soft != program.requires_soft_input:
        msg = (
            "codec expects soft input (LLRs) but a hard Bitstream was given"
            if program.requires_soft_input
            else "a SoftBitstream was given but this codec expects hard bits"
        )
        raise SpecValidationError([ValidationIssue(message=msg)], kind="codec")
    carrier = (
        RxCarrier(bits=np.zeros(0, np.uint8), llrs=read_llrs(bitstream.path))
        if is_soft
        else RxCarrier(bits=read_bits(bitstream.path))
    )
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
