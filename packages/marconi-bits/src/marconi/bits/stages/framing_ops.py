from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.bits import framing
from marconi.bits.builder import ProgramBuilder
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage


class Differential(DuplexStage[ProgramBuilder]):
    name = "differential"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "framing"

    class _Params(StageParams):
        invert: bool = False

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.differential_rx, invert=bool(params.get("invert", False)))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.differential_tx, invert=bool(params.get("invert", False)))


class HdlcDeframe(DuplexStage[ProgramBuilder]):
    name = "hdlc_deframe"
    from_level = Level.BITS
    to_level = Level.FRAMES
    family = "framing"

    class _Params(StageParams):
        bit_order: str = "msb"
        training: str = ""

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.hdlc_deframe_rx, bit_order=str(params.get("bit_order", "msb")))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(
            framing.hdlc_deframe_tx,
            bit_order=str(params.get("bit_order", "msb")),
            training=str(params.get("training", "")),
        )


class NibbleSwap(DuplexStage[ProgramBuilder]):
    name = "nibble_swap"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "framing"

    class _Params(StageParams):
        pass

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.nibble_swap_rx)

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.nibble_swap_tx)


class Segment(DuplexStage[ProgramBuilder]):
    name = "segment"
    from_level = Level.BITS
    to_level = Level.FRAMES
    family = "framing"

    class _Params(StageParams):
        frame_body_len: StrictInt = Field(ge=1)

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.segment_rx, frame_body_len=int(params["frame_body_len"]))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.segment_tx)


class FixedFrame(DuplexStage[ProgramBuilder]):
    name = "fixed_frame"
    from_level = Level.FRAMES
    to_level = Level.FRAMES
    family = "framing"

    class _Params(StageParams):
        payload_bits: StrictInt = Field(ge=1)

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.fixed_frame_rx, payload_bits=int(params["payload_bits"]))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.fixed_frame_tx, payload_bits=int(params["payload_bits"]))


class _SequenceParams(StageParams):
    sequence: str

    @model_validator(mode="after")
    def _ok(self) -> "_SequenceParams":
        try:
            bytes.fromhex(self.sequence)
        except ValueError:
            raise PydanticCustomError(
                "value_error", "sequence must be an even-length hex string"
            ) from None
        return self


class Descramble(DuplexStage[ProgramBuilder]):
    name = "descramble"
    from_level = Level.FRAMES
    to_level = Level.FRAMES
    family = "framing"

    params_model = _SequenceParams

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.descramble_rx, sequence=str(params["sequence"]))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.descramble_tx, sequence=str(params["sequence"]))


class DescrambleBits(DuplexStage[ProgramBuilder]):
    name = "descramble_bits"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "framing"

    params_model = _SequenceParams

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.descramble_bits_rx, sequence=str(params["sequence"]))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.descramble_bits_tx, sequence=str(params["sequence"]))


FRAMING_STAGES: tuple[type[DuplexStage[ProgramBuilder]], ...] = (
    Descramble,
    DescrambleBits,
    Differential,
    FixedFrame,
    HdlcDeframe,
    NibbleSwap,
    Segment,
)
