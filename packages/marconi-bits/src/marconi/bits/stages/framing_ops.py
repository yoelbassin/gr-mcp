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
    seeds_frames = True
    self_slicing = True  # flag-delimited frames are complete as emitted
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


class Codebook(DuplexStage[ProgramBuilder]):
    name = "codebook"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"

    class _Params(StageParams):
        code_bits: StrictInt = Field(ge=1)
        data_bits: StrictInt = Field(ge=1)
        table: list[int]

        @model_validator(mode="after")
        def _sized(self) -> "Codebook._Params":
            if len(self.table) != (1 << self.data_bits):
                raise PydanticCustomError(
                    "value_error",
                    "codebook table needs {want} entries for data_bits={data_bits}, "
                    "got {have}",
                    {
                        "want": 1 << self.data_bits,
                        "data_bits": self.data_bits,
                        "have": len(self.table),
                    },
                )
            return self

    params_model = _Params

    def _kw(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "code_bits": int(params["code_bits"]),
            "data_bits": int(params["data_bits"]),
            "table": [int(x) for x in params["table"]],
        }

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.codebook_rx, **self._kw(params))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.codebook_tx, **self._kw(params))


class _SyncParams(StageParams):
    sync: str
    max_errors: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _hex(self) -> "_SyncParams":
        try:
            if not bytes.fromhex(self.sync):
                raise ValueError
        except ValueError:
            raise PydanticCustomError(
                "value_error", "sync must be a non-empty even-length hex string"
            ) from None
        return self


class SyncWord(DuplexStage[ProgramBuilder]):
    name = "sync_word"
    from_level = Level.BITS
    to_level = Level.FRAMES
    seeds_frames = True  # establishes frame boundaries; a body slicer carves them
    family = "framing"

    params_model = _SyncParams

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(
            framing.sync_word_rx,
            sync=str(params["sync"]),
            max_errors=int(params.get("max_errors", 0)),
        )

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.sync_word_tx, sync=str(params["sync"]))


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
    seeds_frames = True  # seeds a fixed-stride region; needs a body slicer
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
    slices_body = True  # carves the seeded region into fixed-length bodies
    family = "framing"

    class _Params(StageParams):
        payload_bits: StrictInt = Field(ge=1)

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.fixed_frame_rx, payload_bits=int(params["payload_bits"]))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.fixed_frame_tx, payload_bits=int(params["payload_bits"]))


class LengthFrame(DuplexStage[ProgramBuilder]):
    name = "length_frame"
    from_level = Level.FRAMES
    to_level = Level.FRAMES
    slices_body = True  # carves the seeded region using a decoded length field
    family = "framing"

    class _Params(StageParams):
        length_bits: StrictInt = Field(ge=1)
        base_bytes: StrictInt = Field(ge=0)
        unit_bytes: StrictInt = Field(default=1, ge=1)
        bit_order: str = "msb"

    params_model = _Params

    def _kw(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "length_bits": int(params["length_bits"]),
            "base_bytes": int(params["base_bytes"]),
            "unit_bytes": int(params.get("unit_bytes", 1)),
            "bit_order": str(params.get("bit_order", "msb")),
        }

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.length_frame_rx, **self._kw(params))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.length_frame_tx, **self._kw(params))


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
    Codebook,
    Descramble,
    DescrambleBits,
    Differential,
    FixedFrame,
    HdlcDeframe,
    LengthFrame,
    NibbleSwap,
    Segment,
    SyncWord,
)
