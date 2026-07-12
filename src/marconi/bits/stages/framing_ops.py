from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.bits import framing
from marconi.bits.builder import ProgramBuilder
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage, RxStage, Stage


class Differential(DuplexStage[ProgramBuilder]):
    name = "differential"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "framing"

    class _Params(StageParams):
        invert: bool = False

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(framing.differential_rx, invert=p.invert)

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(framing.differential_tx, invert=p.invert)


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
        p = self._Params.model_validate(dict(params))
        b.add(framing.hdlc_deframe_rx, bit_order=p.bit_order)

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(framing.hdlc_deframe_tx, bit_order=p.bit_order, training=p.training)


class _CodebookParams(StageParams):
    code_bits: StrictInt = Field(ge=1)
    data_bits: StrictInt = Field(ge=1)
    table: list[int]

    @model_validator(mode="after")
    def _sized(self) -> "_CodebookParams":
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


def _codebook_kw(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "code_bits": int(params["code_bits"]),
        "data_bits": int(params["data_bits"]),
        "table": [int(x) for x in params["table"]],
    }


class Codebook(DuplexStage[ProgramBuilder]):
    name = "codebook"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"

    params_model = _CodebookParams

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.codebook_rx, **_codebook_kw(params))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.codebook_tx, **_codebook_kw(params))


class FrameCodebook(DuplexStage[ProgramBuilder]):
    name = "frame_codebook"
    from_level = Level.FRAMES
    to_level = Level.FRAMES
    family = "coding"

    params_model = _CodebookParams

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.frame_codebook_rx, **_codebook_kw(params))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.frame_codebook_tx, **_codebook_kw(params))


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
        p = _SyncParams.model_validate(dict(params))
        b.add(framing.sync_word_rx, sync=p.sync, max_errors=p.max_errors)

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = _SyncParams.model_validate(dict(params))
        b.add(framing.sync_word_tx, sync=p.sync)


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
        p = self._Params.model_validate(dict(params))
        return {
            "length_bits": p.length_bits,
            "base_bytes": p.base_bytes,
            "unit_bytes": p.unit_bytes,
            "bit_order": p.bit_order,
        }

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.length_frame_rx, **self._kw(params))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.length_frame_tx, **self._kw(params))


class DelimiterFrame(RxStage[ProgramBuilder]):
    name = "delimiter_frame"
    from_level = Level.FRAMES
    to_level = Level.FRAMES
    slices_body = True
    family = "framing"

    class _Params(StageParams):
        delimiter: str
        tail_bits: StrictInt = Field(default=0, ge=0)

        @model_validator(mode="after")
        def _hex(self) -> "DelimiterFrame._Params":
            try:
                if not bytes.fromhex(self.delimiter):
                    raise ValueError
            except ValueError:
                raise PydanticCustomError(
                    "value_error",
                    "delimiter must be a non-empty even-length hex string",
                ) from None
            return self

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(framing.delimiter_frame_rx, delimiter=p.delimiter, tail_bits=p.tail_bits)


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


class BlockCode(RxStage[ProgramBuilder]):
    name = "block_code"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"

    class _Params(StageParams):
        code_bits: StrictInt = Field(ge=2)
        data_bits: StrictInt = Field(ge=1)
        parity_masks: list[int]
        correct: bool | None = None

        @model_validator(mode="after")
        def _shaped(self) -> "BlockCode._Params":
            if self.data_bits >= self.code_bits:
                raise PydanticCustomError(
                    "value_error", "data_bits must be < code_bits"
                )
            if len(self.parity_masks) != self.code_bits - self.data_bits:
                raise PydanticCustomError(
                    "value_error",
                    "need {want} parity masks for ({n},{k}), got {have}",
                    {
                        "want": self.code_bits - self.data_bits,
                        "n": self.code_bits,
                        "k": self.data_bits,
                        "have": len(self.parity_masks),
                    },
                )
            return self

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(
            framing.block_code_rx,
            code_bits=p.code_bits,
            data_bits=p.data_bits,
            parity_masks=[int(x) for x in p.parity_masks],
            correct=p.correct,
        )


class Realign(RxStage[ProgramBuilder]):
    name = "realign"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "framing"

    class _Params(StageParams):
        bit_offset: StrictInt = Field(ge=0)

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.realign_rx, bit_offset=int(params["bit_offset"]))


FRAMING_STAGES: tuple[type[Stage[ProgramBuilder]], ...] = (
    BlockCode,
    Codebook,
    DelimiterFrame,
    Descramble,
    DescrambleBits,
    Differential,
    FixedFrame,
    FrameCodebook,
    HdlcDeframe,
    LengthFrame,
    NibbleSwap,
    Realign,
    Segment,
    SyncWord,
)
