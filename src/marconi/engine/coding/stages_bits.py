from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.coding import ops_bits
from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.primitives import can_correct
from marconi.engine.stages.base import RxStage, Stage
from marconi.engine.types.descriptor import Carrier
from marconi.engine.types.levels import Level
from marconi.engine.types.params import StageParams


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


class SyncWord(RxStage[CodingBuilder]):
    name = "sync_word"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    seeds_windows = True
    family = "acquisition"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    params_model = _SyncParams

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = _SyncParams.model_validate(dict(params))
        b.add(ops_bits.sync_word_rx, sync=p.sync, max_errors=p.max_errors)


class MarkFrame(RxStage[CodingBuilder]):
    name = "mark_frame"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    seeds_windows = True
    family = "acquisition"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    class _Params(StageParams):
        offset_bits: StrictInt = Field(default=0, ge=0)

    params_model = _Params

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        b.add(ops_bits.mark_frame_rx, offset_bits=int(params.get("offset_bits", 0)))


class Segment(RxStage[CodingBuilder]):
    name = "segment"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    seeds_windows = True
    family = "acquisition"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    class _Params(StageParams):
        frame_body_len: StrictInt = Field(ge=1)

    params_model = _Params

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        b.add(ops_bits.segment_rx, frame_body_len=int(params["frame_body_len"]))


class Codebook(RxStage[CodingBuilder]):
    name = "codebook"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    params_model = _CodebookParams

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        b.add(ops_bits.codebook_rx, **_codebook_kw(params))


class BlockCode(RxStage[CodingBuilder]):
    name = "block_code"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    class _Params(StageParams):
        code_bits: StrictInt = Field(ge=2)
        data_bits: StrictInt = Field(ge=1)
        parity_masks: list[int]
        correct_single: bool | None = None
        emit: str = "data"

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
            if self.emit not in ("data", "codeword"):
                raise PydanticCustomError("value_error", "emit must be data|codeword")
            corrects = (
                can_correct(self.code_bits - self.data_bits, self.data_bits)
                if self.correct_single is None
                else self.correct_single
            )
            if corrects:
                cols = [
                    tuple((mask >> j) & 1 for mask in self.parity_masks)
                    for j in range(self.data_bits)
                ]
                if any(sum(col) < 2 for col in cols) or len(set(cols)) != len(cols):
                    raise PydanticCustomError(
                        "value_error",
                        "correct_single needs pairwise-distinct parity-check "
                        "columns of weight >= 2 - single-error syndromes must "
                        "be unique across data and parity positions",
                    )
            return self

    params_model = _Params

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(
            ops_bits.block_code_rx,
            code_bits=p.code_bits,
            data_bits=p.data_bits,
            parity_masks=[int(x) for x in p.parity_masks],
            correct_single=p.correct_single,
            emit=p.emit,
        )


class ConvCode(RxStage[CodingBuilder]):
    name = "conv_code"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    class _Params(StageParams):
        rate_inv: StrictInt = Field(ge=2)
        polys: list[int] = Field(min_length=2)
        frame_bits: StrictInt = Field(ge=1)
        tail: StrictInt = Field(default=0, ge=0)

        @model_validator(mode="after")
        def _shaped(self) -> "ConvCode._Params":
            if len(self.polys) != self.rate_inv:
                raise PydanticCustomError(
                    "value_error",
                    "need rate_inv={want} polys, got {have}",
                    {"want": self.rate_inv, "have": len(self.polys)},
                )
            if any(int(p).bit_length() < 2 for p in self.polys):
                raise PydanticCustomError(
                    "value_error", "each poly needs memory (bit_length >= 2)"
                )
            return self

    params_model = _Params

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(
            ops_bits.conv_code_rx,
            rate_inv=p.rate_inv,
            polys=[int(x) for x in p.polys],
            frame_bits=p.frame_bits,
            tail=p.tail,
        )


class Permute(RxStage[CodingBuilder]):
    name = "permute"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    class _Params(StageParams):
        perm: list[int] = Field(min_length=1)

        @model_validator(mode="after")
        def _non_negative(self) -> "Permute._Params":
            if any(i < 0 for i in self.perm):
                raise PydanticCustomError(
                    "value_error",
                    "perm indices must be >= 0; a negative index would wrap and "
                    "understate the input stride",
                )
            return self

    params_model = _Params

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(ops_bits.permute_rx, perm=[int(x) for x in p.perm])


class Realign(RxStage[CodingBuilder]):
    name = "realign"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    class _Params(StageParams):
        bit_offset: StrictInt = Field(ge=0)

    params_model = _Params

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        b.add(ops_bits.realign_rx, bit_offset=int(params["bit_offset"]))


class Differential(RxStage[CodingBuilder]):
    name = "differential"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    class _Params(StageParams):
        invert: bool = False

    params_model = _Params

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(ops_bits.differential_rx, invert=p.invert)


class NibbleSwap(RxStage[CodingBuilder]):
    name = "nibble_swap"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    class _Params(StageParams):
        pass

    params_model = _Params

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        b.add(ops_bits.nibble_swap_rx)


class RsCode(RxStage[CodingBuilder]):
    name = "rs_code"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    class _Params(StageParams):
        symbol_bits: StrictInt = Field(ge=3, le=16)
        n: StrictInt = Field(ge=3)
        k: StrictInt = Field(ge=1)
        prim_poly: StrictInt
        fcr: StrictInt = Field(default=0, ge=0)
        generator: StrictInt = Field(default=2, ge=1)
        emit: str = "data"

        @model_validator(mode="after")
        def _shaped(self) -> "RsCode._Params":
            if self.k >= self.n:
                raise PydanticCustomError("value_error", "k must be < n")
            if self.n > (1 << self.symbol_bits) - 1:
                raise PydanticCustomError(
                    "value_error", "n must be <= 2^symbol_bits - 1"
                )
            if self.prim_poly.bit_length() != self.symbol_bits + 1:
                raise PydanticCustomError(
                    "value_error", "prim_poly must have degree symbol_bits"
                )
            if self.emit not in ("data", "codeword"):
                raise PydanticCustomError("value_error", "emit must be data|codeword")
            return self

    params_model = _Params

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(
            ops_bits.rs_code_rx,
            symbol_bits=p.symbol_bits,
            n=p.n,
            k=p.k,
            prim_poly=p.prim_poly,
            fcr=p.fcr,
            generator=p.generator,
            emit=p.emit,
        )


class Descramble(RxStage[CodingBuilder]):
    name = "descramble"
    engine = "coding"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    params_model = _SequenceParams

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        b.add(ops_bits.descramble_rx, sequence=str(params["sequence"]))


CODING_BITS_STAGES: tuple[type[Stage[CodingBuilder]], ...] = (
    BlockCode,
    Codebook,
    ConvCode,
    Descramble,
    Differential,
    MarkFrame,
    NibbleSwap,
    Permute,
    Realign,
    RsCode,
    Segment,
    SyncWord,
)
