from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.coding import ops_bits, ops_symbols
from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.stages_bits import _codebook_kw
from marconi.engine.stages.base import RxStage, Stage
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import DcMode, DecodeMode
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class SyncSymbolsStep(Step):
    conv: Literal["sync_symbols"] = "sync_symbols"
    pattern: list[int]
    max_errors: StrictInt = 0
    pre_symbols: StrictInt = 0


class SyncSymbols(RxStage[CodingBuilder, SyncSymbolsStep]):
    name = "sync_symbols"
    engine = "coding"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    step_model = SyncSymbolsStep
    accepts_item_type = "f"

    def emit_rx(self, b: CodingBuilder, step: SyncSymbolsStep) -> None:
        b.add(
            ops_symbols.sync_symbols_rx,
            pattern=[int(x) for x in step.pattern],
            max_errors=step.max_errors,
            pre_symbols=step.pre_symbols,
        )


class NormalizeStep(Step):
    conv: Literal["normalize"] = "normalize"
    span_symbols: StrictInt = Field(ge=1)
    dc: DcMode = DcMode.MEDIAN
    gain_percentile: float | None = Field(default=None, ge=0.0, le=100.0)


class Normalize(RxStage[CodingBuilder, NormalizeStep]):
    name = "normalize"
    engine = "coding"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    step_model = NormalizeStep
    accepts_item_type = "f"

    def emit_rx(self, b: CodingBuilder, step: NormalizeStep) -> None:
        b.add(
            ops_symbols.normalize_rx,
            span_symbols=step.span_symbols,
            dc=step.dc,
            gain_percentile=step.gain_percentile,
        )


class MSliceStep(Step):
    conv: Literal["m_slice"] = "m_slice"
    thresholds: list[float]
    levels: list[int]

    @model_validator(mode="after")
    def _shaped(self) -> "MSliceStep":
        if len(self.levels) != len(self.thresholds) + 1:
            raise PydanticCustomError(
                "value_error", "levels must be one longer than thresholds"
            )
        if any(b <= a for a, b in zip(self.thresholds, self.thresholds[1:])):
            raise PydanticCustomError(
                "value_error", "thresholds must be strictly ascending"
            )
        return self


class MSlice(RxStage[CodingBuilder, MSliceStep]):
    name = "m_slice"
    engine = "coding"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    step_model = MSliceStep
    accepts_item_type = "f"

    def emit_rx(self, b: CodingBuilder, step: MSliceStep) -> None:
        b.add(
            ops_symbols.m_slice_rx,
            thresholds=[float(x) for x in step.thresholds],
            levels=[int(x) for x in step.levels],
        )

    def out_descriptor(self, in_desc: Descriptor, step: MSliceStep) -> Descriptor:
        return replace(
            in_desc,
            item_type="s",
            carrier=Carrier.HARD,
            order=len(step.levels),
        )

    def required_input_order(self, step: MSliceStep) -> int | None:
        return len(step.levels)


class SymbolMapStep(Step):
    conv: Literal["symbol_map"] = "symbol_map"
    code_bits: StrictInt = Field(ge=1)
    data_bits: StrictInt = Field(ge=1)
    table: list[int]
    decode: DecodeMode = DecodeMode.EXACT

    @model_validator(mode="after")
    def _sized(self) -> "SymbolMapStep":
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
        if self.decode == DecodeMode.EXACT and self.code_bits > 24:
            raise PydanticCustomError(
                "value_error",
                "exact decode builds a 2^code_bits inverse table; beyond 24 "
                "bits use decode='nearest', which never builds it",
            )
        if self.decode == DecodeMode.NEAREST and self.code_bits > 63:
            raise PydanticCustomError(
                "value_error",
                "nearest decode packs codewords through int64; 63 bits is "
                "the ceiling",
            )
        return self


class SymbolMap(RxStage[CodingBuilder, SymbolMapStep]):
    name = "symbol_map"
    engine = "coding"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "coding"
    step_model = SymbolMapStep
    accepts_item_type = "s"
    accepts_carrier = Carrier.HARD

    def emit_rx(self, b: CodingBuilder, step: SymbolMapStep) -> None:
        b.add(ops_bits.codebook_rx, **_codebook_kw(step), symbol_input=True)

    def out_descriptor(self, in_desc: Descriptor, step: SymbolMapStep) -> Descriptor:
        return Descriptor(Level.BITS, "b", Carrier.HARD, Amplitude.UNKNOWN)


CODING_SYMBOL_STAGES: tuple[type[Stage[CodingBuilder, Any]], ...] = (
    SyncSymbols,
    Normalize,
    MSlice,
    SymbolMap,
)
