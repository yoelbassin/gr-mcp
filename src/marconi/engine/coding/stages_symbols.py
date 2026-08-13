from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.coding import ops_bits, ops_symbols
from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.stages_bits import _codebook_kw, check_codebook_sizing
from marconi.engine.stages.base import CodingStage, Stage
from marconi.engine.types.bounds import (
    MAX_FRAME_ITEMS,
    MAX_WINDOW_SYMBOLS,
    check_match_tolerance,
)
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import DcMode, DecodeMode, ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class SyncSymbolsStep(Step):
    conv: Literal["sync_symbols"] = "sync_symbols"
    pattern: list[int] = Field(
        description=(
            "Sign pattern matched against the soft symbols: +1 requires a "
            "positive symbol, -1 a negative one, 0 is a wildcard (don't "
            "care). NOT a 0/1 bit string — encode bits as +-1 first, or "
            "every 0 becomes a wildcard and the correlator over-matches."
        )
    )
    max_errors: StrictInt = Field(default=0, ge=0)
    pre_symbols: StrictInt = Field(default=0, ge=0, le=MAX_FRAME_ITEMS)

    @model_validator(mode="after")
    def _sign_alphabet(self) -> "SyncSymbolsStep":
        if any(x not in (-1, 0, 1) for x in self.pattern):
            raise PydanticCustomError(
                "value_error",
                "pattern entries must be -1 (match negative), 0 (wildcard), "
                "or +1 (match positive)",
            )
        if not any(self.pattern):
            raise PydanticCustomError(
                "value_error",
                "pattern needs at least one non-wildcard (+-1) entry",
            )
        check_match_tolerance(self.max_errors, len(self.pattern), field="max_errors")
        return self


class SyncSymbols(CodingStage[SyncSymbolsStep]):
    name = "sync_symbols"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    step_model = SyncSymbolsStep
    accepts_item_type = ItemType.F

    def emit_rx(self, b: CodingBuilder, step: SyncSymbolsStep) -> None:
        b.add(
            ops_symbols.sync_symbols_rx,
            pattern=[int(x) for x in step.pattern],
            max_errors=step.max_errors,
            pre_symbols=step.pre_symbols,
        )


class NormalizeStep(Step):
    conv: Literal["normalize"] = "normalize"
    span_symbols: StrictInt = Field(ge=1, le=MAX_WINDOW_SYMBOLS)
    dc: DcMode = DcMode.MEDIAN
    gain_percentile: float | None = Field(default=None, ge=0.0, le=100.0)


class Normalize(CodingStage[NormalizeStep]):
    name = "normalize"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    step_model = NormalizeStep
    accepts_item_type = ItemType.F

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


class MSlice(CodingStage[MSliceStep]):
    name = "m_slice"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    step_model = MSliceStep
    accepts_item_type = ItemType.F

    def emit_rx(self, b: CodingBuilder, step: MSliceStep) -> None:
        b.add(
            ops_symbols.m_slice_rx,
            thresholds=[float(x) for x in step.thresholds],
            levels=[int(x) for x in step.levels],
        )

    def out_descriptor(self, in_desc: Descriptor, step: MSliceStep) -> Descriptor:
        return replace(
            in_desc,
            item_type=ItemType.S,
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
        check_codebook_sizing(self.code_bits, self.data_bits, self.table, self.decode)
        return self


class SymbolMap(CodingStage[SymbolMapStep]):
    name = "symbol_map"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "coding"
    step_model = SymbolMapStep
    accepts_item_type = ItemType.S
    accepts_carrier = Carrier.HARD

    def emit_rx(self, b: CodingBuilder, step: SymbolMapStep) -> None:
        b.add(ops_bits.codebook_rx, **_codebook_kw(step), symbol_input=True)

    def out_descriptor(self, in_desc: Descriptor, step: SymbolMapStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.B, Carrier.HARD, Amplitude.UNKNOWN)


CODING_SYMBOL_STAGES: tuple[type[Stage[CodingBuilder, Any]], ...] = (
    SyncSymbols,
    Normalize,
    MSlice,
    SymbolMap,
)
