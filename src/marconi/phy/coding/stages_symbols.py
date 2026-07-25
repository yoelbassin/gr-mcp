from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.core.descriptor import Amplitude, Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import RxStage, Stage
from marconi.phy.coding import ops_bits, ops_symbols
from marconi.phy.coding.builder import CodingBuilder
from marconi.phy.coding.stages_bits import _CodebookParams


class _SyncSymbolsParams(StageParams):
    pattern: list[int]
    max_errors: StrictInt = 0
    pre_symbols: StrictInt = 0


class SyncSymbols(RxStage[CodingBuilder]):
    name = "sync_symbols"
    engine = "coding"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    params_model = _SyncSymbolsParams
    accepts_item_type = "f"

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = _SyncSymbolsParams.model_validate(dict(params))
        b.add(
            ops_symbols.sync_symbols_rx,
            pattern=[int(x) for x in p.pattern],
            max_errors=p.max_errors,
            pre_symbols=p.pre_symbols,
        )


class _NormalizeParams(StageParams):
    span_symbols: StrictInt = Field(ge=1)
    dc: str = "median"
    gain_percentile: float | None = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _dc(self) -> "_NormalizeParams":
        if self.dc not in ("median", "mean", "none"):
            raise PydanticCustomError("value_error", "dc must be median|mean|none")
        return self


class Normalize(RxStage[CodingBuilder]):
    name = "normalize"
    engine = "coding"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    params_model = _NormalizeParams
    accepts_item_type = "f"

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = _NormalizeParams.model_validate(dict(params))
        b.add(
            ops_symbols.normalize_rx,
            span_symbols=p.span_symbols,
            dc=p.dc,
            gain_percentile=p.gain_percentile,
        )


class _MSliceParams(StageParams):
    thresholds: list[float]
    levels: list[int]

    @model_validator(mode="after")
    def _shaped(self) -> "_MSliceParams":
        if len(self.levels) != len(self.thresholds) + 1:
            raise PydanticCustomError(
                "value_error", "levels must be one longer than thresholds"
            )
        if any(b <= a for a, b in zip(self.thresholds, self.thresholds[1:])):
            raise PydanticCustomError(
                "value_error", "thresholds must be strictly ascending"
            )
        return self


class MSlice(RxStage[CodingBuilder]):
    name = "m_slice"
    engine = "coding"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    params_model = _MSliceParams
    accepts_item_type = "f"

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = _MSliceParams.model_validate(dict(params))
        b.add(
            ops_symbols.m_slice_rx,
            thresholds=[float(x) for x in p.thresholds],
            levels=[int(x) for x in p.levels],
        )

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return replace(in_desc, item_type="s", carrier=Carrier.HARD)


class SymbolMap(RxStage[CodingBuilder]):
    name = "symbol_map"
    engine = "coding"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "coding"
    params_model = _CodebookParams
    accepts_item_type = "s"
    accepts_carrier = Carrier.HARD

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        b.add(
            ops_bits.codebook_rx,
            code_bits=int(params["code_bits"]),
            data_bits=int(params["data_bits"]),
            table=[int(x) for x in params["table"]],
            symbol_input=True,
        )

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(
            Level.BITS, "b", in_desc.layout, Carrier.HARD, Amplitude.UNKNOWN
        )


CODING_SYMBOL_STAGES: tuple[type[Stage[CodingBuilder]], ...] = (
    SyncSymbols,
    Normalize,
    MSlice,
    SymbolMap,
)
