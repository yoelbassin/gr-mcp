from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.core import coding
from marconi.core.descriptor import Amplitude, Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import RxStage, Stage
from marconi.phy.coding import css, ops_bits, ops_symbols
from marconi.phy.coding.builder import CodingBuilder
from marconi.phy.coding.stages_bits import _CodebookParams

_SF_MIN, _SF_MAX = 5, 14


class _ExplicitParams(StageParams):
    sf: StrictInt
    header_cr: StrictInt
    reduced: bool = False
    header_symbols: StrictInt
    header_nibbles: StrictInt
    sf_reduction: StrictInt
    header_data_bits: StrictInt
    header_parity: list[int]
    field_payload_len: list[int]
    field_cr: list[int]
    field_has_crc: list[int]
    field_parity: list[int]
    data_bits: StrictInt
    crc_bytes: StrictInt
    parity_masks: list[int]
    reduced_offset: StrictInt = 0
    full_offset: StrictInt = 0

    @model_validator(mode="after")
    def _ok(self) -> "_ExplicitParams":
        if not (_SF_MIN <= self.sf <= _SF_MAX):
            raise PydanticCustomError(
                "value_error",
                "sf {sf} out of range [{lo}, {hi}]",
                {"sf": self.sf, "lo": _SF_MIN, "hi": _SF_MAX, "field": "sf"},
            )
        if self.data_bits < 1 or 8 % self.data_bits != 0:
            raise PydanticCustomError(
                "value_error",
                "data_bits {data_bits} must divide 8",
                {"data_bits": self.data_bits, "field": "data_bits"},
            )
        if not coding.supported_cr(self.parity_masks, self.header_cr):
            raise PydanticCustomError(
                "value_error",
                "header_cr {header_cr} has no parity rows in the supplied table",
                {"header_cr": self.header_cr, "field": "header_cr"},
            )
        if self.reduced and self.sf_reduction >= self.sf:
            raise PydanticCustomError(
                "value_error",
                "sf_reduction {sf_reduction} must be < sf {sf} when reduced is set",
                {
                    "sf_reduction": self.sf_reduction,
                    "sf": self.sf,
                    "field": "sf_reduction",
                },
            )
        return self


class CssExplicitDecode(RxStage[CodingBuilder]):
    """CSS explicit-header decode, SYMBOLS->BITS (RX-only). Parses the explicit
    header, decodes the payload at its own code rate, emits de-FEC'd payload
    bits. Generic over CSS explicit-header frames: header geometry, field
    layout, block-FEC width and parity table, and demap offsets are all
    caller-supplied. Burst marks on the carrier key header parses to burst
    starts; without marks, frames are assumed back-to-back.

    An orchestration fast-path, not new vocabulary: the composition proof
    (tests/bits/test_css_explicit_composition.py) reproduces it bit-exactly
    from the generic pieces — demap / deinterleave / block-FEC / parse — in
    two agent-style passes."""

    name = "css_explicit_decode"
    engine = "coding"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "symbols"
    params_model = _ExplicitParams

    def emit_rx(self, b: CodingBuilder, params: Mapping[str, Any]) -> None:
        p = _ExplicitParams.model_validate(dict(params))
        b.add(
            css.css_explicit_decode_rx,
            sf=p.sf,
            header_cr=p.header_cr,
            reduced=p.reduced,
            header_symbols=p.header_symbols,
            header_nibbles=p.header_nibbles,
            sf_reduction=p.sf_reduction,
            header_data_bits=p.header_data_bits,
            header_parity=[int(x) for x in p.header_parity],
            field_payload_len=[int(x) for x in p.field_payload_len],
            field_cr=[int(x) for x in p.field_cr],
            field_has_crc=[int(x) for x in p.field_has_crc],
            field_parity=[int(x) for x in p.field_parity],
            data_bits=p.data_bits,
            crc_bytes=p.crc_bytes,
            parity_masks=[int(x) for x in p.parity_masks],
            reduced_offset=p.reduced_offset,
            full_offset=p.full_offset,
        )

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(
            Level.BITS, "b", in_desc.layout, Carrier.HARD, Amplitude.UNKNOWN
        )


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
    CssExplicitDecode,
    SyncSymbols,
    Normalize,
    MSlice,
    SymbolMap,
)
