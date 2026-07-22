from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.bits import framing, symbols
from marconi.bits.builder import ProgramBuilder
from marconi.core import coding
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import RxStage, Stage

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


class CssExplicitDecode(RxStage[ProgramBuilder]):
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
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "symbols"
    params_model = _ExplicitParams

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = _ExplicitParams.model_validate(dict(params))
        b.add(
            symbols.css_explicit_decode_rx,
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


class _SyncSymbolsParams(StageParams):
    pattern: list[int]
    max_errors: StrictInt = 0
    pre_symbols: StrictInt = 0


class SyncSymbols(RxStage[ProgramBuilder]):
    name = "sync_symbols"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    params_model = _SyncSymbolsParams
    accepts_item_type = "f"

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = _SyncSymbolsParams.model_validate(dict(params))
        b.add(
            framing.sync_symbols_rx,
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


class Normalize(RxStage[ProgramBuilder]):
    name = "normalize"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    params_model = _NormalizeParams
    accepts_item_type = "f"

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = _NormalizeParams.model_validate(dict(params))
        b.add(
            framing.normalize_rx,
            span_symbols=p.span_symbols,
            dc=p.dc,
            gain_percentile=p.gain_percentile,
        )


SYMBOL_STAGES: tuple[type[Stage[ProgramBuilder]], ...] = (
    CssExplicitDecode,
    SyncSymbols,
    Normalize,
)
