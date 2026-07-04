from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import StrictFloat, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage, RxStage, Stage
from marconi.phy.compile_context import CompileContext
from marconi.phy.modulation.css import coding

_SF_MIN, _SF_MAX = 5, 14
_OSR_MAX = 8


class _CssParams(StageParams):
    sf: StrictInt
    oversample: StrictInt = 2
    zero_pad: StrictInt = 4
    preamble_len: StrictInt = 8
    sfd_symbols: StrictFloat = 2.25
    sync_symbols: StrictInt = 2

    @model_validator(mode="after")
    def _ok(self) -> "_CssParams":
        if not (_SF_MIN <= self.sf <= _SF_MAX):
            raise PydanticCustomError(
                "value_error",
                "sf {sf} out of range [{lo}, {hi}]",
                {"sf": self.sf, "lo": _SF_MIN, "hi": _SF_MAX, "field": "sf"},
            )
        if not (1 <= self.oversample <= _OSR_MAX):
            raise PydanticCustomError(
                "value_error",
                "oversample {oversample} out of range [1, {hi}]",
                {"oversample": self.oversample, "hi": _OSR_MAX, "field": "oversample"},
            )
        if self.zero_pad < 1:
            raise PydanticCustomError("value_error", "zero_pad must be >= 1")
        if self.preamble_len < 5:
            raise PydanticCustomError("value_error", "preamble_len must be >= 5")
        if self.sfd_symbols < 1.0:
            raise PydanticCustomError("value_error", "sfd_symbols must be >= 1")
        if not (1 <= self.sync_symbols < self.preamble_len - 2):
            raise PydanticCustomError(
                "value_error",
                "sync_symbols {sync_symbols} must be in [1, preamble_len-3]",
                {"sync_symbols": self.sync_symbols, "field": "sync_symbols"},
            )
        return self


class ChirpSync(DuplexStage[CompileContext]):
    """CSS acquisition, IQ<->IQ. TX prepends a preamble (up-chirps) and a
    start-of-frame delimiter (down-chirps). RX detects the preamble, aligns on
    the SFD, derotates the carrier offset, and strips to payload. The SFD length
    and the count of sync symbols between preamble and SFD are parameters. The
    CSS analog of preamble_sync."""

    name = "chirp_sync"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "css"
    params_model = _CssParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _CssParams.model_validate(dict(params))
        b.chain(
            "chirp_sync",
            sf=p.sf,
            oversample=p.oversample,
            zero_pad=p.zero_pad,
            preamble_len=p.preamble_len,
            bandwidth=b.symbol_rate * (1 << p.sf),
            sfd_symbols=p.sfd_symbols,
            sync_symbols=p.sync_symbols,
        )

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _CssParams.model_validate(dict(params))
        b.chain(
            "chirp_prepend",
            sf=p.sf,
            oversample=p.oversample,
            preamble_len=p.preamble_len,
            sfd_symbols=p.sfd_symbols,
        )


class Dechirp(DuplexStage[CompileContext]):
    """CSS demod, IQ<->SYMBOLS. RX dechirps each symbol window (FFT, fold, argmax)
    to a hard symbol index (int16) -- decision-directed, so HARD at the seam (the
    hard@SYMBOLS rule, enforced by accepts_carrier). TX modulates symbol indices to
    chirps."""

    name = "dechirp"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "css"
    params_model = _CssParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _CssParams.model_validate(dict(params))
        b.chain("chirp_demod", sf=p.sf, oversample=p.oversample, zero_pad=p.zero_pad)

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _CssParams.model_validate(dict(params))
        b.chain("chirp_mod", sf=p.sf, oversample=p.oversample)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "s", in_desc.layout, Carrier.HARD)

    def required_input_rate(
        self, params: Mapping[str, Any], symbol_rate: float
    ) -> float | None:
        # chirp_demod's window is a fixed oversample*2^sf samples per symbol, so
        # the input must arrive at exactly that many samples per symbol.
        p = _CssParams.model_validate(dict(params))
        return p.oversample * (1 << p.sf) * symbol_rate


class CssDemap(DuplexStage[CompileContext]):
    """CSS symbol<->bits, SYMBOLS<->BITS. RX Gray-decodes the symbol index and
    unpacks sf bits (MSB-first). TX packs sf bits + Gray-encodes."""

    name = "css_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "css"
    params_model = _CssParams
    accepts_item_type = "s"
    accepts_carrier = Carrier.HARD

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("css_demap", sf=int(params["sf"]))

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("css_map", sf=int(params["sf"]))

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


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


class CssExplicitDecode(RxStage[CompileContext]):
    """CSS explicit-header decode, SYMBOLS->BITS (RX-only). Parses the explicit
    header, decodes the payload at its own code rate, emits de-FEC'd payload
    bits. Generic over CSS explicit-header frames: header geometry (symbol/
    nibble counts, reduced-rate delta), field layout, the block-FEC data width
    and parity table, and the demap offsets are all caller-supplied."""

    name = "css_explicit_decode"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "css"
    params_model = _ExplicitParams
    accepts_item_type = "s"
    accepts_carrier = Carrier.HARD

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _ExplicitParams.model_validate(dict(params))
        b.chain(
            "css_explicit_decode",
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
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


CSS_STAGES: tuple[type[Stage[CompileContext]], ...] = (
    ChirpSync,
    Dechirp,
    CssDemap,
    CssExplicitDecode,
)
