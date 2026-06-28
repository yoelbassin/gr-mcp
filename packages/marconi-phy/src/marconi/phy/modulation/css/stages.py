from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage
from marconi.phy.compile_context import CompileContext

_SF_MIN, _SF_MAX = 5, 14


class _CssParams(StageParams):
    sf: StrictInt
    oversample: StrictInt = 2
    zero_pad: StrictInt = 4
    preamble_len: StrictInt = 8

    @model_validator(mode="after")
    def _ok(self) -> "_CssParams":
        if not (_SF_MIN <= self.sf <= _SF_MAX):
            raise PydanticCustomError(
                "value_error",
                "sf {sf} out of range [{lo}, {hi}]",
                {"sf": self.sf, "lo": _SF_MIN, "hi": _SF_MAX, "field": "sf"},
            )
        if self.oversample < 1:
            raise PydanticCustomError("value_error", "oversample must be >= 1")
        if self.zero_pad < 1:
            raise PydanticCustomError("value_error", "zero_pad must be >= 1")
        if self.preamble_len < 3:
            raise PydanticCustomError("value_error", "preamble_len must be >= 3")
        return self


class ChirpSync(DuplexStage[CompileContext]):
    """CSS acquisition, IQ<->IQ. TX prepends preamble (up-chirps) + SFD
    (2.25 down-chirps). RX detects the preamble, aligns on the SFD, derotates the
    carrier offset, and strips to payload. The CSS analog of preamble_sync."""

    name = "chirp_sync"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "css"
    params_model = _CssParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        sf = int(params["sf"])
        b.chain(
            "chirp_sync",
            sf=sf,
            oversample=int(params.get("oversample", 2)),
            zero_pad=int(params.get("zero_pad", 4)),
            preamble_len=int(params.get("preamble_len", 8)),
            bandwidth=b.symbol_rate * (1 << sf),
        )

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "chirp_prepend",
            sf=int(params["sf"]),
            oversample=int(params.get("oversample", 2)),
            preamble_len=int(params.get("preamble_len", 8)),
        )


class Dechirp(DuplexStage[CompileContext]):
    """CSS demod, IQ<->SYMBOLS. RX dechirps each symbol window (FFT, fold, argmax)
    to a hard symbol index (int16) -- decision-directed, so HARD at the seam (the
    QAM precedent). TX modulates symbol indices to chirps."""

    name = "dechirp"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "css"
    params_model = _CssParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "chirp_demod",
            sf=int(params["sf"]),
            oversample=int(params.get("oversample", 2)),
            zero_pad=int(params.get("zero_pad", 4)),
        )

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "chirp_mod",
            sf=int(params["sf"]),
            oversample=int(params.get("oversample", 2)),
        )

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "s", in_desc.layout, Carrier.HARD)


class CssDemap(DuplexStage[CompileContext]):
    """CSS symbol<->bits, SYMBOLS<->BITS. RX Gray-decodes the symbol index and
    unpacks sf bits (MSB-first). TX packs sf bits + Gray-encodes."""

    name = "css_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "css"
    params_model = _CssParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("css_demap", sf=int(params["sf"]))

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("css_map", sf=int(params["sf"]))

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


CSS_STAGES: tuple[type[DuplexStage[CompileContext]], ...] = (
    ChirpSync,
    Dechirp,
    CssDemap,
)
