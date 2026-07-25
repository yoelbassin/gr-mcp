from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import StrictFloat, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.params import StageParams

_SF_MIN, _SF_MAX = 5, 14
_OSR_MAX = 8


def _check_sf(sf: int) -> None:
    if not (_SF_MIN <= sf <= _SF_MAX):
        raise PydanticCustomError(
            "value_error",
            "sf {sf} out of range [{lo}, {hi}]",
            {"sf": sf, "lo": _SF_MIN, "hi": _SF_MAX, "field": "sf"},
        )


def _check_osr_pad(oversample: int, zero_pad: int) -> None:
    if not (1 <= oversample <= _OSR_MAX):
        raise PydanticCustomError(
            "value_error",
            "oversample {oversample} out of range [1, {hi}]",
            {"oversample": oversample, "hi": _OSR_MAX, "field": "oversample"},
        )
    if zero_pad < 1:
        raise PydanticCustomError("value_error", "zero_pad must be >= 1")


class _CssSfParams(StageParams):
    sf: StrictInt

    @model_validator(mode="after")
    def _ok(self) -> "_CssSfParams":
        _check_sf(self.sf)
        return self


class _DechirpParams(StageParams):
    sf: StrictInt
    oversample: StrictInt
    zero_pad: StrictInt

    @model_validator(mode="after")
    def _ok(self) -> "_DechirpParams":
        _check_sf(self.sf)
        _check_osr_pad(self.oversample, self.zero_pad)
        return self


class _CssParams(StageParams):
    sf: StrictInt
    oversample: StrictInt
    zero_pad: StrictInt
    preamble_len: StrictInt
    sfd_symbols: StrictFloat
    sync_symbols: StrictInt

    @model_validator(mode="after")
    def _ok(self) -> "_CssParams":
        _check_sf(self.sf)
        _check_osr_pad(self.oversample, self.zero_pad)
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
    params_model = _DechirpParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _DechirpParams.model_validate(dict(params))
        n = 1 << p.sf
        sn = p.oversample * n
        bins = n * p.zero_pad
        fft_len = p.oversample * n * p.zero_pad

        dechirped = b.chain("multiply_cc")
        ref = b.add("chirp_ref_source", sf=p.sf, oversample=p.oversample)
        b.connect(ref, dechirped, dst_port=1)
        padded = b.chain("stream_mux_c", lengths=[sn, fft_len - sn])
        zeros = b.add("null_source_c")
        b.connect(zeros, padded, dst_port=1)
        b.chain("stream_to_vector", vlen=fft_len)
        b.chain("fft_vcc", fft_len=fft_len)
        b.chain("complex_to_mag", vlen=fft_len)
        spectrum = b.chain("vector_to_stream_f", vlen=fft_len)
        low_image = b.add("keep_m_in_n_f", m=bins, n=fft_len)
        high_image = b.add(
            "keep_m_in_n_f",
            m=bins,
            n=fft_len,
            offset=fft_len - bins,
            propagate_tags=False,
        )
        folded = b.add("add_ff")
        b.connect(spectrum, low_image)
        b.connect(spectrum, high_image)
        b.connect(low_image, folded, dst_port=0)
        b.connect(high_image, folded, dst_port=1)
        b.set_tail(folded)
        b.chain("stream_to_vector_f", vlen=bins)
        b.chain("peak_decision", vlen=bins, divisor=p.zero_pad, modulo=n)

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _DechirpParams.model_validate(dict(params))
        b.chain("chirp_mod", sf=p.sf, oversample=p.oversample)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "s", in_desc.layout, Carrier.HARD)

    def required_input_rate(
        self, params: Mapping[str, Any], symbol_rate: float
    ) -> float | None:
        # the dechirp window is a fixed oversample*2^sf samples per symbol, so
        # the input must arrive at exactly that many samples per symbol.
        p = _DechirpParams.model_validate(dict(params))
        return p.oversample * (1 << p.sf) * symbol_rate


class CssDemap(DuplexStage[CompileContext]):
    """CSS symbol<->bits, SYMBOLS<->BITS. RX Gray-decodes the symbol index and
    unpacks sf bits (MSB-first). TX packs sf bits + Gray-encodes."""

    name = "css_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "css"
    params_model = _CssSfParams
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


CSS_STAGES: tuple[type[Stage[CompileContext]], ...] = (
    ChirpSync,
    Dechirp,
    CssDemap,
)
