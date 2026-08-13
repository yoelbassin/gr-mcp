from __future__ import annotations

from typing import Any, Literal

from pydantic import StrictFloat, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, Stage
from marconi.engine.types.bounds import (
    MAX_CHIRP_PREFIX_SAMPLES,
    MAX_DECHIRP_FFT,
)
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step

_SF_MIN, _SF_MAX = 5, 14
_OSR_MAX = 8


def _check_sf(sf: int) -> None:
    if not (_SF_MIN <= sf <= _SF_MAX):
        raise PydanticCustomError(
            "value_error",
            "sf {sf} out of range [{lo}, {hi}]",
            {"sf": sf, "lo": _SF_MIN, "hi": _SF_MAX, "field": "sf"},
        )


def _check_osr_pad(sf: int, oversample: int, zero_pad: int) -> None:
    if not (1 <= oversample <= _OSR_MAX):
        raise PydanticCustomError(
            "value_error",
            "oversample {oversample} out of range [1, {hi}]",
            {"oversample": oversample, "hi": _OSR_MAX, "field": "oversample"},
        )
    if zero_pad < 1:
        raise PydanticCustomError("value_error", "zero_pad must be >= 1")
    # The dechirp FFT is oversample * 2**sf * zero_pad long and one is taken
    # per symbol window, so the PRODUCT is what has to be bounded — sf and
    # oversample are capped above, but zero_pad alone was not, and no single
    # field's ceiling expresses the FFT this triple asks for.
    fft_len = oversample * (1 << sf) * zero_pad
    if fft_len > MAX_DECHIRP_FFT:
        raise PydanticCustomError(
            "value_error",
            "oversample {oversample} x 2**sf {n} x zero_pad {zero_pad} is a "
            "{fft_len}-point FFT per symbol window; {max} is the ceiling",
            {
                "oversample": oversample,
                "n": 1 << sf,
                "zero_pad": zero_pad,
                "fft_len": fft_len,
                "max": MAX_DECHIRP_FFT,
                "field": "zero_pad",
            },
        )


class CssDemapStep(Step):
    conv: Literal["css_demap"] = "css_demap"
    sf: StrictInt

    @model_validator(mode="after")
    def _check_sf(self) -> "CssDemapStep":
        _check_sf(self.sf)
        return self


class DechirpStep(Step):
    conv: Literal["dechirp"] = "dechirp"
    sf: StrictInt
    oversample: StrictInt
    zero_pad: StrictInt

    @model_validator(mode="after")
    def _ok(self) -> "DechirpStep":
        _check_sf(self.sf)
        _check_osr_pad(self.sf, self.oversample, self.zero_pad)
        return self


class ChirpSyncStep(Step):
    conv: Literal["chirp_sync"] = "chirp_sync"
    sf: StrictInt
    oversample: StrictInt
    zero_pad: StrictInt
    preamble_len: StrictInt
    sfd_symbols: StrictFloat
    sync_symbols: StrictInt

    @model_validator(mode="after")
    def _ok(self) -> "ChirpSyncStep":
        _check_sf(self.sf)
        _check_osr_pad(self.sf, self.oversample, self.zero_pad)
        if self.preamble_len < 5:
            raise PydanticCustomError("value_error", "preamble_len must be >= 5")
        # the TX prefix materializes this many complex samples as one Python
        # list on its way into vector_insert_c
        prefix = round(
            (self.preamble_len + self.sfd_symbols) * self.oversample * (1 << self.sf)
        )
        if prefix > MAX_CHIRP_PREFIX_SAMPLES:
            raise PydanticCustomError(
                "value_error",
                "(preamble_len {preamble_len} + sfd_symbols {sfd}) x oversample "
                "x 2**sf is a {prefix}-sample prefix; {max} is the ceiling",
                {
                    "preamble_len": self.preamble_len,
                    "sfd": self.sfd_symbols,
                    "prefix": prefix,
                    "max": MAX_CHIRP_PREFIX_SAMPLES,
                    "field": "preamble_len",
                },
            )
        if self.sfd_symbols != 0.0 and self.sfd_symbols < 1.0:
            raise PydanticCustomError(
                "value_error", "sfd_symbols must be 0 (no SFD) or >= 1"
            )
        if not (0 <= self.sync_symbols <= self.preamble_len - 3):
            raise PydanticCustomError(
                "value_error",
                "sync_symbols {sync_symbols} must be in [0, preamble_len-3]",
                {"sync_symbols": self.sync_symbols, "field": "sync_symbols"},
            )
        return self


class ChirpSync(DuplexStage[CompileContext, ChirpSyncStep]):
    """CSS acquisition, IQ<->IQ. TX prepends a preamble (up-chirps) and an
    optional start-of-frame delimiter (down-chirps). RX detects the preamble,
    aligns on the SFD when one is declared (sfd_symbols >= 1) or on the end of
    the preamble run when none is (sfd_symbols = 0, where the CFO/STO split is
    resolvable only to within one bin), derotates the carrier offset, and
    strips to payload. sync_symbols is the count of arbitrary-bin symbols
    between the preamble run and the SFD (or payload). The CSS analog of
    preamble_sync."""

    name = "chirp_sync"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "css"
    step_model = ChirpSyncStep

    def emit_rx(self, b: CompileContext, step: ChirpSyncStep) -> None:
        b.chain(
            "chirp_sync",
            sf=step.sf,
            oversample=step.oversample,
            zero_pad=step.zero_pad,
            preamble_len=step.preamble_len,
            bandwidth=b.symbol_rate * (1 << step.sf),
            sfd_symbols=step.sfd_symbols,
            sync_symbols=step.sync_symbols,
        )

    def emit_tx(self, b: CompileContext, step: ChirpSyncStep) -> None:
        b.chain(
            "chirp_prepend",
            sf=step.sf,
            oversample=step.oversample,
            preamble_len=step.preamble_len,
            sfd_symbols=step.sfd_symbols,
        )


class Dechirp(DuplexStage[CompileContext, DechirpStep]):
    """CSS demod, IQ<->SYMBOLS. RX dechirps each symbol window (FFT, fold, argmax)
    to a hard symbol index (int16) -- decision-directed, so HARD at the seam (the
    hard@SYMBOLS rule, enforced by accepts_carrier). TX modulates symbol indices to
    chirps."""

    name = "dechirp"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "css"
    step_model = DechirpStep

    def emit_rx(self, b: CompileContext, step: DechirpStep) -> None:
        n = 1 << step.sf
        sn = step.oversample * n
        bins = n * step.zero_pad
        fft_len = step.oversample * n * step.zero_pad

        dechirped = b.chain("multiply_cc")
        ref = b.add("chirp_ref_source", sf=step.sf, oversample=step.oversample)
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
        b.chain("peak_decision", vlen=bins, divisor=step.zero_pad, modulo=n)

    def emit_tx(self, b: CompileContext, step: DechirpStep) -> None:
        b.chain("chirp_mod", sf=step.sf, oversample=step.oversample)

    def out_descriptor(self, in_desc: Descriptor, step: DechirpStep) -> Descriptor:
        return Descriptor(
            Level.SYMBOLS, ItemType.S, Carrier.HARD, order=1 << int(step.sf)
        )

    def required_input_rate(
        self, step: DechirpStep, symbol_rate: float
    ) -> float | None:
        # the dechirp window is a fixed oversample*2^sf samples per symbol, so
        # the input must arrive at exactly that many samples per symbol.
        return step.oversample * (1 << step.sf) * symbol_rate

    def output_item_rate(
        self, step: DechirpStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return symbol_rate


class CssDemap(DuplexStage[CompileContext, CssDemapStep]):
    """CSS symbol<->bits, SYMBOLS<->BITS. RX Gray-decodes the symbol index and
    unpacks sf bits (MSB-first). TX packs sf bits + Gray-encodes."""

    name = "css_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "css"
    step_model = CssDemapStep
    accepts_item_type = ItemType.S
    accepts_carrier = Carrier.HARD

    def emit_rx(self, b: CompileContext, step: CssDemapStep) -> None:
        b.chain("css_demap", sf=int(step.sf))

    def emit_tx(self, b: CompileContext, step: CssDemapStep) -> None:
        b.chain("css_map", sf=int(step.sf))

    def out_descriptor(self, in_desc: Descriptor, step: CssDemapStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.B, Carrier.HARD)

    def required_input_order(self, step: CssDemapStep) -> int | None:
        return 1 << int(step.sf)

    def output_item_rate(
        self, step: CssDemapStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return in_rate * int(step.sf)


CSS_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (
    ChirpSync,
    Dechirp,
    CssDemap,
)
