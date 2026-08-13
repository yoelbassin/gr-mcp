from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import Stage
from marconi.engine.types.bounds import (
    MAX_DELAY_ITEMS,
    MAX_FILTER_TAPS,
    MAX_OVERSAMPLE,
)
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class SymbolSyncStep(Step):
    conv: Literal["symbol_sync"] = "symbol_sync"
    # explicit rather than derived: the compiler needs rate_factor (1/sps)
    # from the params alone, before any stream exists
    sps: StrictInt = Field(
        ge=2,
        le=MAX_OVERSAMPLE * MAX_OVERSAMPLE,
        description="input samples per symbol the timing loop decimates by",
    )
    alpha: float = 0.35
    loop_bw: float = Field(
        default=0.045,
        ge=0,
        description=(
            "Symbol-timing loop bandwidth. 0 = OPEN-LOOP feedforward timing "
            "(Oerder-Meyr): a per-burst |x|^2 clock-line phase estimate "
            "(excluding the matched-filter transient at each burst edge) "
            "resampled to one sample per symbol, no loop - for bursty or short "
            "PSK where a Gardner loop cannot converge before the payload (the "
            "reverse-engineering case, with no known preamble to burn on "
            "acquisition; expect ~1 degraded-or-zeroed symbol at each burst "
            "edge). Open-loop requires sps>=4 (the |x|^2 line collapses "
            "toward Nyquist at sps=2) and pulse-shaping excess bandwidth "
            "(alpha>=0.1); staggered modulations (OQPSK/MSK-like) produce no "
            "|x|^2 symbol-rate line at all and read as no-signal zeros on "
            "this path. >0 = continuous Gardner loop for sustained signals."
        ),
    )
    # the RRC is round(sps*span)+1 taps long
    span: StrictInt = Field(default=11, ge=1, le=MAX_FILTER_TAPS)

    @model_validator(mode="after")
    def _ok(self) -> "SymbolSyncStep":
        if not 0.0 < self.alpha <= 1.0:
            raise PydanticCustomError("value_error", "alpha must be in (0, 1]")
        if self.loop_bw == 0.0 and self.alpha < 0.1:
            raise PydanticCustomError(
                "value_error",
                "open-loop timing (loop_bw=0) rides the |x|^2 clock line, "
                "which vanishes without excess bandwidth: alpha must be >= 0.1",
            )
        return self


SYMBOL_SYNC_OPEN_LOOP_HINT = (
    "symbol_sync ran closed-loop Gardner timing (loop_bw>0), which rails on "
    "short or bursty signals. Retry with symbol_sync "
    '{"loop_bw": 0} — open-loop feedforward (Oerder-Meyr) timing with '
    "per-burst acquisition; it needs an integer sps >= 4."
)


class SymbolSync(Stage[CompileContext, SymbolSyncStep]):
    name = "symbol_sync"
    min_input_sps = 2.0
    description = (
        "Symbol-timing recovery, IQ->IQ (RRC matched filter + timing), "
        "decimating to one sample per symbol. loop_bw>0 uses a closed-loop "
        "Gardner symbol_sync; loop_bw=0 uses open-loop feedforward "
        "(Oerder-Meyr) timing for bursty/short PSK. min_input_sps is "
        "conditional on loop_bw (compile-enforced): 2 closed-loop, 4 "
        "open-loop. This is psk_demod's timing half split out: it stays at "
        "IQ, so an equalizer (or any per-symbol conditioning) can sit at the "
        "1-sps seam before carrier recovery. The amplitude contract "
        "(normalized input, via an upstream agc) is unchanged across both "
        "modes."
    )
    from_level = Level.IQ
    to_level = Level.IQ
    family = "psk"
    accepts_amplitude = frozenset(
        {Amplitude.PEAK_UNITY, Amplitude.MEAN_MAG_UNITY, Amplitude.RMS_UNITY}
    )
    alters_amplitude = True
    step_model = SymbolSyncStep

    def emit_rx(self, b: CompileContext, step: SymbolSyncStep) -> None:
        b.chain(
            "rrc_filter_ccf",
            interpolation=1,
            rate=b.rate,
            sps=step.sps,
            alpha=step.alpha,
            span=step.span,
        )
        if step.loop_bw == 0.0:
            b.chain(
                "oerder_meyr_timing", sps=step.sps, span=step.span, alpha=step.alpha
            )
        else:
            b.chain("symbol_sync_cc", sps=step.sps, loop_bw=step.loop_bw)

    def rate_factor(self, step: SymbolSyncStep) -> float:
        return 1.0 / int(step.sps)

    def required_input_rate(
        self, step: SymbolSyncStep, symbol_rate: float
    ) -> float | None:
        return int(step.sps) * symbol_rate

    def min_input_sps_for(self, step: SymbolSyncStep) -> float | None:
        return 4.0 if step.loop_bw == 0.0 else None

    def retry_hints(
        self, step: SymbolSyncStep, path_convs: frozenset[str]
    ) -> list[str]:
        return [SYMBOL_SYNC_OPEN_LOOP_HINT] if step.loop_bw > 0.0 else []


class SampleSymbolsStep(Step):
    conv: Literal["sample_symbols"] = "sample_symbols"


class SampleSymbols(Stage[CompileContext, SampleSymbolsStep]):
    """IQ->SYMBOLS at 1 sps with no carrier loop: the timing-only rung.
    Emits no blocks - the level move is the whole stage - so an externally
    recovered or differentially absorbed carrier can reach the symbol seam."""

    name = "sample_symbols"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "psk"
    step_model = SampleSymbolsStep

    def emit_rx(self, b: CompileContext, step: SampleSymbolsStep) -> None:
        pass

    def out_descriptor(
        self, in_desc: Descriptor, step: SampleSymbolsStep
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT)

    def required_input_rate(
        self, step: SampleSymbolsStep, symbol_rate: float
    ) -> float | None:
        return symbol_rate

    def output_item_rate(
        self, step: SampleSymbolsStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return symbol_rate


class DifferentialDemodStep(Step):
    conv: Literal["differential_demod"] = "differential_demod"
    # delay_cc allocates the whole line up front
    delay: StrictInt = Field(default=1, ge=1, le=MAX_DELAY_ITEMS)
    rotate: float = 0.0
    out_order: PskOrder | None = Field(
        default=None,
        description=(
            "Alphabet size of the phase DIFFERENCE this stage emits, when it "
            "differs from the input's. Plain M-DPSK differences stay order M, "
            "so the default (null) carries the input order through. A "
            "staggered quaternary alphabet does not: its symbols occupy an "
            "order-8 grid, so the demod upstream must be told 8, while the "
            "consecutive-symbol differences take only 4 values — set "
            "out_order=4 there (with rotate=-pi/4 to land them on the plain "
            "QPSK grid), or the downstream demap's order check fails against "
            "an alphabet this stage no longer emits."
        ),
    )


class DifferentialDemod(Stage[CompileContext, DifferentialDemodStep]):
    """SYMBOLS->SYMBOLS continuous differential: z[i] = x[i] * conj(x[i-delay]),
    optionally rotated by exp(j*rotate). delay > 1 serves symbol-interleaved
    differential schemes. The output alphabet is the DIFFERENCE alphabet, which
    the input order alone does not determine — see out_order; for pi/4-shifted
    quaternary that is psk_demod order 8 -> differential_demod
    {rotate: -0.7853981633974483, out_order: 4} -> an order-4 demap."""

    name = "differential_demod"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "psk"
    step_model = DifferentialDemodStep
    accepts_item_type = ItemType.C
    accepts_carrier = Carrier.SOFT

    def out_descriptor(
        self, in_desc: Descriptor, step: DifferentialDemodStep
    ) -> Descriptor:
        # The one stage whose job IS changing the alphabet was the one
        # inheriting the base "same level, same order" passthrough, so the
        # pi/4-quaternary composition its own description advertises failed
        # the compiler's order check.
        if step.out_order is None:
            return replace(in_desc)
        return replace(in_desc, order=int(step.out_order))

    def emit_rx(self, b: CompileContext, step: DifferentialDemodStep) -> None:
        src = b.require_tail()
        dly = b.chain("delay_cc", samples=step.delay)
        mc = b.add("multiply_conjugate_cc")
        b.connect(src, mc, dst_port=0)
        b.connect(dly, mc, dst_port=1)
        b.set_tail(mc)
        if step.rotate:
            b.chain(
                "multiply_const_cc",
                re=math.cos(step.rotate),
                im=math.sin(step.rotate),
            )


class PskDemodStep(Step):
    conv: Literal["psk_demod"] = "psk_demod"
    order: PskOrder
    alpha: float = Field(default=0.35, gt=0.0, le=1.0)
    loop_bw: float = Field(
        default=0.045,
        description=(
            "Closed-loop bandwidth shared by the Gardner timing and Costas "
            "carrier loops; must be > 0. psk_demod has no open-loop mode — "
            "for open-loop timing use symbol_sync loop_bw=0 followed by a "
            "demap, not psk_demod."
        ),
    )
    span: StrictInt = Field(default=11, ge=1, le=MAX_FILTER_TAPS)

    @model_validator(mode="after")
    def _loops_run(self) -> "PskDemodStep":
        if self.loop_bw <= 0.0:
            raise PydanticCustomError(
                "value_error",
                "psk_demod has no open-loop mode: loop_bw<=0 freezes the "
                "Gardner and Costas loops, and under residual CFO a spinning "
                "constellation hard-slices into confident garbage that the "
                "constant-modulus false-lock check cannot see. For open-loop "
                "timing use symbol_sync loop_bw=0 followed by a demap",
            )
        return self


class PskDemod(Stage[CompileContext, PskDemodStep]):
    """Coherent linear demod, IQ<->SYMBOLS. RX: RRC matched filter + Gardner
    symbol timing + costas carrier recovery -> soft complex symbols. RRC
    pulse-shape/upsample. rate_factor stays 1.0 (symbol decimation is internal
    via ctx.sps; the modulator reads the IQ-domain rate)."""

    name = "psk_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "psk"
    # MEASURED across a 1e-3..1e3 gain sweep: BER 0 for every statistic. A
    # constant-modulus constellation has no absolute radius to hit; the
    # requirement is only that SOME normalization bounded the scale.
    accepts_amplitude = frozenset(
        {Amplitude.PEAK_UNITY, Amplitude.MEAN_MAG_UNITY, Amplitude.RMS_UNITY}
    )
    min_input_sps = 2.0
    step_model = PskDemodStep

    def emit_rx(self, b: CompileContext, step: PskDemodStep) -> None:
        b.chain(
            "rrc_filter_ccf",
            interpolation=1,
            rate=b.rate,
            sps=b.sps,
            alpha=step.alpha,
            span=step.span,
        )
        b.chain("symbol_sync_cc", sps=b.sps, loop_bw=step.loop_bw)
        b.chain("costas_loop_cc", order=int(step.order), loop_bw=step.loop_bw)

    def out_descriptor(self, in_desc: Descriptor, step: PskDemodStep) -> Descriptor:
        return Descriptor(
            Level.SYMBOLS, ItemType.C, Carrier.SOFT, order=int(step.order)
        )

    def output_item_rate(
        self, step: PskDemodStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return symbol_rate


class PskDemapStep(Step):
    conv: Literal["psk_demap"] = "psk_demap"
    order: PskOrder


class PskDemap(Stage[CompileContext, PskDemapStep]):
    """Constellation demap, SYMBOLS->BITS: hard min-distance decode + unpack to
    bits. Hard bits at the seam (mirrors the general Slice)."""

    name = "psk_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "psk"
    step_model = PskDemapStep
    accepts_item_type = ItemType.C
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: PskDemapStep) -> None:
        order = int(step.order)
        b.chain("constellation_decoder_cb", scheme="psk", order=order)
        b.chain("unpack_k_bits_bb", k=int(math.log2(order)))

    def out_descriptor(self, in_desc: Descriptor, step: PskDemapStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.B, Carrier.HARD)

    def required_input_order(self, step: PskDemapStep) -> int | None:
        return int(step.order)

    def output_item_rate(
        self, step: PskDemapStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return in_rate * math.log2(int(step.order))


class PskSoftDemapStep(Step):
    conv: Literal["psk_soft_demap"] = "psk_soft_demap"
    order: PskOrder


class PskSoftDemap(Stage[CompileContext, PskSoftDemapStep]):
    """Constellation soft demap, SYMBOLS->BITS soft. RX-only: stock
    constellation_soft_decoder, one LLR per bit. The soft counterpart of
    psk_demap, and the single-carrier entry point to the soft coding lane
    (deinterleave/depuncture/fec).

    MEASURED: constellation_soft_decoder emits log P(1)/P(0) -- BPSK point +1
    (hard bit 1) reads +4.0 -- while both soft consumers in this engine use the
    opposite sign (trellis_viterbi's Euclidean table maps encoder bit 0 to +1.0,
    and core.bitfile.harden is llr<0). Without the negation the lane decodes to
    noise on NOISELESS symbols (measured match 0.508); with it, BER 0."""

    name = "psk_soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "psk"
    step_model = PskSoftDemapStep
    accepts_item_type = ItemType.C
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: PskSoftDemapStep) -> None:
        b.chain("constellation_soft_decoder", scheme="psk", order=step.order)
        b.chain_llr_flip()

    def out_descriptor(self, in_desc: Descriptor, step: PskSoftDemapStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.F, Carrier.SOFT)

    def required_input_order(self, step: PskSoftDemapStep) -> int | None:
        return int(step.order)

    def output_item_rate(
        self, step: PskSoftDemapStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return in_rate * math.log2(int(step.order))


PSK_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (
    SymbolSync,
    SampleSymbols,
    DifferentialDemod,
    PskDemod,
    PskDemap,
    PskSoftDemap,
)
