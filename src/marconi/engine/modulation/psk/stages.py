from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, RxStage, Stage
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class SymbolSyncStep(Step):
    conv: Literal["symbol_sync"] = "symbol_sync"
    sps: StrictInt  # explicit: rate_factor (1/sps) must be params-derivable
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
            "acquisition). Open-loop requires sps>=4 (the |x|^2 line collapses "
            "toward Nyquist at sps=2); the closed loop needs sps>=2. >0 = "
            "continuous Gardner loop for sustained signals."
        ),
    )
    span: StrictInt = 11

    @model_validator(mode="after")
    def _ok(self) -> "SymbolSyncStep":
        if self.sps < 2:
            raise PydanticCustomError(
                "value_error", "sps must be >= 2 to recover symbol timing"
            )
        if not 0.0 < self.alpha <= 1.0:
            raise PydanticCustomError("value_error", "alpha must be in (0, 1]")
        if self.span < 1:
            raise PydanticCustomError("value_error", "span must be >= 1")
        return self


class SymbolSync(RxStage[CompileContext, SymbolSyncStep]):
    """Symbol-timing recovery, IQ->IQ: RRC matched filter, decimating an
    oversampled stream to one sample per symbol, followed by either a
    closed-loop Gardner symbol_sync (loop_bw>0) or open-loop feedforward
    (Oerder-Meyr) timing (loop_bw=0). This is psk_demod's timing half split
    out, so a 1-sps seam opens where an equalizer (or any per-symbol
    conditioning) can sit before carrier recovery. Unlike psk_demod -- which
    hides its sps decimation inside the IQ->SYMBOLS level change -- this
    stays at IQ and decimates honestly: rate_factor is 1/sps and
    required_input_rate makes the explicit sps agree with the delivered rate.
    RX-only; output amplitude is UNKNOWN (the filters rescale)."""

    name = "symbol_sync"
    min_input_sps = 2.0
    description = (
        "Symbol-timing recovery, IQ->IQ (RRC matched filter + timing). "
        "loop_bw>0 uses a closed-loop Gardner symbol_sync; loop_bw=0 uses "
        "open-loop feedforward (Oerder-Meyr) timing for bursty/short PSK. The "
        "min_input_sps shown in describe_stages is the closed-loop default "
        "(2); the open-loop path (loop_bw=0) requires sps>=4. The amplitude "
        "contract (normalized input, via an upstream agc) is unchanged across "
        "both modes."
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
            b.chain("oerder_meyr_timing", sps=step.sps, span=step.span)
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


class SampleSymbolsStep(Step):
    conv: Literal["sample_symbols"] = "sample_symbols"


class SampleSymbols(RxStage[CompileContext, SampleSymbolsStep]):
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


class DifferentialDemodStep(Step):
    conv: Literal["differential_demod"] = "differential_demod"
    delay: StrictInt = Field(default=1, ge=1)
    rotate: float = 0.0


class DifferentialDemod(RxStage[CompileContext, DifferentialDemodStep]):
    """SYMBOLS->SYMBOLS continuous differential: z[i] = x[i] * conj(x[i-delay]),
    optionally rotated by exp(j*rotate). With rotate=-pi/4 this lands pi/4-DQPSK
    on the plain QPSK constellation; delay > 1 serves symbol-interleaved
    differential schemes."""

    name = "differential_demod"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "psk"
    step_model = DifferentialDemodStep
    accepts_item_type = ItemType.C
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: DifferentialDemodStep) -> None:
        src = b.tail
        assert src is not None
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
    order: PskOrder  # required: an explicit PSK order, no silent default modulation
    alpha: float = 0.35
    loop_bw: float = 0.045
    span: StrictInt = 11


class PskDemod(DuplexStage[CompileContext, PskDemodStep]):
    """Coherent linear demod, IQ<->SYMBOLS. RX: RRC matched filter + Gardner
    symbol timing + costas carrier recovery -> soft complex symbols. TX: RRC
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

    def emit_tx(self, b: CompileContext, step: PskDemodStep) -> None:
        b.chain(
            "rrc_filter_ccf",
            interpolation=b.sps_int(),
            rate=b.rate,
            sps=b.sps,
            alpha=step.alpha,
            span=step.span,
        )

    def out_descriptor(self, in_desc: Descriptor, step: PskDemodStep) -> Descriptor:
        return Descriptor(
            Level.SYMBOLS, ItemType.C, Carrier.SOFT, order=int(step.order)
        )


class PskDemapStep(Step):
    conv: Literal["psk_demap"] = "psk_demap"
    order: PskOrder  # required


class PskDemap(DuplexStage[CompileContext, PskDemapStep]):
    """Constellation map/demap, SYMBOLS<->BITS. RX: hard min-distance decode +
    unpack to bits. TX: pack bits into symbol indices + map to constellation
    points. Hard bits at the seam (mirrors the general Slice)."""

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

    def emit_tx(self, b: CompileContext, step: PskDemapStep) -> None:
        order = int(step.order)
        b.chain("pack_k_bits_bb", k=int(math.log2(order)))
        b.chain("chunks_to_symbols_bc", scheme="psk", order=order)

    def out_descriptor(self, in_desc: Descriptor, step: PskDemapStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.B, Carrier.HARD)

    def required_input_order(self, step: PskDemapStep) -> int | None:
        return int(step.order)


class PskSoftDemapStep(Step):
    conv: Literal["psk_soft_demap"] = "psk_soft_demap"
    order: PskOrder


class PskSoftDemap(RxStage[CompileContext, PskSoftDemapStep]):
    """Constellation soft demap, SYMBOLS->BITS soft. RX-only: stock
    constellation_soft_decoder, one LLR per bit. The soft counterpart of
    psk_demap, and the single-carrier entry point to the soft coding lane
    (deinterleave/depuncture/fec), which until now only dqpsk_soft_demap could
    feed.

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
        b.chain("multiply_const_ff", value=-1.0)

    def out_descriptor(self, in_desc: Descriptor, step: PskSoftDemapStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.F, Carrier.SOFT)

    def required_input_order(self, step: PskSoftDemapStep) -> int | None:
        return int(step.order)


PSK_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (
    SymbolSync,
    SampleSymbols,
    DifferentialDemod,
    PskDemod,
    PskDemap,
    PskSoftDemap,
)
