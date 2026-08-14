from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import Stage
from marconi.engine.types.bounds import MAX_OVERSAMPLE
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class FskStep(Step):
    conv: Literal["fsk"] = "fsk"
    deviation: float = Field(gt=0)
    loop_bw: float = Field(
        default=0.045,
        ge=0,
        description=(
            "Gardner symbol-timing loop bandwidth. 0 samples open-loop at the "
            "nominal sps with no timing feedback — the mode short or bursty "
            "packets need, since closed-loop timing rails on bursts under "
            "~1000 symbols and recovers almost nothing. Use >0 to track clock "
            "drift on long continuous signals."
        ),
    )


FSK_OPEN_LOOP_HINT = (
    "fsk ran closed-loop Gardner symbol timing (loop_bw>0), which rails on short "
    "or bursty packets and recovers few clean symbols. If this capture is bursty "
    "or the packets are short (under ~1000 symbols), retry with fsk "
    '{"loop_bw": 0} — open-loop, fixed-rate sampling at the nominal sps.'
)


class Fsk(Stage[CompileContext, FskStep]):
    """Binary FSK carrier stage. RX: FM discriminator + Gardner symbol timing ->
    one soft float per symbol. The symbol-rate
    decimation is internal (driven by ctx.sps); rate_factor stays 1.0 so the
    modulator reads the IQ-domain sample rate (rate-model invariant)."""

    name = "fsk"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "fsk"
    # 4.0, not the Gardner-generic 2.0: the discriminator output is a
    # RECTANGULAR waveform (no excess-bandwidth shaping), where Gardner's
    # TED is degenerate at 2 sps - measured BER 0.480 median over 10 seeds
    # at sps 2 AND 3 on noiseless input with status ok, 0.000 at 4. The
    # RRC-shaped psk_demod genuinely works at 2 and keeps its floor.
    min_input_sps = 4.0
    step_model = FskStep

    def emit_rx(self, b: CompileContext, step: FskStep) -> None:
        gain = b.rate / (2.0 * math.pi * step.deviation)
        b.chain("quadrature_demod", gain=gain)
        b.chain("symbol_sync_ff", sps=b.sps, loop_bw=step.loop_bw)

    def out_descriptor(self, in_desc: Descriptor, step: FskStep) -> Descriptor:
        return Descriptor(Level.SYMBOLS, ItemType.F, Carrier.SOFT)

    def output_item_rate(
        self, step: FskStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return symbol_rate

    def retry_hints(self, step: FskStep, path_convs: frozenset[str]) -> list[str]:
        return [FSK_OPEN_LOOP_HINT] if step.loop_bw > 0.0 else []


MSK_LOOP_BW_DEFAULT = 0.0038


class MskStep(Step):
    conv: Literal["msk"] = "msk"
    loop_bw: float = Field(default=MSK_LOOP_BW_DEFAULT, ge=0.0)
    loop_pole: float = Field(default=0.52, ge=0.0, le=1.0)
    # the matched-filter bank is (2*sps+1) x mf_oversample wide
    mf_oversample: StrictInt = Field(default=12, ge=1, le=MAX_OVERSAMPLE)


class Msk(Stage[CompileContext, MskStep]):
    """Coherent MSK (h=0.5 CPFSK) demod -> one soft float per symbol. RX-only:
    an MSK transmitter is the fsk stage at deviation = symbol_rate/4 (h=0.5),
    so no new vocabulary is needed. Coherent detection buys ~2-3 dB at operational
    BER over the best stock differential detector, which stock GR cannot beat (no
    coherent MSK receiver); guarded by
    tests/integration/engine/modulation/test_msk_snr_margin.py."""

    name = "msk"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "fsk"
    min_input_sps = 2.0
    polarity_ambiguous = True  # coherent h=0.5 detection: 180 deg phase ambiguity
    step_model = MskStep

    def emit_rx(self, b: CompileContext, step: MskStep) -> None:
        b.chain(
            "msk_demod",
            sps=b.sps,
            loop_bw=step.loop_bw,
            loop_pole=step.loop_pole,
            mf_oversample=step.mf_oversample,
        )

    def out_descriptor(self, in_desc: Descriptor, step: MskStep) -> Descriptor:
        return Descriptor(Level.SYMBOLS, ItemType.F, Carrier.SOFT)

    def output_item_rate(
        self, step: MskStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return symbol_rate


class MfskSoftDemapStep(Step):
    conv: Literal["mfsk_soft_demap"] = "mfsk_soft_demap"
    levels: list[float]

    @model_validator(mode="after")
    def _ok(self) -> "MfskSoftDemapStep":
        m = len(self.levels)
        if m < 2 or m & (m - 1):
            raise PydanticCustomError(
                "value_error",
                "levels needs a power-of-two count >= 2, got {m}",
                {"m": m, "field": "levels"},
            )
        if len(set(self.levels)) != m:
            raise PydanticCustomError("value_error", "levels must be distinct")
        if not self.rms:
            raise PydanticCustomError("value_error", "levels must not be all zero")
        return self

    @property
    def rms(self) -> float:
        return math.sqrt(sum(x * x for x in self.levels) / len(self.levels))


class MfskSoftDemap(Stage[CompileContext, MfskSoftDemapStep]):
    """M-ary soft demap, SYMBOLS->BITS soft. Turns the one-soft-float-per-symbol
    stream an fsk/msk discriminator emits into log2(M) LLRs per symbol, which is
    what the coding lane (deinterleave/depuncture/fec) consumes.

    `levels` is the discriminator output value for each symbol index, so the
    bit pattern of a level is its index (MSB-first) and any level->dibit map is
    a parameter. Built from stock blocks: the levels become a 1-D real
    constellation and the decision is constellation_soft_decoder, the same block
    psk_soft_demap uses, with the same sign correction.

    Amplitude: no agc precondition. The discriminator's scale is set by its own
    gain (rate/2*pi*deviation), not by the capture level, so the caller's
    `levels` ARE the scale reference; the stage normalizes the stream to the
    constellation's unit-RMS by construction."""

    name = "mfsk_soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "fsk"
    step_model = MfskSoftDemapStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: MfskSoftDemapStep) -> None:
        b.chain("multiply_const_ff", value=1.0 / step.rms)
        b.chain("float_to_complex")
        b.chain(
            "constellation_soft_decoder",
            scheme="explicit",
            points_i=list(step.levels),
            points_q=[0.0] * len(step.levels),
        )
        b.chain_llr_flip()

    def out_descriptor(
        self, in_desc: Descriptor, step: MfskSoftDemapStep
    ) -> Descriptor:
        k = len(step.levels).bit_length() - 1
        frame = None if in_desc.frame_len is None else in_desc.frame_len * k
        return Descriptor(Level.BITS, ItemType.F, Carrier.SOFT, frame_len=frame)

    def required_input_order(self, step: MfskSoftDemapStep) -> int | None:
        return len(step.levels)

    def output_item_rate(
        self, step: MfskSoftDemapStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return in_rate * math.log2(len(step.levels))


FSK_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (Fsk, Msk, MfskSoftDemap)
