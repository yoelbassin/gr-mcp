from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, RxStage, Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class FskStep(Step):
    conv: Literal["fsk"] = "fsk"
    deviation: float
    loop_bw: float = Field(default=0.045, ge=0)


class Fsk(DuplexStage[CompileContext, FskStep]):
    """Binary FSK carrier stage. RX: FM discriminator + Gardner symbol timing ->
    one soft float per symbol. TX: upsample + FM modulate. The symbol-rate
    decimation is internal (driven by ctx.sps); rate_factor stays 1.0 so the
    modulator reads the IQ-domain sample rate (rate-model invariant)."""

    name = "fsk"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "fsk"
    min_input_sps = 2.0
    step_model = FskStep

    def emit_rx(self, b: CompileContext, step: FskStep) -> None:
        gain = b.rate / (2.0 * math.pi * step.deviation)
        b.chain("quadrature_demod", gain=gain)
        b.chain("symbol_sync_ff", sps=b.sps, loop_bw=step.loop_bw)

    def emit_tx(self, b: CompileContext, step: FskStep) -> None:
        b.chain("repeat_f", interp=b.sps_int())
        b.chain(
            "frequency_modulator",
            sensitivity=2.0 * math.pi * step.deviation / b.rate,
        )

    def out_descriptor(self, in_desc: Descriptor, step: FskStep) -> Descriptor:
        return Descriptor(Level.SYMBOLS, ItemType.F, Carrier.SOFT)


class MskStep(Step):
    conv: Literal["msk"] = "msk"
    loop_bw: float = 0.0038  # pinned: GR_BLOCKS["msk_demod"]'s registered default


class Msk(RxStage[CompileContext, MskStep]):
    """Coherent MSK (h=0.5 CPFSK) demod -> one soft float per symbol. RX-only:
    an MSK transmitter is the fsk stage at deviation = symbol_rate/4 (h=0.5),
    so TX needs no new vocabulary. Several dB more sensitive than the
    non-coherent fsk RX path (issue 22); guarded by
    tests/phy/test_msk_snr_margin.py."""

    name = "msk"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "fsk"
    min_input_sps = 2.0
    step_model = MskStep

    def emit_rx(self, b: CompileContext, step: MskStep) -> None:
        b.chain("msk_demod", sps=b.sps, loop_bw=step.loop_bw)

    def out_descriptor(self, in_desc: Descriptor, step: MskStep) -> Descriptor:
        return Descriptor(Level.SYMBOLS, ItemType.F, Carrier.SOFT)


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


class MfskSoftDemap(RxStage[CompileContext, MfskSoftDemapStep]):
    """M-ary soft demap, SYMBOLS->BITS soft. Turns the one-soft-float-per-symbol
    stream an fsk/msk discriminator emits into log2(M) LLRs per symbol, which is
    what the coding lane (deinterleave/depuncture/fec) consumes. Before this the
    only SYMBOLS->BITS stage accepting "f" was the binary `slice`, so every
    M-ary FSK signal had to leave phy as hard decisions.

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
        b.chain("multiply_const_ff", value=-1.0)

    def out_descriptor(
        self, in_desc: Descriptor, step: MfskSoftDemapStep
    ) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.F, Carrier.SOFT)

    def required_input_order(self, step: MfskSoftDemapStep) -> int | None:
        return len(step.levels)


FSK_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (Fsk, Msk, MfskSoftDemap)
