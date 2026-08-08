from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, Stage
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class OokEnvelopeStep(Step):
    conv: Literal["ook_envelope"] = "ook_envelope"
    loop_bw: float = Field(
        default=0.045,
        ge=0,
        description=(
            "Symbol-timing loop bandwidth. 0 = open-loop per-burst sampling: "
            "fixed nominal-rate grid with the decimation phase re-acquired "
            "per burst (variance-max), deterministic, recommended for bursty "
            "or pulsed OOK/PPM. >0 = continuous Gardner loop for sustained "
            "OOK; measured to rail on burst environments (chip rate 2.2-2.8x "
            "off nominal), decoding nothing."
        ),
    )


class OokEnvelope(DuplexStage[CompileContext, OokEnvelopeStep]):
    """Non-coherent OOK envelope, IQ<->SYMBOLS. RX: magnitude -> centre to +/-1
    -> Gardner symbol timing (one soft float per symbol; CFO-immune by
    construction). TX: invert the centring, upsample, embed as the real part.
    Pairs with the general Slice for the SYMBOLS<->BITS decision."""

    name = "ook_envelope"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "ook"
    # MEASURED across a 1e-3..1e3 gain sweep: peak and RMS are BER 0; the
    # mean-mag statistic is not gain-invariant for an on/off envelope
    # (0.51/0.00/1.00), whose duty cycle sets the mean.
    accepts_amplitude = frozenset({Amplitude.PEAK_UNITY, Amplitude.RMS_UNITY})
    min_input_sps = 2.0
    step_model = OokEnvelopeStep

    def emit_rx(self, b: CompileContext, step: OokEnvelopeStep) -> None:
        b.chain("complex_to_mag")
        b.chain("multiply_const_ff", value=2.0)
        b.chain("add_const_ff", value=-1.0)
        b.chain("symbol_sync_ff", sps=b.sps, loop_bw=step.loop_bw)

    def emit_tx(self, b: CompileContext, step: OokEnvelopeStep) -> None:
        b.chain("add_const_ff", value=1.0)
        b.chain("multiply_const_ff", value=0.5)
        b.chain("repeat_f", interp=b.sps_int())
        b.chain("float_to_complex")

    def out_descriptor(self, in_desc: Descriptor, step: OokEnvelopeStep) -> Descriptor:
        return Descriptor(Level.SYMBOLS, ItemType.F, Carrier.SOFT)


OOK_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (OokEnvelope,)
