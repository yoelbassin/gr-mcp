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
            "off nominal), decoding nothing. Open-loop needs no agc stage: "
            "the sampler normalizes each burst internally, and a sliding-"
            "window agc would step its gain mid-burst and defeat fixed-"
            "threshold slicing. Open-loop also runs at sps>=1 (closed-loop "
            "still needs sps>=2), so a capture already at the symbol rate "
            "decodes with no resample stage - avoiding the anti-imaging "
            "low-pass that smears short pulses."
        ),
    )


class OokEnvelope(DuplexStage[CompileContext, OokEnvelopeStep]):
    name = "ook_envelope"
    description = (
        "Non-coherent OOK/PPM envelope demod, IQ<->SYMBOLS, whose amplitude "
        "contract is conditional on loop_bw. Closed-loop (loop_bw>0) needs "
        "peak_unity or rms_unity input - pair it with an agc. Open-loop "
        "(loop_bw=0) is amplitude-agnostic and must run WITHOUT an agc stage: "
        "the per-burst sampler normalizes each burst itself, and a "
        "sliding-window agc steps its gain mid-burst on pulsed signals and "
        "corrupts fixed-threshold slicing. accepts_amplitude and "
        "min_input_sps are conditional on loop_bw (compile-enforced): "
        "closed-loop needs normalized amplitude and sps>=2; open-loop is "
        "amplitude-agnostic down to sps=1."
    )
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "ook"
    # MEASURED across a 1e-3..1e3 gain sweep: peak and RMS are BER 0; the
    # mean-mag statistic is not gain-invariant for an on/off envelope
    # (0.51/0.00/1.00), whose duty cycle sets the mean.
    accepts_amplitude = frozenset({Amplitude.PEAK_UNITY, Amplitude.RMS_UNITY})
    min_input_sps = 2.0
    step_model = OokEnvelopeStep

    def accepts_amplitude_for(
        self, step: OokEnvelopeStep
    ) -> frozenset[Amplitude] | None:
        return None if step.loop_bw == 0.0 else self.accepts_amplitude

    def min_input_sps_for(self, step: OokEnvelopeStep) -> float | None:
        return 1.0 if step.loop_bw == 0.0 else self.min_input_sps

    def emit_rx(self, b: CompileContext, step: OokEnvelopeStep) -> None:
        b.chain("complex_to_mag")
        if step.loop_bw == 0.0:
            b.chain("burst_sampler", sps=b.sps)
            b.chain("multiply_const_ff", value=2.0)
            b.chain("add_const_ff", value=-1.0)
            return
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

    def output_item_rate(
        self, step: OokEnvelopeStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return symbol_rate


OOK_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (OokEnvelope,)
