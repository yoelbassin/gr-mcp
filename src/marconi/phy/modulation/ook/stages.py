from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marconi.core.descriptor import Amplitude, Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage
from marconi.phy.compile_context import CompileContext


class _OokParams(StageParams):
    loop_bw: float = 0.045


class OokEnvelope(DuplexStage[CompileContext]):
    """Non-coherent OOK envelope, IQ<->SYMBOLS. RX: magnitude -> centre to +/-1
    -> Gardner symbol timing (one soft float per symbol; CFO-immune by
    construction). TX: invert the centring, upsample, embed as the real part.
    Pairs with the general Slice for the SYMBOLS<->BITS decision."""

    name = "ook_envelope"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "ook"
    accepts_amplitude = Amplitude.NORMALIZED
    params_model = _OokParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _OokParams.model_validate(dict(params))
        b.chain("complex_to_mag")
        b.chain("multiply_const_ff", value=2.0)
        b.chain("add_const_ff", value=-1.0)
        b.chain("symbol_sync_ff", sps=b.sps, loop_bw=p.loop_bw)

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("add_const_ff", value=1.0)
        b.chain("multiply_const_ff", value=0.5)
        b.chain("repeat_f", interp=b.sps_int())
        b.chain("float_to_complex")

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "f", in_desc.layout, Carrier.SOFT)


OOK_STAGES: tuple[type[DuplexStage[CompileContext]], ...] = (OokEnvelope,)
