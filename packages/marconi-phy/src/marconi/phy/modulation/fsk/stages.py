from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage
from marconi.phy.compile_context import CompileContext


class _FskParams(StageParams):
    deviation: float
    loop_bw: float = 0.045


class Fsk(DuplexStage[CompileContext]):
    """Binary FSK carrier stage. RX: FM discriminator + Gardner symbol timing ->
    one soft float per symbol. TX: upsample + FM modulate. The symbol-rate
    decimation is internal (driven by ctx.sps); rate_factor stays 1.0 so the
    modulator reads the IQ-domain sample rate (rate-model invariant)."""

    name = "fsk"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "fsk"
    params_model = _FskParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _FskParams.model_validate(dict(params))
        gain = b.rate / (2.0 * math.pi * p.deviation)
        b.chain("quadrature_demod", gain=gain)
        b.chain("symbol_sync_ff", sps=b.sps, loop_bw=p.loop_bw)

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _FskParams.model_validate(dict(params))
        b.chain("repeat_f", interp=b.sps_int())
        b.chain(
            "frequency_modulator",
            sensitivity=2.0 * math.pi * p.deviation / b.rate,
        )

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "f", in_desc.layout, Carrier.SOFT)


FSK_STAGES: tuple[type[DuplexStage[CompileContext]], ...] = (Fsk,)
