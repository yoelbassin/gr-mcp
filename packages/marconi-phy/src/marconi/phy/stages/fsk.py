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


class _SliceParams(StageParams):
    pass


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
        deviation = float(params["deviation"])
        gain = b.rate / (2.0 * math.pi * deviation)
        b.chain("quadrature_demod", gain=gain)
        b.chain(
            "symbol_sync_ff", sps=b.sps, loop_bw=float(params.get("loop_bw", 0.045))
        )

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        deviation = float(params["deviation"])
        b.chain("repeat_f", interp=int(round(b.sps)))
        b.chain(
            "frequency_modulator",
            sensitivity=2.0 * math.pi * deviation / b.rate,
        )

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "f", in_desc.layout, Carrier.SOFT)


class Slice(DuplexStage[CompileContext]):
    """Binary symbol<->bit decision. RX: hard-slice soft floats to bits.
    TX: map bits to antipodal +/-1.0 symbols."""

    name = "slice"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "general"
    params_model = _SliceParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("binary_slicer")

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("chunks_to_symbols", symbols=[-1.0, 1.0])

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)
