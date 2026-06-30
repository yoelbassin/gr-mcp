from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marconi.bits import framing
from marconi.bits.builder import ProgramBuilder
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage


class Differential(DuplexStage[ProgramBuilder]):
    name = "differential"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "framing"

    class _Params(StageParams):
        invert: bool = False

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.differential_rx, invert=bool(params.get("invert", False)))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.differential_tx, invert=bool(params.get("invert", False)))


class HdlcDeframe(DuplexStage[ProgramBuilder]):
    name = "hdlc_deframe"
    from_level = Level.BITS
    to_level = Level.FRAMES
    family = "framing"

    class _Params(StageParams):
        bit_order: str = "msb"

    params_model = _Params

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.hdlc_deframe_rx, bit_order=str(params.get("bit_order", "msb")))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.hdlc_deframe_tx, bit_order=str(params.get("bit_order", "msb")))


FRAMING_STAGES: tuple[type[DuplexStage[ProgramBuilder]], ...] = (
    Differential,
    HdlcDeframe,
)
