from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage
from marconi.phy.compile_context import CompileContext


class _SliceParams(StageParams):
    pass


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


GENERAL_STAGES: tuple[type[DuplexStage[CompileContext]], ...] = (Slice,)
