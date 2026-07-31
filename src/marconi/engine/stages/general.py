from __future__ import annotations

from typing import Any, Literal

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class SliceStep(Step):
    conv: Literal["slice"] = "slice"


class Slice(DuplexStage[CompileContext, SliceStep]):
    """Binary symbol<->bit decision. RX: hard-slice soft floats to bits.
    TX: map bits to antipodal +/-1.0 symbols."""

    name = "slice"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "general"
    step_model = SliceStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: SliceStep) -> None:
        b.chain("binary_slicer")

    def emit_tx(self, b: CompileContext, step: SliceStep) -> None:
        b.chain("chunks_to_symbols", symbols=[-1.0, 1.0])

    def out_descriptor(self, in_desc: Descriptor, step: SliceStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.B, Carrier.HARD)


GENERAL_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (Slice,)
