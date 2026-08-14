from __future__ import annotations

from typing import Any, Literal

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class SliceStep(Step):
    conv: Literal["slice"] = "slice"


class Slice(Stage[CompileContext, SliceStep]):
    """Binary symbol -> bit decision: hard-slice soft floats to bits."""

    name = "slice"
    description = (
        "Hard binary slicer at zero: soft symbols to bits. Fast and universal, "
        "but hard bits carry no per-bit confidence, so a slice path earns no soft "
        "quality evidence - prefer a soft demap when the verdict matters."
    )
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "general"
    step_model = SliceStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: SliceStep) -> None:
        b.chain("binary_slicer")

    def out_descriptor(self, in_desc: Descriptor, step: SliceStep) -> Descriptor:
        return Descriptor(
            Level.BITS, ItemType.B, Carrier.HARD, frame_len=in_desc.frame_len
        )


GENERAL_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (Slice,)
