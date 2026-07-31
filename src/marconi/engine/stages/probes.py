from __future__ import annotations

from typing import Any, Literal

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import RxStage, Stage
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class BurstProbeStep(Step):
    conv: Literal["burst_probe"] = "burst_probe"


class BurstProbe(RxStage[CompileContext, BurstProbeStep]):
    """Passthrough probe recording each upstream "burst" tag's symbol offset
    into run diagnostics ({"bursts": [...]}) — burst starts for an offline
    consumer, since stream tags don't survive a file sink."""

    name = "burst_probe"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "probe"
    step_model = BurstProbeStep
    accepts_item_type = ItemType.S

    def emit_rx(self, b: CompileContext, step: BurstProbeStep) -> None:
        b.chain("burst_probe")


PROBE_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (BurstProbe,)
