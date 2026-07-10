from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import RxStage
from marconi.phy.compile_context import CompileContext


class _BurstProbeParams(StageParams):
    pass


class BurstProbe(RxStage[CompileContext]):
    """Passthrough probe recording each upstream "burst" tag's symbol offset
    into run diagnostics ({"bursts": [...]}) — burst starts for an offline
    consumer, since stream tags don't survive a file sink. SYMBOLS->SYMBOLS,
    rate unchanged."""

    name = "burst_probe"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "probe"
    params_model = _BurstProbeParams
    accepts_item_type = "s"

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("burst_probe")


PROBE_STAGES: tuple[type[RxStage[CompileContext]], ...] = (BurstProbe,)
