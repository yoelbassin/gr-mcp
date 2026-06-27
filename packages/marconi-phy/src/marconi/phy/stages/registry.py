from __future__ import annotations

from marconi.core.stages import Stage
from marconi.phy.compile_context import CompileContext
from marconi.phy.modulation.fsk.stages import FSK_STAGES
from marconi.phy.modulation.ook.stages import OOK_STAGES
from marconi.phy.modulation.psk.stages import PSK_STAGES
from marconi.phy.modulation.qam.stages import QAM_STAGES
from marconi.phy.stages.conditioning import CONDITIONING_STAGES
from marconi.phy.stages.general import GENERAL_STAGES

# Each entry is a tuple of stage CLASSES; the registry instantiates fresh per call
# (no module-level mutable shared state). Family verticals append here.
_GROUPS: tuple[tuple[type[Stage[CompileContext]], ...], ...] = (
    GENERAL_STAGES,
    CONDITIONING_STAGES,
    FSK_STAGES,
    PSK_STAGES,
    OOK_STAGES,
    QAM_STAGES,
)


def stage_registry() -> dict[str, Stage[CompileContext]]:
    return {cls().name: cls() for group in _GROUPS for cls in group}
