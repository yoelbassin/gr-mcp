from __future__ import annotations

from typing import Any

from marconi.core.stages import Stage
from marconi.phy.coding.stages_bits import CODING_BITS_STAGES
from marconi.phy.coding.stages_symbols import CODING_SYMBOL_STAGES
from marconi.phy.modulation.coding.stages import CODING_STAGES
from marconi.phy.modulation.css.stages import CSS_STAGES
from marconi.phy.modulation.fsk.stages import FSK_STAGES
from marconi.phy.modulation.ofdm.stages import OFDM_STAGES
from marconi.phy.modulation.ook.stages import OOK_STAGES
from marconi.phy.modulation.psk.stages import PSK_STAGES
from marconi.phy.modulation.qam.stages import QAM_STAGES
from marconi.phy.stages.acquisition import ACQUISITION_STAGES
from marconi.phy.stages.conditioning import CONDITIONING_STAGES
from marconi.phy.stages.general import GENERAL_STAGES
from marconi.phy.stages.probes import PROBE_STAGES

# Each entry is a tuple of stage CLASSES; the registry instantiates fresh per call
# (no module-level mutable shared state). Family verticals append here.
_GROUPS: tuple[tuple[type[Stage[Any]], ...], ...] = (
    GENERAL_STAGES,
    CONDITIONING_STAGES,
    ACQUISITION_STAGES,
    FSK_STAGES,
    PSK_STAGES,
    OOK_STAGES,
    QAM_STAGES,
    CSS_STAGES,
    OFDM_STAGES,
    CODING_STAGES,
    CODING_BITS_STAGES,
    CODING_SYMBOL_STAGES,
    PROBE_STAGES,
)


def stage_registry() -> dict[str, Stage[Any]]:
    stages = (cls() for group in _GROUPS for cls in group)
    return {s.name: s for s in stages}
