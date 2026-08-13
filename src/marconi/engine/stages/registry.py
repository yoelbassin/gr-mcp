from __future__ import annotations

from collections import Counter
from typing import Any

from marconi.engine.coding.stages_bits import CODING_BITS_STAGES
from marconi.engine.coding.stages_symbols import CODING_SYMBOL_STAGES
from marconi.engine.modulation.coding.stages import CODING_STAGES
from marconi.engine.modulation.css.stages import CSS_STAGES
from marconi.engine.modulation.fsk.stages import FSK_STAGES
from marconi.engine.modulation.ofdm.stages import OFDM_STAGES
from marconi.engine.modulation.ook.stages import OOK_STAGES
from marconi.engine.modulation.psk.stages import PSK_STAGES
from marconi.engine.modulation.qam.stages import QAM_STAGES
from marconi.engine.stages.acquisition import ACQUISITION_STAGES
from marconi.engine.stages.base import Stage
from marconi.engine.stages.conditioning import CONDITIONING_STAGES
from marconi.engine.stages.general import GENERAL_STAGES
from marconi.engine.stages.probes import PROBE_STAGES
from marconi.engine.types.step import Step

# Each entry is a tuple of stage CLASSES; the registry instantiates fresh per call
# (no module-level mutable shared state). Family verticals append here.
_GROUPS: tuple[tuple[type[Stage[Any, Any]], ...], ...] = (
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


def stage_registry() -> dict[str, Stage[Any, Any]]:
    """Every stage by name. A dict comprehension alone let two classes claiming
    one name collapse to whichever was listed last, silently taking the other
    out of describe_stages, step_models and every path resolution — the largest
    vocabulary in the tree, with no uniqueness check on it."""
    stages = [cls() for group in _GROUPS for cls in group]
    registry = {s.name: s for s in stages}
    if len(registry) != len(stages):
        clashes = sorted(
            name for name, n in Counter(s.name for s in stages).items() if n > 1
        )
        raise RuntimeError(
            f"stage names must be unique; {clashes} is claimed by more than one "
            "class, and only the last would survive"
        )
    return registry


def step_models() -> dict[str, type[Step]]:
    return {conv: st.step_model for conv, st in stage_registry().items()}
