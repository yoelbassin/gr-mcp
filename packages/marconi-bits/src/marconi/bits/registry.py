from __future__ import annotations

from marconi.bits.builder import ProgramBuilder
from marconi.bits.stages.framing_ops import FRAMING_STAGES
from marconi.bits.stages.integrity import INTEGRITY_STAGES
from marconi.bits.stages.message import MESSAGE_STAGES
from marconi.core.stages import Stage

_GROUPS = (FRAMING_STAGES, INTEGRITY_STAGES, MESSAGE_STAGES)


def registry() -> dict[str, Stage[ProgramBuilder]]:
    return {cls().name: cls() for group in _GROUPS for cls in group}
