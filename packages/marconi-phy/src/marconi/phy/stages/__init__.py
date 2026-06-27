from __future__ import annotations

from marconi.core.stages import Stage
from marconi.phy.compile_context import CompileContext
from marconi.phy.stages.fsk import Fsk, Slice


def stage_registry() -> dict[str, Stage[CompileContext]]:
    stages: list[Stage[CompileContext]] = [Fsk(), Slice()]
    return {s.name: s for s in stages}
