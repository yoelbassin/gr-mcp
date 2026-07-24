from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from marconi.bits.program import ProgramStep


@dataclass
class ProgramBuilder:
    steps: list[ProgramStep] = field(default_factory=list)
    # set by the compiler to the codec step being emitted, so a stage that adds
    # several ops labels them all without every emit site passing a name
    label: str = ""

    def add(self, fn: Callable[..., Any], **params: Any) -> None:
        name = self.label or getattr(fn, "__name__", "step")
        self.steps.append(ProgramStep(name=name, call=partial(fn, **params)))
