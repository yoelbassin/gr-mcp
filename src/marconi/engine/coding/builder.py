from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from marconi.engine.coding.program import CodingStep


@dataclass
class CodingBuilder:
    steps: list[CodingStep] = field(default_factory=list)
    label: str = ""
    kind: str = ""

    def add(self, fn: Callable[..., Any], **params: Any) -> None:
        name = self.label or getattr(fn, "__name__", "step")
        kind = self.kind or name
        self.steps.append(CodingStep(name=name, kind=kind, call=partial(fn, **params)))
