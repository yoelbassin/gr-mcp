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

    def begin_step(self, label: str, kind: str) -> None:
        self.label = label
        self.kind = kind

    def add(self, fn: Callable[..., Any], **params: Any) -> None:
        if not self.label or not self.kind:
            raise RuntimeError("CodingBuilder.add before begin_step")
        self.steps.append(
            CodingStep(name=self.label, kind=self.kind, call=partial(fn, **params))
        )
