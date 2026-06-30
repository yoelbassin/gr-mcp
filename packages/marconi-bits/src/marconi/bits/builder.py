from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any


@dataclass
class ProgramBuilder:
    steps: list[Callable[..., Any]] = field(default_factory=list)

    def add(self, fn: Callable[..., Any], **params: Any) -> None:
        self.steps.append(partial(fn, **params))
