from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Window:
    start: int
    cursor: int


@dataclass(frozen=True)
class CodingCarrier:
    bits: np.ndarray
    windows: list[Window] = field(default_factory=list)
    symbols: np.ndarray | None = None
    marks: tuple[int, ...] = ()
