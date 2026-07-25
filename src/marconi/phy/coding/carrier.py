from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    start: int
    cursor: int


@dataclass(frozen=True)
class CodingCarrier:
    """windows is a scope, not a container: None means no seeder ran (blind,
    whole-stream ops are correct); [] means a seeder ran and found nothing
    (decoding anything from it would fabricate data)."""

    bits: np.ndarray
    windows: list[Window] | None = None
    symbols: np.ndarray | None = None
    marks: tuple[int, ...] = ()
