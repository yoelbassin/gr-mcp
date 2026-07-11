from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _Frame:
    start: int
    cursor: int
    payload: bytes = b""
    crc_ok: bool | None = None
    message: dict[str, int | str] | None = None


@dataclass(frozen=True)
class RxCarrier:
    bits: np.ndarray
    frames: list[_Frame] = field(default_factory=list)
    symbols: np.ndarray | None = None
    marks: tuple[int, ...] = ()


@dataclass(frozen=True)
class TxCarrier:
    items: list[Any]  # varies per TX stage: bytes | ndarray | message dict
