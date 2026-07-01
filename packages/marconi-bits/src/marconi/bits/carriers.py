from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class _Frame:
    start: int
    cursor: int
    payload: bytes = b""
    crc_ok: bool | None = None  # None = no oracle checked it
    message: dict | None = None


@dataclass(frozen=True)
class RxCarrier:
    bits: np.ndarray
    frames: list[_Frame] = field(default_factory=list)
    llrs: np.ndarray | None = None  # soft lane: present, unused in slice 1


@dataclass(frozen=True)
class TxCarrier:
    items: list
