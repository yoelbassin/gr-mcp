from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Any, Literal

import numpy as np


class PskOrder(IntEnum):
    BPSK = 2
    QPSK = 4
    PSK8 = 8


class QamOrder(IntEnum):
    QAM16 = 16
    QAM64 = 64


class AgcMode(StrEnum):
    FEEDFORWARD = "feedforward"
    FEEDBACK = "feedback"
    POWER = "power"


class DecodeMode(StrEnum):
    EXACT = "exact"
    NEAREST = "nearest"


class EmitMode(StrEnum):
    DATA = "data"
    CODEWORD = "codeword"


class DcMode(StrEnum):
    MEDIAN = "median"
    MEAN = "mean"
    NONE = "none"


class ItemType(StrEnum):
    C = "c"
    F = "f"
    B = "b"
    S = "s"

    @property
    def suffix(self) -> str:
        return _ITEM_SUFFIXES[self]

    @property
    def np_dtype(self) -> np.dtype[Any]:
        return _ITEM_DTYPES[self]

    @property
    def item_bytes(self) -> int:
        return self.np_dtype.itemsize

    def require_symbol(self) -> Literal["s", "f"]:
        if self is ItemType.S:
            return "s"
        if self is ItemType.F:
            return "f"
        raise ValueError(f"expected a symbol item type, got {self.value!r}")


_ITEM_SUFFIXES: dict[ItemType, str] = {
    ItemType.C: ".cf32",
    ItemType.F: ".f32",
    ItemType.S: ".i16",
    ItemType.B: ".u8",
}

_ITEM_DTYPES: dict[ItemType, np.dtype[Any]] = {
    ItemType.C: np.dtype(np.complex64),
    ItemType.F: np.dtype(np.float32),
    ItemType.S: np.dtype(np.int16),
    ItemType.B: np.dtype(np.uint8),
}
