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


class CaptureDtype(StrEnum):
    """How an on-disk IQ capture is laid out. CF32 is the engine's native wire
    and what a recording writes; the rest are interleaved integer formats a
    caller's SDR tooling produced, converted on the way in."""

    CF32 = "cf32"
    CI16 = "ci16"
    CI8 = "ci8"
    CU8 = "cu8"


class RunStatus(StrEnum):
    """How a run ended, for the backend, the engine and a hardware capture
    alike. One vocabulary rather than a Literal re-typed per model: the three
    used to be hand-written four/four/three-member Literals bridged by an
    unguarded lookup table, so a new member reached the wire from one producer
    and KeyError'd in another."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    # the graph ran clean but wrote nothing — a decoded-nothing run is never OK
    EMPTY = "empty"


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

    def require_symbol(self) -> "Literal[ItemType.S, ItemType.F]":
        """Narrowed to the two item types that carry one value per symbol, so a
        reader keyed on the distinction (io.bitfile.read_symbols' int16-vs-
        float32 return) stays precise instead of widening to the whole enum."""
        if self is ItemType.S:
            return ItemType.S
        if self is ItemType.F:
            return ItemType.F
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
