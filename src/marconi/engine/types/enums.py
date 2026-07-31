from __future__ import annotations

from enum import IntEnum, StrEnum


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
