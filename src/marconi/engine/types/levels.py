from __future__ import annotations

from enum import Enum


class Level(str, Enum):
    IQ = "iq"
    SYMBOLS = "symbols"
    BITS = "bits"
    AUDIO = "audio"
