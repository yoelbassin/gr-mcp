from __future__ import annotations

from enum import Enum


class Level(str, Enum):
    IQ = "iq"
    SYMBOLS = "symbols"
    BITS = "bits"
    FRAMES = "frames"
    MESSAGES = "messages"
    AUDIO = "audio"
