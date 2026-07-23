from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from marconi.core.levels import Level


class Layout(str, Enum):
    STREAM = "stream"
    # GRID deferred (OFDM) — see the spec's expressibility note.


class Carrier(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class Amplitude(str, Enum):
    UNKNOWN = "unknown"
    NORMALIZED = "normalized"


@dataclass(frozen=True)
class Descriptor:
    level: Level
    item_type: str  # GR wire type "c/f/b/s"; selects IO blocks
    layout: Layout = Layout.STREAM
    carrier: Carrier = Carrier.HARD  # decision-hardness, a seam invariant
    amplitude: Amplitude = Amplitude.UNKNOWN
