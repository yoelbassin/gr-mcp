from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from marconi.engine.types.levels import Level


class Layout(str, Enum):
    STREAM = "stream"
    # GRID deferred (OFDM) — see the spec's expressibility note.


class Carrier(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class Amplitude(str, Enum):
    """Which amplitude STATISTIC a stream has been normalized to. "Normalized"
    alone is not a contract: a demod with absolute decision boundaries needs a
    specific statistic, and the wrong one is silent garbage rather than a
    tolerable offset (measured: qam_demod is SER 0.91 at every gain on a
    peak-normalized stream, SER 0 on an RMS-normalized one)."""

    UNKNOWN = "unknown"
    PEAK_UNITY = "peak_unity"  # max |x| -> reference
    MEAN_MAG_UNITY = "mean_mag_unity"  # mean |x| -> reference
    RMS_UNITY = "rms_unity"  # sqrt(mean |x|^2) -> reference


@dataclass(frozen=True)
class Descriptor:
    level: Level
    item_type: str  # GR wire type "c/f/b/s"; selects IO blocks
    layout: Layout = Layout.STREAM
    carrier: Carrier = Carrier.HARD  # decision-hardness, a seam invariant
    amplitude: Amplitude = Amplitude.UNKNOWN
