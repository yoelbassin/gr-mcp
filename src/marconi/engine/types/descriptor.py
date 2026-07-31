from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level


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
    item_type: ItemType  # GR wire type c/f/b/s; selects IO blocks
    carrier: Carrier = Carrier.HARD  # decision-hardness, a seam invariant
    amplitude: Amplitude = Amplitude.UNKNOWN
    order: int | None = None
    # Items per frame when the stream is back-to-back framed (a framing stage
    # pinned it), else None. Never propagated by default — a frame dies on any
    # item-count change, so only frame-aware stages carry it: producers pin,
    # 1:1 stages pass it through explicitly, expanders rescale, consumers with
    # fixed frame geometry check it and fail the compile on mismatch.
    frame_len: int | None = None

    def __post_init__(self) -> None:
        if self.frame_len is not None and self.frame_len < 1:
            raise ValueError(f"frame_len must be >= 1, got {self.frame_len}")
        if self.order is None:
            return
        if self.level is not Level.SYMBOLS:
            raise ValueError(
                f"order={self.order} at level {self.level.value}; a symbol "
                "alphabet only exists at symbols"
            )
        if self.order < 2:
            raise ValueError(f"order must be >= 2, got {self.order}")
