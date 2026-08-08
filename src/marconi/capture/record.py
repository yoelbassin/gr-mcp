from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from marconi.errors import register_error

_SETTLE_S = 0.2
_MAX_DURATION_S = 300.0
_TIMEOUT_MARGIN_S = 30.0
_LEVELS_MAX_SAMPLES = 1 << 22
_CLIP_THRESHOLD = 0.99


class CaptureError(Exception):
    pass


register_error(CaptureError, "capture")


class CaptureLevels(BaseModel):
    rms: float
    dc_offset: float
    clip_fraction: float


class CaptureResult(BaseModel):
    status: Literal["ok", "error", "timeout"]
    path: str
    dtype: Literal["cf32"] = "cf32"
    sample_rate: float
    center_hz: float
    num_samples: int
    duration_s: float
    levels: CaptureLevels
    warnings: list[str] = []
    error: str | None = None


def _reconcile(
    requested_rate: float,
    requested_freq: float,
    probe: tuple[float, float] | None,
) -> tuple[float, float, list[str]]:
    if probe is None:
        return (
            requested_rate,
            requested_freq,
            [
                "SoapySDR python bindings unavailable — reported "
                "sample_rate/center_hz are the requested values, unverified"
            ],
        )
    actual_rate, actual_freq = probe
    if abs(actual_freq - requested_freq) > actual_rate / 2:
        raise CaptureError(
            f"device tuned {actual_freq:g} Hz, more than half the sample rate "
            f"away from requested {requested_freq:g} Hz — the requested span "
            "is not in this capture"
        )
    warnings: list[str] = []
    if actual_rate != requested_rate:
        warnings.append(
            f"sample_rate: requested {requested_rate:g}, "
            f"device delivered {actual_rate:g}"
        )
    if actual_freq != requested_freq:
        warnings.append(
            f"center_hz: requested {requested_freq:g}, " f"device tuned {actual_freq:g}"
        )
    return actual_rate, actual_freq, warnings


def _levels(path: Path) -> CaptureLevels:
    data = np.memmap(path, dtype=np.complex64, mode="r")
    stride = max(1, data.size // _LEVELS_MAX_SAMPLES)
    z = np.asarray(data[::stride])
    rms = float(np.sqrt(np.mean(np.abs(z) ** 2)))
    dc = float(abs(np.mean(z)))
    clipped = (np.abs(z.real) >= _CLIP_THRESHOLD) | (np.abs(z.imag) >= _CLIP_THRESHOLD)
    return CaptureLevels(
        rms=round(rms, 4),
        dc_offset=round(dc, 4),
        clip_fraction=round(float(np.mean(clipped)), 4),
    )
