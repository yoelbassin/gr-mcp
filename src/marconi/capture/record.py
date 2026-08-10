from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend
from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline
from marconi.engine.types.params import ParamValue
from marconi.errors import register_error

_SETTLE_S = 0.2
_MAX_DURATION_S = 300.0
_TIMEOUT_MARGIN_S = 30.0
_LEVELS_MAX_SAMPLES = 1 << 22
_CLIP_THRESHOLD = 0.99
_CLIP_WARN_FRACTION = 0.01
_RMS_DEAD = 1e-3


class CaptureError(Exception):
    pass


register_error(CaptureError, "failed_precondition")


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
    data = np.memmap(
        path, dtype=np.complex64, mode="r", shape=(path.stat().st_size // 8,)
    )
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


def _level_warnings(levels: CaptureLevels) -> list[str]:
    warnings: list[str] = []
    if levels.clip_fraction > _CLIP_WARN_FRACTION:
        warnings.append(
            f"clip_fraction {levels.clip_fraction:g}: capture is railing at full "
            "scale and distorting — lower gain_db (or set it, if on hardware AGC)"
        )
    if levels.rms < _RMS_DEAD:
        warnings.append(
            f"rms {levels.rms:g} near zero: no signal reaching the ADC — raise "
            "gain_db or check the antenna/frequency"
        )
    return warnings


def build_capture_pipeline(
    *,
    out_path: Path,
    device: str,
    sample_rate: float,
    center_hz: float,
    gain_db: float | None,
    ppm: float,
    settle_samples: int,
    num_samples: int,
) -> GrPipeline:
    source_params: dict[str, ParamValue] = {
        "device": device,
        "sample_rate": sample_rate,
        "center_hz": center_hz,
        "ppm": ppm,
        "agc": gain_db is None,
    }
    if gain_db is not None:
        source_params["gain_db"] = gain_db
    return GrPipeline(
        name="capture",
        sample_rate=sample_rate,
        blocks=[
            GrBlock(id="src", kind="soapy_source", params=source_params),
            GrBlock(
                id="settle", kind="iq_skiphead", params={"num_items": settle_samples}
            ),
            GrBlock(id="bound", kind="iq_head", params={"num_items": num_samples}),
            GrBlock(id="sink", kind="iq_file_sink", params={"path": str(out_path)}),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="settle"),
            GrConnection(src_block="settle", dst_block="bound"),
            GrConnection(src_block="bound", dst_block="sink"),
        ],
    )


def _probe_device(
    device: str, sample_rate: float, center_hz: float, ppm: float
) -> tuple[float, float] | None:
    try:
        import SoapySDR
    except ImportError:
        return None
    try:
        found = SoapySDR.Device.enumerate(device)
    except Exception as e:
        raise CaptureError(
            f"failed to enumerate SDR devices (device={device!r}): {e}"
        ) from e
    if not found:
        raise CaptureError(
            f"no SDR device found (device={device!r}) — is one plugged in, "
            "with its Soapy driver module installed?"
        )
    try:
        dev: Any = SoapySDR.Device(device)
    except Exception as e:
        raise CaptureError(f"failed to open SDR (device={device!r}): {e}") from e
    try:
        rx = SoapySDR.SOAPY_SDR_RX
        dev.setSampleRate(rx, 0, sample_rate)
        if ppm:
            dev.setFrequencyCorrection(rx, 0, ppm)
        dev.setFrequency(rx, 0, center_hz)
        return float(dev.getSampleRate(rx, 0)), float(dev.getFrequency(rx, 0))
    except CaptureError:
        raise
    except Exception as e:
        raise CaptureError(f"device refused tune/rate: {e}") from e
    finally:
        dev.close()


def capture_iq(
    out_path: Path,
    *,
    center_hz: float,
    sample_rate: float = 2.048e6,
    duration_s: float = 5.0,
    gain_db: float | None = None,
    ppm: float = 0.0,
    device: str = "",
) -> CaptureResult:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if not 0 < duration_s <= _MAX_DURATION_S:
        raise ValueError(f"duration_s must be in (0, {_MAX_DURATION_S:g}]")
    probe = _probe_device(device, sample_rate, center_hz, ppm)
    rate, freq, warnings = _reconcile(sample_rate, center_hz, probe)
    pipeline = build_capture_pipeline(
        out_path=out_path,
        device=device,
        sample_rate=sample_rate,
        center_hz=center_hz,
        gain_db=gain_db,
        ppm=ppm,
        settle_samples=round(_SETTLE_S * rate),
        num_samples=round(duration_s * rate),
    )
    result = GnuRadioBackend().run_pipeline(
        pipeline, timeout=_SETTLE_S + duration_s + _TIMEOUT_MARGIN_S
    )
    num_samples = out_path.stat().st_size // 8 if out_path.is_file() else 0
    if num_samples == 0:
        raise CaptureError(
            f"capture produced no samples: {result.error or result.status}"
        )
    status: Literal["ok", "error", "timeout"] = (
        "ok"
        if result.status == "ok"
        else ("timeout" if result.status == "timeout" else "error")
    )
    levels = _levels(out_path)
    return CaptureResult(
        status=status,
        path=str(out_path),
        sample_rate=rate,
        center_hz=freq,
        num_samples=num_samples,
        duration_s=round(num_samples / rate, 3),
        levels=levels,
        warnings=[*warnings, *_level_warnings(levels)],
        error=result.error,
    )
