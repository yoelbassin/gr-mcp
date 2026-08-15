from __future__ import annotations

import math
import shutil
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from marconi.deadline import RunTimeout, check_deadline, remaining
from marconi.engine.backends.gnuradio.runner import (
    GnuRadioBackend,
    reap_process,
    receive_payload,
    worker_context,
)
from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline
from marconi.engine.types.bounds import check_sample_rate
from marconi.engine.types.enums import CaptureDtype, RunStatus
from marconi.engine.types.params import ParamValue
from marconi.errors import register_error
from marconi.wire import Payload

_SETTLE_S = 0.2
_MAX_DURATION_S = 300.0
_TIMEOUT_MARGIN_S = 30.0
_LEVELS_MAX_SAMPLES = 1 << 22
_CLIP_THRESHOLD = 0.99
_CLIP_WARN_FRACTION = 0.01
_RMS_DEAD = 1e-3
_BYTES_PER_SAMPLE = 8
_GIB = float(1 << 30)

# Enumerate, open and tune are blocking driver calls with no poll inside them,
# so this is a bound on a CHILD PROCESS rather than a cooperative check: a 6 s
# block measured 6.01 s even inside set_deadline(0.1), because check_deadline
# cannot reach a C call. The cap has to clear the slowest LEGITIMATE open, not
# the local negative — enumerate+open with nothing attached measured 6.4 ms
# cold and 0.1 ms warm here (macOS, SoapySDR 0.8), while a UHD device loads
# firmware on first open and a SoapyRemote one opens over the network, both
# seconds. Lower and a slow-but-working radio is refused mid-open; higher and a
# wedged one is that much dead wait. The call's own timeout composes over this:
# the probe gets whichever is smaller. The bound costs one forkserver child,
# measured 250-260 ms round trip (fork, import, enumerate, pipe) when warm.
_PROBE_TIMEOUT_S = 20.0

# How long a probe that answered nothing is given to report an exit code, which
# is the only thing separating a driver that crashed its process from one still
# inside the call. It is spent PAST a deadline that is already gone, so it is
# also the call's worst-case overshoot: a 2 s budget measured 3.02 s at 1.0 s
# here and 2.27 s at this value, while a child that called _exit is still
# collected every time.
_PROBE_REAP_S = 0.25

# sample_rate sizes the WRITE here, unlike at every other tool entry where a
# wrong rate is cheap (bounds.py leaves physical quantities unbounded for that
# reason): every sample is 8 bytes to disk. check_sample_rate alone (finite,
# > 0) accepted 1e12 — _reconcile returned it verbatim and the compiler turned
# it into a 3.000e+14-item head block, 2183 TiB, with a clean status. 1 Gsps is
# 8 GB/s of sustained write — past what a PCIe 4 NVMe drive sustains, ~6x a
# 10GbE link and ~13x USB3 gen1, which are the buses SDRs actually stream over,
# so no host can record past this whatever a radio claims. Realistic requests
# are bounded by the free-space preflight below; this catches the typo'd or
# fabricated number.
_MAX_SAMPLE_RATE = 1e9


class CaptureError(Exception):
    pass


register_error(CaptureError, "failed_precondition")


class CaptureLevels(Payload):
    rms: float
    dc_offset: float
    clip_fraction: float


class CaptureResult(Payload):
    """capture's wire shape. A Payload like every other tool response, so the
    omission rule applies here too: a clean capture used to ship "error": null
    because this one model of the eight bypassed the base and dumped itself."""

    status: RunStatus
    path: str
    dtype: CaptureDtype = CaptureDtype.CF32
    sample_rate: float
    center_hz: float
    num_samples: int
    duration_s: float
    levels: CaptureLevels
    overflow_events: int = 0
    warnings: list[str] = []
    error: str | None = None


class DeviceReadback(BaseModel):
    sample_rate: float
    center_hz: float


class Reconciled(BaseModel):
    sample_rate: float
    center_hz: float
    warnings: list[str] = []


def _reconcile(
    requested_rate: float,
    requested_freq: float,
    probe: DeviceReadback | None,
) -> Reconciled:
    if probe is None:
        return Reconciled(
            sample_rate=requested_rate,
            center_hz=requested_freq,
            warnings=[
                "SoapySDR python bindings unavailable — reported "
                "sample_rate/center_hz are the requested values, unverified"
            ],
        )
    actual_rate, actual_freq = probe.sample_rate, probe.center_hz
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
    return Reconciled(sample_rate=actual_rate, center_hz=actual_freq, warnings=warnings)


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


def _overflow_warnings(events: int) -> list[str]:
    if events <= 0:
        return []
    return [
        f"overflow_events {events}: the SDR driver dropped buffers and spliced "
        "the gaps out, so the file is NOT contiguous in time even though the "
        "sample count is complete — durations, burst spacing and any timing "
        "measured across a gap are wrong. Lower sample_rate, shorten "
        "duration_s, or free the machine, then re-capture"
    ]


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


class _SoapyDevice(Protocol):
    def setSampleRate(self, direction: int, channel: int, rate: float) -> None: ...
    def setFrequency(self, direction: int, channel: int, freq: float) -> None: ...

    def setFrequencyCorrection(
        self, direction: int, channel: int, ppm: float
    ) -> None: ...

    def getSampleRate(self, direction: int, channel: int) -> float: ...
    def getFrequency(self, direction: int, channel: int) -> float: ...
    def close(self) -> None: ...


def _probe_device(
    device: str, sample_rate: float, center_hz: float, ppm: float
) -> DeviceReadback | None:
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
        dev: _SoapyDevice = SoapySDR.Device(device)
    except Exception as e:
        raise CaptureError(f"failed to open SDR (device={device!r}): {e}") from e
    try:
        rx = SoapySDR.SOAPY_SDR_RX
        dev.setSampleRate(rx, 0, sample_rate)
        if ppm:
            dev.setFrequencyCorrection(rx, 0, ppm)
        dev.setFrequency(rx, 0, center_hz)
        return DeviceReadback(
            sample_rate=float(dev.getSampleRate(rx, 0)),
            center_hz=float(dev.getFrequency(rx, 0)),
        )
    except CaptureError:
        raise
    except Exception as e:
        raise CaptureError(f"device refused tune/rate: {e}") from e
    finally:
        dev.close()


ProbeFn = Callable[[str, float, float, float], DeviceReadback | None]


class _ProbeOutcome(BaseModel):
    """error_type is the driver-side exception's class, or None when the
    failure is the mechanism's own."""

    readback: DeviceReadback | None = None
    error: str | None = None
    error_type: str | None = None


def _probe_child(
    probe: ProbeFn,
    device: str,
    sample_rate: float,
    center_hz: float,
    ppm: float,
    send: Connection,
) -> None:
    try:
        outcome = _ProbeOutcome(readback=probe(device, sample_rate, center_hz, ppm))
    except Exception as exc:
        outcome = _ProbeOutcome(
            error=str(exc) or type(exc).__name__, error_type=type(exc).__name__
        )
    send.send(outcome.model_dump_json())


def _run_probe_child(
    budget: float, device: str, sample_rate: float, center_hz: float, ppm: float
) -> _ProbeOutcome | None:
    """None when the child never answered — it is still inside the driver."""
    ctx = worker_context()
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(  # type: ignore[attr-defined]
        target=_probe_child,
        args=(_probe_device, device, sample_rate, center_hz, ppm, send),
    )
    try:
        proc.start()
        send.close()
        payload = receive_payload(recv, budget)
        if payload is None:
            proc.join(_PROBE_REAP_S)
            # drain before reading the exit code, the way the flowgraph path
            # does after its kill: a probe that answered as its budget ran out
            # leaves the readback in the pipe and exits 0, and reaping first
            # threw that answer away to report the clean exit as a death
            payload = receive_payload(recv, 0.0)
        if payload is not None:
            return _ProbeOutcome.model_validate_json(payload)
        if proc.exitcode is None:
            return None
        if proc.exitcode == 0:
            return _ProbeOutcome(
                error=f"the SDR probe (device={device!r}) exited without saying "
                "what the radio answered"
            )
        return _ProbeOutcome(
            error=f"the SDR probe process died (exitcode={proc.exitcode}) "
            f"without reporting (device={device!r}) — the driver took the "
            "process down with it"
        )
    finally:
        reap_process(proc)
        send.close()
        recv.close()


def _probe_within(
    device: str, sample_rate: float, center_hz: float, ppm: float
) -> DeviceReadback | None:
    budget = min(_PROBE_TIMEOUT_S, remaining())
    if budget <= 0.0:
        raise RunTimeout(
            "the call's timeout was spent before the SDR could be probed — "
            "raise timeout"
        )
    outcome = _run_probe_child(budget, device, sample_rate, center_hz, ppm)
    if outcome is None:
        raise RunTimeout(
            f"the SDR (device={device!r}) did not answer enumerate/open/tune "
            f"within {budget:.3g} s and its probe was killed — the radio or its "
            "driver is wedged: replug it, close whatever else is holding it, "
            f"or name another device. capture waits at most {_PROBE_TIMEOUT_S:g} s "
            "for a radio to answer and timeout only lowers that, so a longer "
            "timeout will not open this one"
        )
    if outcome.error is not None:
        driver_side = outcome.error_type not in (None, CaptureError.__name__)
        raise CaptureError(
            f"{outcome.error_type}: {outcome.error}" if driver_side else outcome.error
        )
    return outcome.readback


def _check_free_space(
    out_path: Path, *, num_samples: int, duration_s: float, sample_rate: float
) -> None:
    needed = num_samples * _BYTES_PER_SAMPLE
    free = shutil.disk_usage(out_path.parent).free
    if needed <= free:
        return
    fits_s = free / (sample_rate * _BYTES_PER_SAMPLE)
    raise CaptureError(
        f"this capture needs {needed / _GIB:.1f} GiB (duration_s {duration_s:g} s "
        f"x sample_rate {sample_rate:g} Hz x {_BYTES_PER_SAMPLE} bytes per complex "
        f"sample) and {out_path.parent} has {free / _GIB:.1f} GiB free — lower "
        f"duration_s below {fits_s:.1f} s at this rate, lower sample_rate, or "
        "free space"
    )


def capture_budget_s(duration_s: float, timeout: float | None) -> float:
    """The wall-clock budget the whole call runs under. None derives it from
    the recording, which is the one part whose length the caller already
    stated: a fixed default like the other tools' would kill a legal 300 s
    capture."""
    recording = _SETTLE_S + duration_s
    if timeout is None:
        return recording + _PROBE_TIMEOUT_S + _TIMEOUT_MARGIN_S
    if not math.isfinite(timeout):
        raise ValueError(f"timeout must be a finite number of seconds, got {timeout!r}")
    if timeout <= recording:
        raise ValueError(
            f"timeout {timeout:g} s cannot cover a duration_s {duration_s:g} s "
            f"recording and its {_SETTLE_S:g} s settle transient — raise timeout "
            f"above {recording:g} s, or lower duration_s"
        )
    return timeout


# A recording has no "decoded nothing" outcome: a graph that wrote no samples
# raises below, so EMPTY folds into ERROR. Keyed by the enum and checked for
# completeness at import, so a new RunStatus member cannot reach this bridge as
# a bare KeyError inside a tool call.
_CAPTURE_STATUS: dict[RunStatus, RunStatus] = {
    RunStatus.OK: RunStatus.OK,
    RunStatus.TIMEOUT: RunStatus.TIMEOUT,
    RunStatus.ERROR: RunStatus.ERROR,
    RunStatus.EMPTY: RunStatus.ERROR,
}

_unbridged = sorted(s.value for s in RunStatus if s not in _CAPTURE_STATUS)
if _unbridged:
    raise RuntimeError(f"RunStatus members with no capture outcome: {_unbridged}")


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
    # finiteness first: `nan <= 0` is False, so NaN cleared the old sign
    # check and reached the device as setSampleRate(nan) - and bounds.py
    # exists for exactly this ("non-finite is the one that bit")
    check_sample_rate(sample_rate)
    if sample_rate > _MAX_SAMPLE_RATE:
        raise ValueError(
            f"sample_rate {sample_rate:g} exceeds {_MAX_SAMPLE_RATE:g} Hz — at "
            f"{_BYTES_PER_SAMPLE} bytes per complex sample that is "
            f"{_MAX_SAMPLE_RATE * _BYTES_PER_SAMPLE / 1e9:g} GB/s of sustained "
            "write, past any host bus or disk, and past every streaming SDR"
        )
    for name, value in (("center_hz", center_hz), ("ppm", ppm)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if gain_db is not None and not math.isfinite(gain_db):
        raise ValueError(f"gain_db must be finite, got {gain_db!r}")
    if not 0 < duration_s <= _MAX_DURATION_S:
        raise ValueError(f"duration_s must be in (0, {_MAX_DURATION_S:g}]")
    probe = _probe_within(device, sample_rate, center_hz, ppm)
    settled = _reconcile(sample_rate, center_hz, probe)
    rate, freq = settled.sample_rate, settled.center_hz
    requested_samples = round(duration_s * rate)
    _check_free_space(
        out_path,
        num_samples=requested_samples,
        duration_s=duration_s,
        sample_rate=rate,
    )
    pipeline = build_capture_pipeline(
        out_path=out_path,
        device=device,
        sample_rate=sample_rate,
        center_hz=center_hz,
        gain_db=gain_db,
        ppm=ppm,
        settle_samples=round(_SETTLE_S * rate),
        num_samples=requested_samples,
    )
    check_deadline()
    result = GnuRadioBackend().run_pipeline(
        pipeline,
        timeout=min(_SETTLE_S + duration_s + _TIMEOUT_MARGIN_S, remaining()),
    )
    num_samples = out_path.stat().st_size // 8 if out_path.is_file() else 0
    if num_samples == 0:
        raise CaptureError(
            f"capture produced no samples: {result.error or result.status}"
        )
    levels = _levels(out_path)
    return CaptureResult.build(
        status=_CAPTURE_STATUS[result.status],
        path=str(out_path),
        sample_rate=rate,
        center_hz=freq,
        num_samples=num_samples,
        duration_s=round(num_samples / rate, 3),
        levels=levels,
        overflow_events=result.overflow_events,
        warnings=[
            *settled.warnings,
            *_level_warnings(levels),
            *_overflow_warnings(result.overflow_events),
        ],
        error=result.error,
    )
