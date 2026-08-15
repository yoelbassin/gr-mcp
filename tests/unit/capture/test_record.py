from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from helpers._capture import (
    SNAPPED_RATE,
    WEDGED_SECONDS,
    crashing_probe,
    refusing_probe,
    snapping_probe,
    stub_probe,
    wedged_probe,
)
from helpers._deadline import deadline_expiring_after

from marconi.capture import CaptureError
from marconi.capture import record as record_mod
from marconi.capture.record import (
    CaptureLevels,
    CaptureResult,
    DeviceReadback,
    _level_warnings,
    _levels,
    _reconcile,
    build_capture_pipeline,
    capture_iq,
)
from marconi.deadline import RunTimeout, set_deadline
from marconi.engine.backends.base import RunResult
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.types.enums import RunStatus
from marconi.errors import classify_error
from marconi.mcp.tools import capture_tool


def test_capture_error_classified() -> None:
    assert classify_error(CaptureError("boom")) == ("failed_precondition", "boom")


def _readback(sample_rate: float, center_hz: float) -> DeviceReadback:
    return DeviceReadback(sample_rate=sample_rate, center_hz=center_hz)


def test_reconcile_exact_match_no_warnings() -> None:
    got = _reconcile(2.048e6, 100e6, _readback(2.048e6, 100e6))
    assert (got.sample_rate, got.center_hz, got.warnings) == (2.048e6, 100e6, [])


def test_reconcile_rate_snap_warns_and_returns_actual() -> None:
    got = _reconcile(1.9e6, 100e6, _readback(1.92e6, 100e6))
    assert got.sample_rate == 1.92e6
    assert got.center_hz == 100e6
    assert len(got.warnings) == 1 and "sample_rate" in got.warnings[0]


def test_reconcile_small_freq_offset_warns() -> None:
    got = _reconcile(2.048e6, 100e6, _readback(2.048e6, 100.0001e6))
    assert got.center_hz == 100.0001e6
    assert len(got.warnings) == 1 and "center_hz" in got.warnings[0]


def test_reconcile_gross_freq_divergence_raises() -> None:
    with pytest.raises(CaptureError):
        _reconcile(2.048e6, 1.8e9, _readback(2.048e6, 1.766e9))


def test_reconcile_no_bindings_falls_back_to_requested() -> None:
    got = _reconcile(2.048e6, 100e6, None)
    assert (got.sample_rate, got.center_hz) == (2.048e6, 100e6)
    assert len(got.warnings) == 1 and "unverified" in got.warnings[0]


def test_level_warnings_flag_heavy_clipping() -> None:
    w = _level_warnings(CaptureLevels(rms=0.8, dc_offset=0.0, clip_fraction=0.33))
    assert len(w) == 1 and "clip" in w[0].lower()


def test_level_warnings_flag_dead_rms() -> None:
    w = _level_warnings(CaptureLevels(rms=0.0001, dc_offset=0.0, clip_fraction=0.0))
    assert len(w) == 1 and "rms" in w[0].lower()


def test_level_warnings_silent_on_healthy_capture() -> None:
    healthy = CaptureLevels(rms=0.12, dc_offset=0.0, clip_fraction=0.0)
    assert _level_warnings(healthy) == []


def _write_cf32(path: Path, z: np.ndarray) -> Path:
    z.astype(np.complex64).tofile(path)
    return path


def test_levels_constant_half_scale(tmp_path: Path) -> None:
    path = _write_cf32(tmp_path / "half.cf32", np.full(4096, 0.5 + 0.0j))
    lv = _levels(path)
    assert lv.rms == pytest.approx(0.5, abs=1e-3)
    assert lv.dc_offset == pytest.approx(0.5, abs=1e-3)
    assert lv.clip_fraction == 0.0


def test_levels_full_scale_clips(tmp_path: Path) -> None:
    path = _write_cf32(tmp_path / "clip.cf32", np.full(4096, 1.0 + 1.0j))
    assert _levels(path).clip_fraction == 1.0


def test_levels_zero_mean_tone_has_no_dc(tmp_path: Path) -> None:
    n = 8192
    tone = 0.7 * np.exp(2j * np.pi * 0.125 * np.arange(n))
    lv = _levels(_write_cf32(tmp_path / "tone.cf32", tone))
    assert lv.rms == pytest.approx(0.7, abs=1e-3)
    assert lv.dc_offset == pytest.approx(0.0, abs=1e-3)
    assert lv.clip_fraction == 0.0


def test_pipeline_shape_agc(tmp_path: Path) -> None:
    p = build_capture_pipeline(
        out_path=tmp_path / "iq.cf32",
        device="",
        sample_rate=2.048e6,
        center_hz=100e6,
        gain_db=None,
        ppm=0.0,
        settle_samples=409600,
        num_samples=10240000,
    )
    assert [b.kind for b in p.blocks] == [
        "soapy_source",
        "iq_skiphead",
        "iq_head",
        "iq_file_sink",
    ]
    src = p.block("src")
    assert src.params["agc"] is True
    assert "gain_db" not in src.params
    assert src.params["sample_rate"] == 2.048e6
    assert src.params["center_hz"] == 100e6
    assert p.block("settle").params["num_items"] == 409600
    assert p.block("bound").params["num_items"] == 10240000
    assert p.block("sink").params["path"] == str(tmp_path / "iq.cf32")
    assert [(c.src_block, c.dst_block) for c in p.connections] == [
        ("src", "settle"),
        ("settle", "bound"),
        ("bound", "sink"),
    ]


class _StubDriver:
    """Stands in for the SDR run: writes the samples the flowgraph would have
    written and reports what the driver said about the outcome and about
    dropped buffers. A run that ends in TIMEOUT or ERROR still leaves whatever
    the file sink had written — the partial capture the tool documents."""

    def __init__(
        self,
        overflow_events: int = 0,
        status: RunStatus = RunStatus.OK,
        samples: int = 2048,
        error: str | None = None,
    ) -> None:
        self._overflow_events = overflow_events
        self._status = status
        self._samples = samples
        self._error = error

    def run_pipeline(self, pipeline: GrPipeline, timeout: float = 30.0) -> RunResult:
        path = Path(str(pipeline.block("sink").params["path"]))
        artifacts: list[str] = []
        if self._samples:
            _write_cf32(path, np.full(self._samples, 0.25 + 0.25j))
            artifacts.append(str(path))
        return RunResult(
            status=self._status,
            artifacts=artifacts,
            error=self._error,
            overflow_events=self._overflow_events,
        )


def _drive_capture(
    monkeypatch: pytest.MonkeyPatch, driver: _StubDriver
) -> pytest.MonkeyPatch:
    monkeypatch.setattr(record_mod, "_probe_device", stub_probe)
    monkeypatch.setattr(record_mod, "GnuRadioBackend", lambda: driver)
    return monkeypatch


def _stub_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    overflow_events: int = 0,
    status: RunStatus = RunStatus.OK,
    samples: int = 2048,
    error: str | None = None,
) -> CaptureResult:
    _drive_capture(
        monkeypatch,
        _StubDriver(
            overflow_events=overflow_events,
            status=status,
            samples=samples,
            error=error,
        ),
    )
    return capture_iq(
        tmp_path / "iq.cf32",
        center_hz=100e6,
        sample_rate=2.048e6,
        duration_s=0.01,
    )


def test_capture_reports_driver_drops_and_warns_the_file_has_gaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A driver overflow splices time out of the recording while the item
    count still arrives, so status stays ok — the count and the warning are
    the only evidence the samples are not contiguous."""
    cap = _stub_capture(monkeypatch, tmp_path, overflow_events=4)
    assert cap.status == RunStatus.OK
    assert cap.overflow_events == 4
    (warning,) = [w for w in cap.warnings if "overflow" in w.lower()]
    assert "4" in warning and "gap" in warning.lower()
    assert cap.as_payload()["overflow_events"] == 4


def test_clean_capture_reports_no_drops_without_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = _stub_capture(monkeypatch, tmp_path, overflow_events=0)
    assert cap.overflow_events == 0
    assert [w for w in cap.warnings if "overflow" in w.lower()] == []
    # 0 ships rather than being omitted: "the driver reported no drops" is an
    # answer the agent needs, and an absent key cannot carry it
    assert cap.as_payload()["overflow_events"] == 0


def test_capture_docstring_documents_the_drops_the_payload_carries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool docstring is product: it tells the agent that status ok with
    overflow_events > 0 is a time-spliced file, so the surface has to be able
    to produce exactly that pair."""
    doc = capture_tool.__doc__ or ""
    assert "overflow_events" in doc
    payload = _stub_capture(monkeypatch, tmp_path, overflow_events=3).as_payload()
    assert payload["status"] == "ok"
    assert payload["overflow_events"] == 3


def test_a_wedged_probe_is_killed_and_the_call_answers_on_the_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one tool with a hang path: enumerate/open/tune are blocking driver
    calls, so a wedged dongle held the call forever — measured 6.01 s for a 6 s
    block even inside set_deadline(0.1), because a C call polls nothing. The
    probe runs in a child now, and the bound is on the parent's clock."""
    _drive_capture(monkeypatch, _StubDriver())
    monkeypatch.setattr(record_mod, "_probe_device", wedged_probe)
    start = time.monotonic()
    with set_deadline(2.0), pytest.raises(RunTimeout) as excinfo:
        capture_iq(tmp_path / "iq.cf32", center_hz=100e6, duration_s=0.01)
    elapsed = time.monotonic() - start
    assert elapsed < WEDGED_SECONDS / 2, f"probe ran unbounded for {elapsed:.1f}s"
    assert classify_error(excinfo.value)[0] == "deadline_exceeded"
    assert "device" in str(excinfo.value).lower()


def test_the_devices_readback_crosses_back_from_the_probes_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe runs in another process now, so what it learned about the
    radio has to survive the trip: a rate the device snapped to is the value
    the capture reports and warns about, not the requested one."""
    _drive_capture(monkeypatch, _StubDriver())
    monkeypatch.setattr(record_mod, "_probe_device", snapping_probe)
    cap = capture_iq(
        tmp_path / "iq.cf32", center_hz=100e6, sample_rate=2.048e6, duration_s=0.01
    )
    assert cap.sample_rate == SNAPPED_RATE
    (warning,) = [w for w in cap.warnings if "sample_rate" in w]
    assert "unverified" not in warning


def test_a_probes_refusal_crosses_back_as_the_refusal_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _drive_capture(monkeypatch, _StubDriver())
    monkeypatch.setattr(record_mod, "_probe_device", refusing_probe)
    with pytest.raises(CaptureError) as excinfo:
        capture_iq(tmp_path / "iq.cf32", center_hz=100e6, duration_s=0.01)
    code, message = classify_error(excinfo.value)
    assert code == "failed_precondition"
    assert "no SDR device found" in message
    assert "CaptureError:" not in message


def test_a_driver_that_kills_its_own_process_is_told_apart_from_a_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashed probe answers nothing, exactly like a wedged one; only the
    child's exit code separates "replug the radio" from "wait longer"."""
    _drive_capture(monkeypatch, _StubDriver())
    monkeypatch.setattr(record_mod, "_probe_device", crashing_probe)
    with pytest.raises(CaptureError) as excinfo:
        capture_iq(tmp_path / "iq.cf32", center_hz=100e6, duration_s=0.01)
    assert "70" in str(excinfo.value)


def test_the_probe_budget_is_the_runs_remaining_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministically, on a fake clock: the probe's own cap is only half the
    bound — a call whose deadline is already spent must not open a device at
    all, let alone wait the cap out. Both halves are asserted, because a probe
    bounded by its own constant alone still raises RunTimeout eventually: the
    wedged message and _PROBE_TIMEOUT_S of wall clock are what separate them."""
    monkeypatch.setattr(record_mod, "_probe_device", wedged_probe)
    start = time.monotonic()
    with deadline_expiring_after(monkeypatch, polls=0):
        with pytest.raises(RunTimeout) as excinfo:
            capture_iq(tmp_path / "iq.cf32", center_hz=100e6, duration_s=0.01)
    elapsed = time.monotonic() - start
    assert "spent before" in str(excinfo.value)
    assert elapsed < 5.0, f"a spent deadline still opened a device for {elapsed:.1f}s"


def test_the_probe_cap_only_ever_tightens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rule the tool docstring states, on the number itself: the probe's
    ceiling is a fixed constant that a longer timeout cannot raise, and only a
    tighter run deadline lowers it."""
    budgets: list[float] = []

    def _record_budget(
        budget: float, device: str, rate: float, freq: float, ppm: float
    ) -> record_mod._ProbeOutcome:
        budgets.append(budget)
        return record_mod._ProbeOutcome(
            readback=DeviceReadback(sample_rate=rate, center_hz=freq)
        )

    _drive_capture(monkeypatch, _StubDriver())
    monkeypatch.setattr(record_mod, "_run_probe_child", _record_budget)
    for timeout in (3600.0, 1.5):
        with set_deadline(timeout):
            capture_iq(tmp_path / "iq.cf32", center_hz=100e6, duration_s=0.01)
    generous, tight = budgets
    assert generous == pytest.approx(record_mod._PROBE_TIMEOUT_S)
    assert tight < record_mod._PROBE_TIMEOUT_S


def test_capture_refuses_a_sample_rate_no_host_could_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sample_rate sizes the write here, and check_sample_rate only asked for
    finite and > 0: 1e12 was reconciled verbatim and compiled into a
    3.000e+14-item head block (2183 TiB) with a clean status."""
    _drive_capture(monkeypatch, _StubDriver())
    with pytest.raises(ValueError) as excinfo:
        capture_iq(
            tmp_path / "iq.cf32",
            center_hz=100e6,
            sample_rate=1e12,
            duration_s=0.01,
        )
    code, message = classify_error(excinfo.value)
    assert code == "invalid_argument"
    assert "sample_rate" in message
    assert not (tmp_path / "iq.cf32").exists()


def test_capture_refuses_a_recording_the_disk_cannot_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """num_samples is known before the graph starts and was never compared to
    free space: a request that cannot fit ran to its timeout writing until the
    volume filled. The refusal names both parameters that sized it."""
    _drive_capture(monkeypatch, _StubDriver())
    with pytest.raises(CaptureError) as excinfo:
        capture_iq(
            tmp_path / "iq.cf32",
            center_hz=100e6,
            sample_rate=1e9,
            duration_s=300.0,
        )
    code, message = classify_error(excinfo.value)
    assert code == "failed_precondition"
    assert "duration_s" in message and "sample_rate" in message
    assert not (tmp_path / "iq.cf32").exists()


def test_a_capture_that_fits_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _stub_capture(monkeypatch, tmp_path).status == RunStatus.OK


def test_a_timed_out_run_reports_the_partial_file_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The docstring's partial-capture contract: status is the run's, and the
    samples that reached the file are real and countable."""
    cap = _stub_capture(
        monkeypatch,
        tmp_path,
        status=RunStatus.TIMEOUT,
        samples=512,
        error="run exceeded timeout; worker killed",
    )
    assert cap.status == RunStatus.TIMEOUT
    assert cap.num_samples == 512
    payload = cap.as_payload()
    assert payload["status"] == "timeout"
    partial = Path(str(payload["path"]))
    assert partial.is_file() and partial.stat().st_size == 512 * 8
    assert payload["error"]


def test_a_run_that_wrote_nothing_raises_instead_of_reporting_a_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(CaptureError, match="no samples"):
        _stub_capture(monkeypatch, tmp_path, status=RunStatus.ERROR, samples=0)


def test_an_empty_run_that_wrote_samples_is_reported_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recording has no "decoded nothing" outcome, so EMPTY folds into
    ERROR — the bridge no test had ever crossed."""
    cap = _stub_capture(monkeypatch, tmp_path, status=RunStatus.EMPTY, samples=256)
    assert cap.status == RunStatus.ERROR
    assert cap.as_payload()["status"] == "error"


def test_capture_docstring_partial_capture_claim_is_what_the_tool_ships(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The docstring tells the agent a non-ok status with a path is a partial
    capture whose samples are usable — a claim nothing executed."""
    doc = capture_tool.__doc__ or ""
    assert "partial capture" in doc
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    _drive_capture(monkeypatch, _StubDriver(status=RunStatus.TIMEOUT, samples=512))
    payload = capture_tool(center_hz=100e6, sample_rate=2.048e6, duration_s=0.01)
    assert payload["status"] == "timeout"
    assert payload["num_samples"] == 512
    partial = Path(str(payload["path"]))
    assert partial.is_file() and partial.stat().st_size == 512 * 8
    assert np.fromfile(partial, dtype=np.complex64).size == 512


def test_pipeline_manual_gain_and_ppm(tmp_path: Path) -> None:
    p = build_capture_pipeline(
        out_path=tmp_path / "iq.cf32",
        device="driver=rtlsdr",
        sample_rate=1.024e6,
        center_hz=433.92e6,
        gain_db=28.0,
        ppm=2.5,
        settle_samples=204800,
        num_samples=1024000,
    )
    src = p.block("src")
    assert src.params["agc"] is False
    assert src.params["gain_db"] == 28.0
    assert src.params["ppm"] == 2.5
    assert src.params["device"] == "driver=rtlsdr"
