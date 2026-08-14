from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
    written and reports what the driver said about dropped buffers."""

    def __init__(self, overflow_events: int) -> None:
        self._overflow_events = overflow_events

    def run_pipeline(self, pipeline: GrPipeline, timeout: float = 30.0) -> RunResult:
        path = Path(str(pipeline.block("sink").params["path"]))
        _write_cf32(path, np.full(2048, 0.25 + 0.25j))
        return RunResult(
            status=RunStatus.OK,
            artifacts=[str(path)],
            overflow_events=self._overflow_events,
        )


def _stub_probe(device: str, rate: float, freq: float, ppm: float) -> DeviceReadback:
    return DeviceReadback(sample_rate=rate, center_hz=freq)


def _stub_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, overflow_events: int
) -> CaptureResult:
    monkeypatch.setattr(record_mod, "_probe_device", _stub_probe)
    monkeypatch.setattr(
        record_mod, "GnuRadioBackend", lambda: _StubDriver(overflow_events)
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
