from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.capture import CaptureError
from marconi.capture.record import _levels, _reconcile
from marconi.errors import classify_error


def test_capture_error_classified() -> None:
    assert classify_error(CaptureError("boom")) == ("capture", "boom")


def test_reconcile_exact_match_no_warnings() -> None:
    rate, freq, warnings = _reconcile(2.048e6, 100e6, (2.048e6, 100e6))
    assert (rate, freq, warnings) == (2.048e6, 100e6, [])


def test_reconcile_rate_snap_warns_and_returns_actual() -> None:
    rate, freq, warnings = _reconcile(1.9e6, 100e6, (1.92e6, 100e6))
    assert rate == 1.92e6
    assert freq == 100e6
    assert len(warnings) == 1 and "sample_rate" in warnings[0]


def test_reconcile_small_freq_offset_warns() -> None:
    rate, freq, warnings = _reconcile(2.048e6, 100e6, (2.048e6, 100.0001e6))
    assert freq == 100.0001e6
    assert len(warnings) == 1 and "center_hz" in warnings[0]


def test_reconcile_gross_freq_divergence_raises() -> None:
    with pytest.raises(CaptureError):
        _reconcile(2.048e6, 1.8e9, (2.048e6, 1.766e9))


def test_reconcile_no_bindings_falls_back_to_requested() -> None:
    rate, freq, warnings = _reconcile(2.048e6, 100e6, None)
    assert (rate, freq) == (2.048e6, 100e6)
    assert len(warnings) == 1 and "unverified" in warnings[0]


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
