from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers.hardware import sdr_present

from marconi.capture import capture_iq
from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.survey import survey_iq

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.xdist_group("sdr"),
    pytest.mark.skipif(not sdr_present(), reason="no SDR attached"),
]


def test_live_capture_one_second(tmp_path: Path) -> None:
    ensure_worker_warm()
    out = tmp_path / "iq.cf32"
    res = capture_iq(out, center_hz=100.0e6, sample_rate=1.024e6, duration_s=1.0)
    assert res.status == "ok", res
    assert res.path == str(out)
    assert res.num_samples == round(1.0 * res.sample_rate)
    assert out.stat().st_size == res.num_samples * 8
    data = np.fromfile(out, dtype=np.complex64)
    assert float(np.sqrt(np.mean(np.abs(data) ** 2))) > 0.0
    assert res.levels.rms > 0.0
    assert 0.0 <= res.levels.clip_fraction <= 1.0
    assert res.duration_s == pytest.approx(1.0, abs=0.01)
    survey = survey_iq(out, res.sample_rate)
    assert survey.spectrum.psd_db, "capture not survey-able"
