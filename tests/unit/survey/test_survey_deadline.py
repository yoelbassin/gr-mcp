from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.deadline import RunTimeout, set_deadline
from marconi.survey.iqfile import channelize_to_file, iter_iq, iter_probes
from marconi.survey.measure import survey_iq


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    p = tmp_path / "cap.cf32"
    rng = np.random.default_rng(0)
    n = 1 << 17
    (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64).tofile(p)
    return p


def test_survey_honors_an_expired_deadline(capture: Path) -> None:
    with set_deadline(0.0), pytest.raises(RunTimeout):
        survey_iq(capture, 1.0e6)


def test_streaming_reads_honor_an_expired_deadline(capture: Path) -> None:
    with set_deadline(0.0), pytest.raises(RunTimeout):
        list(iter_iq(capture, 0, 0, chunk=4096))
    with set_deadline(0.0), pytest.raises(RunTimeout):
        list(iter_probes(capture, 0, 0))


def test_channelize_honors_an_expired_deadline(capture: Path, tmp_path: Path) -> None:
    with set_deadline(0.0), pytest.raises(RunTimeout):
        channelize_to_file(
            capture, tmp_path / "ch.cf32", 1.0e6, center_hz=0.0, decim=4
        )


def test_survey_completes_with_a_generous_deadline(capture: Path) -> None:
    with set_deadline(120.0):
        result = survey_iq(capture, 1.0e6)
    assert result.analyzed_samples > 0
