from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.engine.stages.conditioning import ClockCorrectStep


def test_clock_correct_rejects_pole_ppm() -> None:
    with pytest.raises(ValidationError):
        ClockCorrectStep(ppm=-1e6)


def test_clock_correct_accepts_normal_ppm() -> None:
    ClockCorrectStep(ppm=6.0)
