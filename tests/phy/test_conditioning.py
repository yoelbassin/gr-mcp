from __future__ import annotations

from marconi.core.stages import validate_params


def test_clock_correct_rejects_pole_ppm() -> None:
    from marconi.phy.stages.conditioning import ClockCorrect

    bad: list = []
    validate_params("clock_correct[0]", ClockCorrect().params_model, {"ppm": -1e6}, bad)
    assert bad, "ppm = -1e6 (pole) should be rejected"


def test_clock_correct_accepts_normal_ppm() -> None:
    from marconi.phy.stages.conditioning import ClockCorrect

    ok: list = []
    validate_params("clock_correct[0]", ClockCorrect().params_model, {"ppm": 6.0}, ok)
    assert not ok, "ppm = 6.0 should be accepted"
