"""survey's channelizer and the channelize stage answer to one rule.

They are different DSP — a streaming numpy polyphase channelizer and a GR
freq_xlating_fir_filter — so they cannot share an implementation. They CAN
share what a caller is allowed to ask for, and until they did, survey validated
`decim >= 1` and nothing else while the stage refused the same request three
ways, under a survey docstring promising both describe that channel alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.stages.conditioning import Channelize, ChannelizeStep
from marconi.engine.stages.registry import step_models
from marconi.engine.types.bounds import MAX_DECIM, channelization_problem
from marconi.survey.iqfile import channelize_to_file

_RATE = 2.048e6

# (decim, bandwidth_hz, why) — each is a sub-band request that must be refused
# by BOTH channelizers, for the same reason.
_REFUSED: list[tuple[int, float, str]] = [
    (1_000_000, 1000.0, "decim past the ceiling"),
    (MAX_DECIM + 1, 1000.0, "decim one past the ceiling"),
    (0, 1000.0, "decim below one"),
    (4, 1.0e9, "passband wider than the decimated rate"),
    (4, 1.0, "passband so narrow the filter cannot be built"),
]


@pytest.fixture
def capture(tmp_path: Path) -> Path:
    p = tmp_path / "iq.cf32"
    tone = np.exp(2j * np.pi * 0.01 * np.arange(200_000)).astype(np.complex64)
    tone.tofile(p)
    return p


@pytest.mark.parametrize("decim, bandwidth_hz, why", _REFUSED, ids=lambda v: str(v))
def test_the_shared_rule_refuses_it(decim: int, bandwidth_hz: float, why: str) -> None:
    assert (
        channelization_problem(
            rate=_RATE, decim=decim, bandwidth_hz=bandwidth_hz, center_hz=0.0
        )
        is not None
    ), why


@pytest.mark.parametrize("decim, bandwidth_hz, why", _REFUSED, ids=lambda v: str(v))
def test_surveys_channelizer_refuses_it(
    capture: Path, tmp_path: Path, decim: int, bandwidth_hz: float, why: str
) -> None:
    with pytest.raises(ValueError):
        channelize_to_file(
            capture,
            tmp_path / "out.cf32",
            _RATE,
            center_hz=0.0,
            decim=decim,
            bandwidth_hz=bandwidth_hz,
        )


@pytest.mark.parametrize("decim, bandwidth_hz, why", _REFUSED, ids=lambda v: str(v))
def test_the_channelize_stage_refuses_it(
    decim: int, bandwidth_hz: float, why: str
) -> None:
    try:
        step = ChannelizeStep.model_validate(
            {"conv": "channelize", "decim": decim, "bandwidth_hz": bandwidth_hz}
        )
    except ValidationError:
        return  # refused at the spec boundary, which is earlier and also fine
    assert Channelize().validate_input_rate(step, _RATE) is not None, why


def test_the_stage_step_model_is_the_one_the_registry_serves() -> None:
    assert step_models()["channelize"] is ChannelizeStep


def test_a_reasonable_sub_band_is_accepted_by_both(
    capture: Path, tmp_path: Path
) -> None:
    written, out_rate = channelize_to_file(
        capture, tmp_path / "out.cf32", _RATE, center_hz=1000.0, decim=4
    )
    assert written > 0
    assert out_rate == pytest.approx(_RATE / 4)
    assert (
        channelization_problem(
            rate=_RATE, decim=4, bandwidth_hz=0.9 * _RATE / 4, center_hz=1000.0
        )
        is None
    )


def test_an_absolute_rf_frequency_still_gets_its_own_hint(
    capture: Path, tmp_path: Path
) -> None:
    """The confusion this message exists for: capture returns an ABSOLUTE tuned
    frequency, survey's center_hz is an offset from the capture's own centre."""
    with pytest.raises(ValueError, match="absolute RF frequency"):
        channelize_to_file(
            capture, tmp_path / "out.cf32", _RATE, center_hz=433.9e6, decim=1
        )
