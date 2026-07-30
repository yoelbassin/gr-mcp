from __future__ import annotations

from typing import Literal

import pytest
from pydantic import ValidationError

from marconi.engine.types.step import Step, steps_from_spec


class _FakeStep(Step):
    conv: Literal["fake"] = "fake"
    order: int
    alpha: float = 0.5


def test_step_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _FakeStep(order=4, bogus=1)  # type: ignore[call-arg]


def test_step_carries_conv_and_fields() -> None:
    s = _FakeStep(order=4)
    assert s.conv == "fake"
    assert s.order == 4 and s.alpha == 0.5


def test_steps_from_spec_parses_by_conv() -> None:
    out = steps_from_spec(
        [{"conv": "fake", "order": 8, "alpha": 0.1}], {"fake": _FakeStep}
    )
    assert isinstance(out[0], _FakeStep)
    assert out[0].order == 8 and out[0].alpha == 0.1


def test_steps_from_spec_unknown_conv_raises() -> None:
    with pytest.raises(KeyError):
        steps_from_spec([{"conv": "nope"}], {"fake": _FakeStep})


def test_steps_from_spec_bad_param_raises_eagerly() -> None:
    with pytest.raises(ValidationError):
        steps_from_spec([{"conv": "fake", "order": "x"}], {"fake": _FakeStep})
