import pytest
from pydantic import StrictInt, ValidationError

from marconi.engine.types.params import StageParams


class _Demo(StageParams):
    sf: StrictInt
    gray: bool = True


def test_valid_params_parse() -> None:
    p = _Demo.model_validate({"sf": 7})
    assert p.sf == 7 and p.gray is True


def test_unknown_param_is_forbidden() -> None:
    with pytest.raises(ValidationError):
        _Demo.model_validate({"sf": 7, "bogus": 1})


def test_missing_required_param_raises() -> None:
    with pytest.raises(ValidationError):
        _Demo.model_validate({})
