import pytest

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.gnuradio.blocks import _as_float, _as_int


def test_as_float_accepts_numbers() -> None:
    assert _as_float(2) == 2.0 and _as_float(2.5) == 2.5


def test_as_int_accepts_integral_only() -> None:
    assert _as_int(4.0) == 4 and _as_int(7) == 7


def test_as_int_rejects_non_integral_float() -> None:
    # a StrictInt spec never produces 2.7, but the IR-direct path skips
    # pydantic; the backend must not silently truncate (issue 10)
    with pytest.raises(BackendError):
        _as_int(2.7)


@pytest.mark.parametrize("bad", [True, "x", [1.0], None])
def test_coercion_rejects_non_numbers(bad: object) -> None:
    with pytest.raises(BackendError):
        _as_float(bad)  # type: ignore[arg-type]
    with pytest.raises(BackendError):
        _as_int(bad)  # type: ignore[arg-type]
