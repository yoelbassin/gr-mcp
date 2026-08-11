import pytest

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.gnuradio.blocks import BlockParams
from marconi.engine.types.params import ParamValue


def _p(**values: ParamValue) -> BlockParams:
    return BlockParams(dict(values))


def test_reads_numbers_by_type() -> None:
    p = _p(a=2, b=2.5)
    assert p.f("a") == 2.0 and p.f("b") == 2.5
    assert p.i("a") == 2


def test_int_read_accepts_integral_float_only() -> None:
    assert _p(n=4.0).i("n") == 4 and _p(n=7).i("n") == 7


def test_int_read_rejects_non_integral_float() -> None:
    # a StrictInt spec never produces 2.7, but the IR-direct path skips
    # pydantic; the backend must not silently truncate
    with pytest.raises(BackendError, match="sps"):
        _p(sps=2.7).i("sps")


@pytest.mark.parametrize("bad", [True, "x", [1.0]])
def test_number_reads_reject_non_numbers(bad: ParamValue) -> None:
    for read in (BlockParams.f, BlockParams.i):
        with pytest.raises(BackendError, match="gain"):
            read(_p(gain=bad), "gain")


def test_string_read_rejects_a_non_string() -> None:
    # str(v) would have stringified a list into a plausible-looking path
    with pytest.raises(BackendError, match="path"):
        _p(path=[1.0, 2.0]).s("path")


def test_bool_read_rejects_a_non_bool() -> None:
    with pytest.raises(BackendError, match="repeat"):
        _p(repeat=1).b("repeat")


def test_missing_required_param_names_itself() -> None:
    with pytest.raises(BackendError, match="phase_inc"):
        _p().f("phase_inc")


def test_defaults_apply_only_when_absent() -> None:
    assert _p().f("alpha", 0.35) == 0.35
    assert _p(alpha=0.2).f("alpha", 0.35) == 0.2
    assert _p().b("repeat", False) is False
    assert _p().s("device", "") == ""


def test_list_reads_validate_every_element() -> None:
    assert _p(taps=[1, 2.0]).floats("taps") == [1.0, 2.0]
    assert _p(lens=[2, 4.0]).ints("lens") == [2, 4]
    with pytest.raises(BackendError, match="lens"):
        _p(lens=[2, 4.5]).ints("lens")
    with pytest.raises(BackendError, match="taps"):
        _p(taps=3.0).floats("taps")
