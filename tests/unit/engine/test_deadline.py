import pytest

from marconi.engine.deadline import RunTimeout, check_deadline, remaining, set_deadline
from marconi.errors import classify_error


def test_check_deadline_no_op_without_context() -> None:
    check_deadline()  # must not raise
    assert remaining() == float("inf")


def test_expired_deadline_raises() -> None:
    with set_deadline(0.0):
        with pytest.raises(RunTimeout):
            check_deadline()


def test_live_deadline_does_not_raise() -> None:
    with set_deadline(100.0):
        check_deadline()
        assert 0.0 < remaining() <= 100.0


def test_deadline_resets_on_exit() -> None:
    with set_deadline(0.0):
        pass
    check_deadline()  # context cleared; no raise
    assert remaining() == float("inf")


def test_nested_set_deadline_keeps_the_earliest_deadline() -> None:
    with set_deadline(0.0):
        with set_deadline(1000.0):  # a much longer nested budget must not win
            with pytest.raises(RunTimeout):
                check_deadline()


def test_nested_set_deadline_restores_the_outer_deadline_on_exit() -> None:
    with set_deadline(1000.0):
        with set_deadline(0.0):
            with pytest.raises(RunTimeout):
                check_deadline()
        check_deadline()  # back to the still-live outer deadline; no raise
        assert 0.0 < remaining() <= 1000.0


def test_runtimeout_classifies() -> None:
    assert classify_error(RunTimeout("late"))[0] == "deadline_exceeded"
