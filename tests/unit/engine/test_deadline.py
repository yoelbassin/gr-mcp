import pytest

from marconi.engine.deadline import RunTimeout, check_deadline, remaining, set_deadline
from marconi.errors import classify_error


def test_check_deadline_no_op_without_context():
    check_deadline()  # must not raise
    assert remaining() == float("inf")


def test_expired_deadline_raises():
    with set_deadline(0.0):
        with pytest.raises(RunTimeout):
            check_deadline()


def test_live_deadline_does_not_raise():
    with set_deadline(100.0):
        check_deadline()
        assert 0.0 < remaining() <= 100.0


def test_deadline_resets_on_exit():
    with set_deadline(0.0):
        pass
    check_deadline()  # context cleared; no raise
    assert remaining() == float("inf")


def test_runtimeout_classifies():
    assert classify_error(RunTimeout("late"))[0] == "deadline_exceeded"
