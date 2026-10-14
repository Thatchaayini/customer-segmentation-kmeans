import pytest

from logger import get_logger, timed_step


def test_logger_is_reused():
    assert get_logger("same_name") is get_logger("same_name")


def test_timed_step_logs_and_returns_value(caplog):
    @timed_step
    def add(a, b):
        return a + b

    with caplog.at_level("INFO"):
        assert add(1, 2) == 3


def test_timed_step_reraises_errors():
    @timed_step
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        boom()