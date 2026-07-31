from __future__ import annotations

from marconi.engine.backends.base import Diagnostic, find_diagnostic


def test_counter_shape() -> None:
    d = Diagnostic(block="b4", key="locks", count=2)
    assert d.count == 2 and d.marks is None


def test_marks_shape() -> None:
    d = Diagnostic(block="b7", key="bursts", marks=[0, 512])
    assert d.marks == [0, 512] and d.count is None


def test_find_diagnostic() -> None:
    rows = [Diagnostic(block="b7", key="bursts", marks=[0, 512])]
    assert find_diagnostic(rows, "b7", "bursts") is not None
    assert find_diagnostic(rows, "b7", "locks") is None
