from __future__ import annotations

from marconi.engine.backends.base import Diagnostic, find_diagnostic
from marconi.engine.backends.gnuradio.worker import _harvest_diagnostics


class _Blk:
    def __init__(self, diagnostics: object) -> None:
        self.diagnostics = diagnostics


class _Tb:
    def __init__(self, instances: dict[str, object]) -> None:
        self._py_instances = instances


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


def test_harvest_returns_typed_counter_and_marks_rows() -> None:
    tb = _Tb({"b4": _Blk({"locks": 2}), "b7": _Blk({"bursts": [0, 512]})})
    rows = _harvest_diagnostics(tb)
    assert all(isinstance(d, Diagnostic) for d in rows)
    by = {(d.block, d.key): d for d in rows}
    assert by[("b4", "locks")].count == 2 and by[("b4", "locks")].marks is None
    assert by[("b7", "bursts")].marks == [0, 512] and by[("b7", "bursts")].count is None


def test_harvest_skips_blocks_without_diagnostics() -> None:
    tb = _Tb({"b0": _Blk(None), "b1": _Blk({})})
    assert _harvest_diagnostics(tb) == []
