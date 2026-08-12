from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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


def test_bump_accepts_a_counter_numpy_already_widened() -> None:
    # np.count_nonzero and friends return np.int64, which is NOT an
    # isinstance of int. A counter that absorbed one raised on the NEXT bump,
    # inside general_work, so a decode that had already run correctly came
    # back as status error / "embedded block raised". np.float64 passes the
    # float branch either way, which is what made the asymmetry invisible.
    from marconi.engine.backends.gnuradio.embedded.lifecycle import Diagnostics, bump

    diag: Diagnostics = {"hits": np.int64(3)}  # type: ignore[dict-item]
    bump(diag, "hits", int(np.count_nonzero(np.ones(4))))
    assert diag["hits"] == 7
    assert isinstance(diag["hits"], int)


def test_an_unharvestable_diagnostic_does_not_fail_a_run_that_worked() -> None:
    # _harvest_census states the rule outright ("never let a diagnostic fail
    # a run that otherwise worked") and guards itself; its sibling did not, so
    # one stray value type raised past the ok path AND past both error paths,
    # turning a flowgraph that ran to completion and wrote its sinks into
    # "worker exited abnormally" with no stream, census or quality — and
    # masking the real error when there was one.
    from marconi.engine.backends.gnuradio.worker import _harvest_diagnostics

    blk = SimpleNamespace(diagnostics={"good": 4, "bad": {"not": "harvestable"}})
    rows = _harvest_diagnostics(SimpleNamespace(_py_instances={"b0": blk}))
    assert [r.key for r in rows if r.key == "good"] == ["good"]
    assert not [r for r in rows if r.key == "bad"]


def test_a_dropped_diagnostic_is_counted_not_silently_swallowed() -> None:
    # dropping it silently was the other half of the rename trap: a producer
    # whose value type went wrong looked exactly like a producer with nothing
    # to say, and the weakened verdict had no visible cause in the response
    from marconi.engine.backends.gnuradio.worker import (
        _UNHARVESTABLE,
        _harvest_diagnostics,
    )

    tb = SimpleNamespace(
        _py_instances={
            "b0": SimpleNamespace(diagnostics={"ok": 1, "bad": object()}),
            "b1": SimpleNamespace(diagnostics={"ok": 2}),
        }
    )
    rows = _harvest_diagnostics(tb)
    dropped = [r for r in rows if r.key == _UNHARVESTABLE]
    assert [(r.block, r.count) for r in dropped] == [("b0", 1)]


def test_a_clean_harvest_reports_no_drops() -> None:
    from marconi.engine.backends.gnuradio.worker import (
        _UNHARVESTABLE,
        _harvest_diagnostics,
    )

    tb = SimpleNamespace(
        _py_instances={
            "b0": SimpleNamespace(diagnostics={"locks": 1, "r": 2.0, "m": [1, 2]})
        }
    )
    rows = _harvest_diagnostics(tb)
    assert not [r for r in rows if r.key == _UNHARVESTABLE]
    assert len(rows) == 3
