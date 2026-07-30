from __future__ import annotations

from marconi.engine.backends.base import Diagnostic
from marconi.engine.run import PipelineResult, _harvest_marks


def test_harvest_marks_reads_bursts_row() -> None:
    rows = [
        Diagnostic(block="b7", key="bursts", marks=[0, 512]),
        Diagnostic(block="b4", key="locks", count=2),
    ]
    assert _harvest_marks(rows) == [0, 512]


def test_pipeline_result_diagnostic_lookup() -> None:
    r = PipelineResult(
        status="ok", diagnostics=[Diagnostic(block="b4", key="locks", count=2)]
    )
    assert r.diagnostic("b4", "locks") is not None
    assert r.diagnostic("b4", "nope") is None
