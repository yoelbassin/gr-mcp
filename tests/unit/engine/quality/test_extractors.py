from __future__ import annotations

from marconi.engine.backends.base import BlockCensus, Diagnostic
from marconi.engine.quality import (
    lock_evidence,
    marks_evidence,
    survival_evidence,
    sync_evidence,
)
from marconi.engine.stages.registry import stage_registry


def _row(kind: str, **kw: int | None) -> BlockCensus:
    return BlockCensus(block=f"{kind}[3]", kind=kind, **kw)


def test_sync_search_hits_are_positive() -> None:
    ev = sync_evidence([_row("sync_word", windows_out=5)], stage_registry())
    assert [e.assessment for e in ev] == ["positive"]
    assert ev[0].metric == "sync_matches"
    assert ev[0].value == 5.0


def test_sync_search_zero_is_negative() -> None:
    ev = sync_evidence([_row("sync_word", windows_out=0)], stage_registry())
    assert [e.assessment for e in ev] == ["negative"]


def test_unconditional_seeders_are_not_evidence() -> None:
    rows = [_row("segment", windows_out=9), _row("mark_frame", windows_out=1)]
    assert sync_evidence(rows, stage_registry()) == []


def test_gr_census_rows_are_skipped() -> None:
    assert sync_evidence([_row("symbol_sync_cc", items_out=7)], stage_registry()) == []


def test_survival_high_positive_low_negative_equal_skipped() -> None:
    high = survival_evidence(
        [_row("block_code", windows_in=10, windows_out=8)], stage_registry()
    )
    low = survival_evidence(
        [_row("block_code", windows_in=10, windows_out=1)], stage_registry()
    )
    same = survival_evidence(
        [_row("block_code", windows_in=10, windows_out=10)], stage_registry()
    )
    assert [e.assessment for e in high] == ["positive"]
    assert [e.assessment for e in low] == ["negative"]
    assert same == []


def test_survival_midband_is_skipped() -> None:
    rows = [_row("block_code", windows_in=10, windows_out=3)]
    assert survival_evidence(rows, stage_registry()) == []


def test_non_validating_stage_is_not_survival_evidence() -> None:
    rows = [_row("codebook", windows_in=10, windows_out=1)]
    assert survival_evidence(rows, stage_registry()) == []


def test_survival_unknown_kind_is_skipped() -> None:
    rows = [_row("symbol_sync_cc", windows_in=10, windows_out=1)]
    assert survival_evidence(rows, stage_registry()) == []


def test_marks_positive_only() -> None:
    assert [e.assessment for e in marks_evidence([0, 512])] == ["positive"]
    assert marks_evidence([]) == []


def test_ofdm_lock_positive_only() -> None:
    good = lock_evidence(
        [Diagnostic(block="b4", key="lock_ratio_best_permille", count=900)]
    )
    weak = lock_evidence(
        [Diagnostic(block="b4", key="lock_ratio_best_permille", count=80)]
    )
    assert [e.assessment for e in good] == ["positive"]
    assert weak == []
