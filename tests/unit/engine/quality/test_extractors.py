from __future__ import annotations

from typing import Any

from marconi.engine.backends.base import BlockCensus, Diagnostic
from marconi.engine.quality import (
    lock_evidence,
    marks_evidence,
    survival_evidence,
    sync_evidence,
)
from marconi.engine.stages.registry import stage_registry


def _row(kind: str, **kw: Any) -> BlockCensus:
    return BlockCensus(block=f"{kind}[3]", kind=kind, **kw)


def test_sync_hits_above_chance_are_positive() -> None:
    # a 32-bit sync over any realistic stream expects ~0 chance matches, so a
    # single hit clears the floor
    ev = sync_evidence(
        [_row("sync_word", windows_out=5, chance_windows=1e-5)], stage_registry()
    )
    assert [e.assessment for e in ev] == ["positive"]
    assert ev[0].metric == "sync_matches"
    assert ev[0].value == 5.0


def test_sync_search_zero_is_negative() -> None:
    ev = sync_evidence(
        [_row("sync_word", windows_out=0, chance_windows=1e-5)], stage_registry()
    )
    assert [e.assessment for e in ev] == ["negative"]


def test_sync_hits_at_chance_level_are_not_evidence() -> None:
    # the C1 repro shape: an 8-bit sync on 60k random bits expects ~234 chance
    # matches and finds 233 - that must never read as positive
    rows = [_row("sync_word", windows_out=233, chance_windows=234.4)]
    assert sync_evidence(rows, stage_registry()) == []


def test_sync_hits_well_above_chance_floor_are_positive() -> None:
    rows = [_row("sync_word", windows_out=400, chance_windows=234.4)]
    ev = sync_evidence(rows, stage_registry())
    assert [e.assessment for e in ev] == ["positive"]


def test_unconditional_seeders_are_not_evidence() -> None:
    rows = [_row("segment", windows_out=9), _row("mark_frame", windows_out=1)]
    assert sync_evidence(rows, stage_registry()) == []


def test_gr_census_rows_are_skipped() -> None:
    assert sync_evidence([_row("symbol_sync_cc", items_out=7)], stage_registry()) == []


def test_word_validity_high_positive_chance_level_negative() -> None:
    high = survival_evidence(
        [_row("block_code", words_valid=95, words_total=100, chance_word_rate=0.031)],
        stage_registry(),
    )
    garbage = survival_evidence(
        [_row("block_code", words_valid=3, words_total=100, chance_word_rate=0.031)],
        stage_registry(),
    )
    assert [e.assessment for e in high] == ["positive"]
    assert high[0].metric == "word_validity"
    assert [e.assessment for e in garbage] == ["negative"]


def test_word_validity_midband_is_skipped() -> None:
    rows = [_row("block_code", words_valid=30, words_total=100, chance_word_rate=0.031)]
    assert survival_evidence(rows, stage_registry()) == []


def test_tiny_word_counts_are_not_positive_evidence() -> None:
    # 2/2 chance-valid words at the 0.125 cap is 23%-likely garbage: a perfect
    # ratio without statistical mass must not read as signal
    for total in (1, 2):
        rows = [
            _row(
                "block_code",
                words_valid=total,
                words_total=total,
                chance_word_rate=0.0625,
            )
        ]
        assert survival_evidence(rows, stage_registry()) == []


def test_tiny_word_counts_are_not_negative_evidence() -> None:
    rows = [_row("block_code", words_valid=0, words_total=2, chance_word_rate=0.03)]
    assert survival_evidence(rows, stage_registry()) == []


def test_short_stream_sync_search_is_untestable_not_negative() -> None:
    # a stream shorter than the sync pattern never ran a search: no
    # chance_windows stat, so zero windows must not read as "no signal"
    rows = [_row("sync_word", windows_out=0, chance_windows=None)]
    assert sync_evidence(rows, stage_registry()) == []


def test_perfect_code_cannot_discriminate_and_is_skipped() -> None:
    # Hamming(7,4): every syndrome is correctable, chance-valid rate 1.0 - a
    # 100% valid tally on garbage must not read as signal
    rows = [_row("block_code", words_valid=100, words_total=100, chance_word_rate=1.0)]
    assert survival_evidence(rows, stage_registry()) == []


def test_rows_without_word_tallies_are_not_evidence() -> None:
    rows = [_row("block_code", windows_in=10, windows_out=10)]
    assert survival_evidence(rows, stage_registry()) == []


def test_non_validating_stage_is_not_survival_evidence() -> None:
    rows = [_row("codebook", words_valid=1, words_total=10, chance_word_rate=0.01)]
    assert survival_evidence(rows, stage_registry()) == []


def test_survival_unknown_kind_is_skipped() -> None:
    rows = [
        _row("symbol_sync_cc", words_valid=1, words_total=10, chance_word_rate=0.01)
    ]
    assert survival_evidence(rows, stage_registry()) == []


def test_marks_positive_only() -> None:
    assert [e.assessment for e in marks_evidence([0, 512])] == ["positive"]
    assert marks_evidence([]) == []


def test_lock_against_configured_floor() -> None:
    locked = lock_evidence(
        [
            Diagnostic(block="b4", key="lock_ratio_best_permille", count=2500),
            Diagnostic(block="b4", key="lock_min_permille", count=2000),
        ]
    )
    unlocked = lock_evidence(
        [
            Diagnostic(block="b4", key="lock_ratio_best_permille", count=1500),
            Diagnostic(block="b4", key="lock_min_permille", count=2000),
        ]
    )
    assert [e.assessment for e in locked] == ["positive"]
    assert [e.assessment for e in unlocked] == ["negative"]


def test_lock_noise_floor_reading_is_not_positive_without_min() -> None:
    # cp_symbol_sync's own calibration: pure noise measures 1300-1600 permille;
    # the C3 hole was a 500-permille bar sitting below that floor
    noise = lock_evidence(
        [Diagnostic(block="b4", key="lock_ratio_best_permille", count=1600)]
    )
    assert [e.assessment for e in noise] == ["negative"]
    locked = lock_evidence(
        [Diagnostic(block="b4", key="lock_ratio_best_permille", count=2400)]
    )
    assert [e.assessment for e in locked] == ["positive"]
