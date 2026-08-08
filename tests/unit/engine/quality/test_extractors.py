from __future__ import annotations

from typing import Any

from marconi.engine.backends.base import BlockCensus, Diagnostic
from marconi.engine.quality import (
    dominance_evidence,
    lock_evidence,
    marks_evidence,
    survival_evidence,
    sync_evidence,
    tag_sync_evidence,
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
    # well above chance means >3x ratio to filter structured interference
    rows = [_row("sync_word", windows_out=704, chance_windows=234.4)]
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


def _tag_rows(tags: int, chance_micro: int, scanned: int = 60_000) -> list[Diagnostic]:
    return [
        Diagnostic(block="b6", key="sync_tags", count=tags),
        Diagnostic(block="b6", key="sync_chance_micro", count=chance_micro),
        Diagnostic(block="b6", key="sync_items_scanned", count=scanned),
    ]


def test_tag_sync_above_chance_is_positive() -> None:
    ev = tag_sync_evidence(_tag_rows(12, 0))
    assert [e.assessment for e in ev] == ["positive"]
    assert ev[0].metric == "sync_matches"
    assert ev[0].value == 12.0


def test_tag_sync_at_chance_level_is_no_evidence() -> None:
    assert tag_sync_evidence(_tag_rows(233, 234_400_000)) == []


def test_tag_sync_zero_is_negative() -> None:
    ev = tag_sync_evidence(_tag_rows(0, 0))
    assert [e.assessment for e in ev] == ["negative"]


def test_tag_sync_never_scanned_is_untestable_not_negative() -> None:
    # a stream too short for the correlator consumed nothing: zero tags is
    # not evidence of absence
    assert tag_sync_evidence(_tag_rows(0, 0, scanned=0)) == []


def test_unrelated_diagnostics_are_not_tag_sync_evidence() -> None:
    rows = [Diagnostic(block="b4", key="locks", count=2)]
    assert tag_sync_evidence(rows) == []


def _dom_rows(
    dominant: int, total: int, chance_micro: int = 150_000
) -> list[Diagnostic]:
    return [
        Diagnostic(block="b9", key="dominant_symbols", count=dominant),
        Diagnostic(block="b9", key="symbols_total", count=total),
        Diagnostic(block="b9", key="dominance_chance_micro", count=chance_micro),
    ]


def test_dominance_majority_with_mass_is_positive() -> None:
    ev = dominance_evidence(_dom_rows(78, 78))
    assert [e.assessment for e in ev] == ["positive"]
    assert ev[0].metric == "peak_dominance"
    assert ev[0].value == 1.0


def test_dominance_tiny_run_has_no_mass() -> None:
    # 5/5 dominant at a 0.15 chance ceiling is only ~9.5 nats of surprise —
    # a perfect fraction without statistical mass must not read as signal
    assert dominance_evidence(_dom_rows(5, 5)) == []


def test_dominance_near_chance_is_negative() -> None:
    ev = dominance_evidence(_dom_rows(4, 100))
    assert [e.assessment for e in ev] == ["negative"]


def test_dominance_midband_is_silent() -> None:
    # the deep-SNR edge (measured ~0.08-0.3 dominant near where symbol
    # errors begin) must stay uncertain, never no_signal
    assert dominance_evidence(_dom_rows(30, 100)) == []
    assert dominance_evidence(_dom_rows(8, 100)) == []


def test_dominance_negative_needs_mass() -> None:
    assert dominance_evidence(_dom_rows(0, 4)) == []


def test_dominance_without_total_is_untestable() -> None:
    rows = [Diagnostic(block="b9", key="dominant_symbols", count=3)]
    assert dominance_evidence(rows) == []


def test_dominance_zero_symbols_is_untestable() -> None:
    assert dominance_evidence(_dom_rows(0, 0)) == []


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
