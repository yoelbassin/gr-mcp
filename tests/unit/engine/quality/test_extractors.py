from __future__ import annotations

from typing import Any

from marconi.engine.backends.base import BlockCensus, Diagnostic
from marconi.engine.quality import (
    Verdict,
    dominance_evidence,
    lock_evidence,
    marks_evidence,
    survival_evidence,
    sync_evidence,
    tag_sync_evidence,
    verdict_from,
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


def test_sync_search_zero_over_a_long_stream_is_negative() -> None:
    ev = sync_evidence(
        [_row("sync_word", windows_out=0, chance_windows=1e-5, items_in=60_000)],
        stage_registry(),
    )
    assert [e.assessment for e in ev] == ["negative"]


def test_sync_search_zero_over_a_short_stream_is_not_evidence() -> None:
    """Zero hits is only "absent" if the search had room to find one. Without
    items_in the row cannot claim it scanned anything, and a genuinely short
    scan cannot either — both must read untestable rather than no_signal."""
    short = _row("sync_word", windows_out=0, chance_windows=1e-5, items_in=64)
    unreported = _row("sync_word", windows_out=0, chance_windows=1e-5)
    assert sync_evidence([short], stage_registry()) == []
    assert sync_evidence([unreported], stage_registry()) == []


def test_sync_hits_at_chance_level_are_not_evidence() -> None:
    # the C1 repro shape: an 8-bit sync on 60k random bits expects ~234 chance
    # matches and finds 233 - that must never read as positive
    rows = [_row("sync_word", windows_out=233, chance_windows=234.4)]
    assert sync_evidence(rows, stage_registry()) == []


def test_sync_hits_well_above_chance_floor_are_positive() -> None:
    # well above: 4.27x ratio, comfortably clears both 5-sigma and 3x gates
    rows = [_row("sync_word", windows_out=1000, chance_windows=234.4)]
    ev = sync_evidence(rows, stage_registry())
    assert [e.assessment for e in ev] == ["positive"]


def test_a_chance_of_zero_certifies_nothing() -> None:
    """A non-positive expectation is the ABSENCE of a null, not a strict one:
    the 5-sigma and 3x bars both multiply it, so 0.0 clears them for any hit
    count and ONE match read verdict "decoded".

    Reachable without a malformed harvest: _chance_valid_rate underflows to
    exactly 0.0 past ~1075 pattern bits and _surrogate_chance returns 0 when the
    rotation matches nowhere, so sync_word_rx's max() of the two is 0.0. Zero
    hits stay negative — that reading needs no chance model."""
    for hits in (1, 5, 6250):
        rows = [
            _row("sync_word", windows_out=hits, chance_windows=0.0, items_in=60_000)
        ]
        assert sync_evidence(rows, stage_registry()) == []
    absent = [_row("sync_word", windows_out=0, chance_windows=0.0, items_in=60_000)]
    assert [e.assessment for e in sync_evidence(absent, stage_registry())] == [
        "negative"
    ]


def test_both_sync_lanes_agree_about_a_missing_chance_model() -> None:
    """The rule lived only in tag_sync_evidence, so the GR lane refused to judge
    a zero expectation while the coding lane certified on it — the same reading,
    two answers. It now sits in _sync_assessment, which both lanes call."""
    coding = sync_evidence(
        [_row("sync_word", windows_out=1, chance_windows=0.0, items_in=60_000)],
        stage_registry(),
    )
    gr = tag_sync_evidence(
        [
            Diagnostic(block="sync_align[0]", key="sync_tags", count=1),
            Diagnostic(block="sync_align[0]", key="sync_items_scanned", count=60_000),
            Diagnostic(block="sync_align[0]", key="sync_chance", value=0.0),
        ]
    )
    assert coding == gr == []


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


def _tag_rows(tags: int, chance: float, scanned: int = 60_000) -> list[Diagnostic]:
    return [
        Diagnostic(block="b6", key="sync_tags", count=tags),
        Diagnostic(block="b6", key="sync_chance", value=chance),
        Diagnostic(block="b6", key="sync_items_scanned", count=scanned),
    ]


def test_tag_sync_above_chance_is_positive() -> None:
    ev = tag_sync_evidence(_tag_rows(12, 0.5))
    assert [e.assessment for e in ev] == ["positive"]
    assert ev[0].metric == "sync_matches"
    assert ev[0].value == 12.0


def test_tag_sync_without_a_chance_cannot_certify() -> None:
    # a zero expectation is no expectation: it clears both the sigma and the
    # ratio bar, so it would promote any hit to positive. tag_gate refuses to
    # be built without a real chance; one arriving here is untestable
    assert tag_sync_evidence(_tag_rows(12, 0.0)) == []


def test_tag_sync_at_chance_level_is_no_evidence() -> None:
    assert tag_sync_evidence(_tag_rows(233, 234.4)) == []


def test_tag_sync_zero_is_negative() -> None:
    ev = tag_sync_evidence(_tag_rows(0, 0.0))
    assert [e.assessment for e in ev] == ["negative"]


def test_tag_sync_never_scanned_is_untestable_not_negative() -> None:
    # a stream too short for the correlator consumed nothing: zero tags is
    # not evidence of absence
    assert tag_sync_evidence(_tag_rows(0, 0.0, scanned=0)) == []


def test_unrelated_diagnostics_are_not_tag_sync_evidence() -> None:
    rows = [Diagnostic(block="b4", key="locks", count=2)]
    assert tag_sync_evidence(rows) == []


def _dom_rows(dominant: int, total: int, chance: float = 0.15) -> list[Diagnostic]:
    return [
        Diagnostic(block="b9", key="dominant_symbols", count=dominant),
        Diagnostic(block="b9", key="symbols_total", count=total),
        Diagnostic(block="b9", key="dominance_chance", value=chance),
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


def test_cp_lock_ratio_is_detection_grade_and_cannot_certify_alone() -> None:
    # Tier's own docstring: "a chirp detector firing on real chirps says
    # nothing about the symbols decoded after it. Only DECODE positives reach
    # 'decoded'." A cyclic-prefix correlation peak is computed before any
    # symbol decision - it is exactly that detector, and it rode DECODE.
    ev = lock_evidence(
        [
            Diagnostic(block="cp[0]", key="lock_ratio_best", value=4.5),
            Diagnostic(block="cp[0]", key="lock_min", value=2.0),
        ]
    )
    assert [e.assessment for e in ev] == ["positive"]
    verdict, _ = verdict_from(ev)
    assert verdict is Verdict.UNCERTAIN


def test_cp_lock_below_floor_is_not_evidence_of_absence() -> None:
    # a non-OFDM transmission fails a CP search without being absent, and a
    # detection-grade reading cannot assert absence (the soft_eye rule)
    ev = lock_evidence(
        [
            Diagnostic(block="cp[0]", key="lock_ratio_best", value=1.5),
            Diagnostic(block="cp[0]", key="lock_min", value=2.0),
        ]
    )
    assert ev == []


def test_pilot_score_is_the_decode_grade_lock() -> None:
    # the pilot-lattice score measures the equalizer's consistency on the
    # decoded lattice itself, so it may certify AND refute
    good = lock_evidence(
        [
            Diagnostic(block="pl[0]", key="lock_score_best", value=0.9),
            Diagnostic(block="pl[0]", key="lock_score_min", value=0.35),
        ]
    )
    assert [e.assessment for e in good] == ["positive"]
    verdict, _ = verdict_from(good)
    assert verdict is Verdict.DECODED
    bad = lock_evidence(
        [
            Diagnostic(block="pl[0]", key="lock_score_best", value=0.1),
            Diagnostic(block="pl[0]", key="lock_score_min", value=0.35),
        ]
    )
    assert [e.assessment for e in bad] == ["negative"]


def test_lock_without_configured_floor_is_untestable() -> None:
    # the block always ships lock_min alongside lock_ratio_best; a best-ratio
    # row arriving alone is a malformed pair and must never be judged against
    # a smuggled default (the C3 hole was exactly such a default sitting below
    # the block's own noise calibration)
    rows = [Diagnostic(block="b4", key="lock_ratio_best", value=1.6)]
    assert lock_evidence(rows) == []
    rows = [Diagnostic(block="b4", key="lock_ratio_best", value=2.4)]
    assert lock_evidence(rows) == []


def test_a_weak_code_cannot_certify_noise_as_decoded() -> None:
    # A code that accepts half of all random words carries about one check
    # bit: 100/100 "valid" words is what noise looks like through it, not
    # evidence of a decode. Coverage stopped at chance 0.0625 and jumped to
    # the degenerate 1.0 (which _word_excess_significant rejects on its own),
    # leaving the whole band where _WORD_CHANCE_MAX is the only thing
    # standing between a chance-level code and a "decoded" verdict.
    rows = [_row("block_code", words_valid=100, words_total=100, chance_word_rate=0.5)]
    assert survival_evidence(rows, stage_registry()) == []


def test_a_handful_of_tags_over_a_chance_of_one_proves_nothing() -> None:
    # found=4 against an expectation of 1.0 clears the 3x ratio bar, so only
    # the Poisson margin withholds the verdict. Every existing case was
    # decided by the ratio bar, so nothing exercised the margin at all.
    rows = [
        Diagnostic(block="gate[1]", key="sync_tags", count=4),
        Diagnostic(block="gate[1]", key="sync_items_scanned", count=4096),
        Diagnostic(block="gate[1]", key="sync_chance", value=1.0),
    ]
    assert tag_sync_evidence(rows) == []


def test_zero_chance_word_rate_is_untestable_not_positive() -> None:
    # _chance_valid_rate underflows to exactly 0.0 past ~1075 redundancy bits
    # (block_code(2048, 512) is a legal spec). A zero null is the ABSENCE of a
    # null - both positive bars multiply it - so a dead all-zero demod scored
    # 300/300 "valid" words and the run read "decoded". The sync lane's
    # _sync_assessment already refuses a non-positive expectation; this is its
    # word-lane twin.
    rows = [_row("block_code", words_valid=300, words_total=300, chance_word_rate=0.0)]
    assert survival_evidence(rows, stage_registry()) == []


def test_one_perfect_word_does_not_certify() -> None:
    # 255 bytes of silence through RS(255,223) is 1/1 valid at chance 2.6e-14:
    # the Chernoff bar is cleared by a single word once the chance is small
    # enough, and the negative arm's min-words floor had no positive twin.
    rows = [_row("rs_code", words_valid=1, words_total=1, chance_word_rate=2.6e-14)]
    assert survival_evidence(rows, stage_registry()) == []


def test_constant_codewords_earn_no_positive() -> None:
    # The all-zero word is a codeword of every linear code, so a dead demod
    # satisfies the code with probability 1, not chance_word_rate: validity of
    # a constant word is free and must not be judged against the uniform null.
    rows = [
        _row(
            "rs_code",
            words_valid=400,
            words_total=400,
            words_constant=400,
            chance_word_rate=2.6e-14,
        )
    ]
    assert survival_evidence(rows, stage_registry()) == []


def test_mostly_constant_valid_words_read_negative() -> None:
    # RS drags near-silence onto the zero codeword and tallies it as a repair:
    # 247 free words plus 153 outright failures is garbage. With the free
    # words excluded, the remaining 0/153 is decisively negative.
    rows = [
        _row(
            "rs_code",
            words_valid=247,
            words_total=400,
            words_constant=247,
            chance_word_rate=2.6e-14,
        )
    ]
    ev = survival_evidence(rows, stage_registry())
    assert [e.assessment for e in ev] == ["negative"]


def test_padding_heavy_real_decode_is_still_positive() -> None:
    # Constant words are excluded, never fatal: a genuine frame train whose
    # payload contains zero-padding words keeps its positive from the varied
    # words that remain.
    rows = [
        _row(
            "block_code",
            words_valid=400,
            words_total=400,
            words_constant=100,
            chance_word_rate=1e-6,
        )
    ]
    ev = survival_evidence(rows, stage_registry())
    assert [e.assessment for e in ev] == ["positive"]


def test_gr_tags_with_unreported_extent_still_certify() -> None:
    # the coding lane's documented rule ("a POSITIVE needs no extent - the
    # hits themselves are the evidence") applies to the GR lane too: the
    # chance expectation already embeds the scanned extent, so hits above it
    # certify whether or not the extent counter made it into diagnostics
    rows = [
        Diagnostic(block="gate[1]", key="sync_tags", count=40),
        Diagnostic(block="gate[1]", key="sync_chance", value=1e-5),
    ]
    ev = tag_sync_evidence(rows)
    assert [e.assessment for e in ev] == ["positive"]


def test_search_hit_marks_are_not_burst_evidence() -> None:
    # sync_symbols REPLACES the carried burst marks with its pattern hits, and
    # marks_evidence is the one extractor with no null at all: 9997 hits of an
    # alternating pattern on an alternating stream read "burst_marks positive"
    # while the stage's own description promises it contributes NO evidence.
    from marconi.engine.quality import assess_quality

    report = assess_quality(
        registry=stage_registry(),
        census=[_row("sync_symbols", items_in=4000, items_out=4000)],
        diagnostics=[],
        marks=[10, 400, 3999],
        soft_stream=None,
    )
    assert all(e.metric != "burst_marks" for e in report.evidence)


def test_detector_marks_without_a_search_stage_still_count() -> None:
    # the control: marks from a burst DETECTOR (no mark-replacing search ran)
    # keep their detection-grade positive
    from marconi.engine.quality import assess_quality

    report = assess_quality(
        registry=stage_registry(),
        census=[_row("fsk", items_in=4000, items_out=400)],
        diagnostics=[],
        marks=[10, 400],
        soft_stream=None,
    )
    assert [e.metric for e in report.evidence] == ["burst_marks"]
