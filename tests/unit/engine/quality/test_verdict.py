from __future__ import annotations

from marconi.engine.quality import (
    Assessment,
    QualityEvidence,
    QualityMetric,
    verdict_from,
)


def _ev(
    assessment: Assessment,
    metric: QualityMetric = QualityMetric.SYNC_MATCHES,
) -> QualityEvidence:
    return QualityEvidence(
        source="s[0]", metric=metric, value=1.0, assessment=assessment
    )


def test_only_positives_is_decoded() -> None:
    verdict, rationale = verdict_from(
        [_ev(Assessment.POSITIVE), _ev(Assessment.POSITIVE)]
    )
    assert verdict == "decoded"
    assert rationale


def test_only_negatives_is_no_signal() -> None:
    verdict, _ = verdict_from([_ev(Assessment.NEGATIVE)])
    assert verdict == "no_signal"


def test_mixed_is_uncertain() -> None:
    verdict, _ = verdict_from([_ev(Assessment.POSITIVE), _ev(Assessment.NEGATIVE)])
    assert verdict == "uncertain"


def test_no_evidence_is_uncertain_with_honest_rationale() -> None:
    verdict, rationale = verdict_from([])
    assert verdict == "uncertain"
    assert "no checkable evidence" in rationale


def test_detection_only_positives_are_not_decoded() -> None:
    # a burst detector firing proves presence, not a correct decode: a CSS
    # path that detects real chirp preambles but decodes garbage symbols must
    # not read "decoded" on detection alone
    verdict, rationale = verdict_from(
        [_ev(Assessment.POSITIVE, metric=QualityMetric.BURST_MARKS)]
    )
    assert verdict == "uncertain"
    assert "detection only" in rationale


def test_detection_plus_decode_grade_positive_is_decoded() -> None:
    verdict, _ = verdict_from(
        [
            _ev(Assessment.POSITIVE, metric=QualityMetric.BURST_MARKS),
            _ev(Assessment.POSITIVE, metric=QualityMetric.SYNC_MATCHES),
        ]
    )
    assert verdict == "decoded"


def test_bare_demod_soft_eye_alone_is_not_decoded() -> None:
    # a bare demod's per-symbol eye attests signal-present, not a validated
    # decode: soft_eye is detection-tier, so it stays "uncertain" on its own
    verdict, rationale = verdict_from(
        [_ev(Assessment.POSITIVE, metric=QualityMetric.SOFT_EYE)]
    )
    assert verdict == "uncertain"
    assert "detection only" in rationale
