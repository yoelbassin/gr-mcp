from __future__ import annotations

from marconi.engine.quality import Assessment, QualityEvidence, verdict_from


def _ev(assessment: Assessment, metric: str = "m") -> QualityEvidence:
    return QualityEvidence(
        source="s[0]", metric=metric, value=1.0, assessment=assessment
    )


def test_only_positives_is_decoded() -> None:
    verdict, rationale = verdict_from([_ev("positive"), _ev("positive")])
    assert verdict == "decoded"
    assert rationale


def test_only_negatives_is_no_signal() -> None:
    verdict, _ = verdict_from([_ev("negative")])
    assert verdict == "no_signal"


def test_mixed_is_uncertain() -> None:
    verdict, _ = verdict_from([_ev("positive"), _ev("negative")])
    assert verdict == "uncertain"


def test_no_evidence_is_uncertain_with_honest_rationale() -> None:
    verdict, rationale = verdict_from([])
    assert verdict == "uncertain"
    assert "no checkable evidence" in rationale
