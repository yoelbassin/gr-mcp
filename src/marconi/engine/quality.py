from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

Assessment = Literal["positive", "negative"]
Verdict = Literal["decoded", "uncertain", "no_signal"]


class QualityEvidence(BaseModel):
    source: str
    metric: str
    value: float
    assessment: Assessment


class QualityReport(BaseModel):
    verdict: Verdict
    evidence: list[QualityEvidence] = []
    rationale: str


def verdict_from(evidence: Sequence[QualityEvidence]) -> tuple[Verdict, str]:
    positives = [e for e in evidence if e.assessment == "positive"]
    negatives = [e for e in evidence if e.assessment == "negative"]
    if not evidence:
        return "uncertain", (
            "path produces no checkable evidence (no sync search, no soft "
            "stream, no validating decoder); bits out does not mean signal in"
        )
    if positives and negatives:
        names = ", ".join(sorted({e.metric for e in negatives}))
        return "uncertain", f"conflicting evidence; negative: {names}"
    if negatives:
        names = ", ".join(sorted({e.metric for e in negatives}))
        return "no_signal", f"negative evidence: {names}"
    names = ", ".join(sorted({e.metric for e in positives}))
    return "decoded", f"positive evidence: {names}"
