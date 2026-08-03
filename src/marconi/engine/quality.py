from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from marconi.engine.backends.base import BlockCensus, Diagnostic
from marconi.engine.stages.base import Stage

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


_SURVIVAL_POSITIVE = 0.5
_SURVIVAL_NEGATIVE = 0.1
_LOCK_POSITIVE_PERMILLE = 500


def sync_evidence(
    census: Sequence[BlockCensus], registry: Mapping[str, Stage[Any, Any]]
) -> list[QualityEvidence]:
    out: list[QualityEvidence] = []
    for row in census:
        stage = registry.get(row.kind)
        if stage is None or not stage.sync_search or row.windows_out is None:
            continue
        found = row.windows_out > 0
        out.append(
            QualityEvidence(
                source=row.block,
                metric="sync_matches",
                value=float(row.windows_out),
                assessment="positive" if found else "negative",
            )
        )
    return out


def survival_evidence(census: Sequence[BlockCensus]) -> list[QualityEvidence]:
    out: list[QualityEvidence] = []
    for row in census:
        if row.windows_in is None or row.windows_out is None:
            continue
        if row.windows_in <= 0 or row.windows_out >= row.windows_in:
            continue
        ratio = row.windows_out / row.windows_in
        if ratio >= _SURVIVAL_POSITIVE:
            assessment: Assessment = "positive"
        elif ratio <= _SURVIVAL_NEGATIVE:
            assessment = "negative"
        else:
            continue
        out.append(
            QualityEvidence(
                source=row.block,
                metric="frame_survival",
                value=ratio,
                assessment=assessment,
            )
        )
    return out


def marks_evidence(marks: Sequence[int]) -> list[QualityEvidence]:
    if not marks:
        return []
    return [
        QualityEvidence(
            source="acquisition",
            metric="burst_marks",
            value=float(len(marks)),
            assessment="positive",
        )
    ]


def lock_evidence(diagnostics: Sequence[Diagnostic]) -> list[QualityEvidence]:
    out: list[QualityEvidence] = []
    for d in diagnostics:
        if d.key != "lock_ratio_best_permille" or d.count is None:
            continue
        if d.count >= _LOCK_POSITIVE_PERMILLE:
            out.append(
                QualityEvidence(
                    source=d.block,
                    metric="ofdm_lock_ratio",
                    value=d.count / 1000.0,
                    assessment="positive",
                )
            )
    return out
