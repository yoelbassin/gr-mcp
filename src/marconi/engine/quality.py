from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
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


_SOFT_MIN_ITEMS = 1000
_SOFT_SAMPLE_ITEMS = 65536
_SOFT_POSITIVE = 1.8
_SOFT_NEGATIVE = 1.45


def soft_evidence(path: Path | None) -> list[QualityEvidence]:
    if path is None or not path.is_file():
        return []
    with path.open("rb") as f:
        x = np.fromfile(f, dtype=np.float32, count=_SOFT_SAMPLE_ITEMS)
    x = x[np.isfinite(x)]
    if x.size < _SOFT_MIN_ITEMS:
        return []
    mag = np.abs(x)
    spread = float(mag.std())
    mean = float(mag.mean())
    if spread == 0.0:
        ratio = np.inf if mean > 0.0 else 0.0
    else:
        ratio = mean / spread
    if ratio >= _SOFT_POSITIVE:
        assessment: Assessment = "positive"
    elif ratio <= _SOFT_NEGATIVE:
        assessment = "negative"
    else:
        return []
    return [
        QualityEvidence(
            source="soft_stream",
            metric="soft_confidence",
            value=float(min(ratio, 1e6)),
            assessment=assessment,
        )
    ]
