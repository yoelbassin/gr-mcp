from __future__ import annotations

import math
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
            "path produces no checkable evidence (no sync matches beyond "
            "chance, no soft stream, no validating decoder); bits out does "
            "not mean signal in"
        )
    if positives and negatives:
        names = ", ".join(sorted({e.metric for e in negatives}))
        return "uncertain", f"conflicting evidence; negative: {names}"
    if negatives:
        names = ", ".join(sorted({e.metric for e in negatives}))
        return "no_signal", f"negative evidence: {names}"
    names = ", ".join(sorted({e.metric for e in positives}))
    return "decoded", f"positive evidence: {names}"


# A sync count is only evidence once it clears what pure chance would find on
# random bits: expected chance matches plus a 5-sigma Poisson margin. Short or
# error-tolerant sync words raise the bar on their own; a 32-bit sync keeps
# expecting ~0 so a single hit still counts. Chance is modeled on UNIFORM
# bits: a degenerate low-entropy stream against a matching low-entropy sync
# word can still exceed the floor.
_SYNC_CHANCE_SIGMA = 5.0

_WORD_VALIDITY_POSITIVE = 0.5
_WORD_VALIDITY_NEGATIVE = 0.1
# A code whose chance-valid rate exceeds this cannot separate signal from
# noise (a perfect code validates EVERY word), so it contributes no evidence.
_WORD_CHANCE_MAX = 0.125
# The ratio bar alone leaks at tiny word counts (2/2 chance-valid words at the
# 0.125 cap is 23%-likely garbage), so a positive also needs Chernoff-bound
# significance: P(valid >= k | chance) <= exp(-total*D(k/total||chance)) must
# sit at ~5-sigma odds (exp(-15) ~ 3e-7). Negatives need a minimum mass too -
# two failed words say nothing.
_WORD_EXCESS_MIN_LOG_ODDS = 15.0
_WORD_NEGATIVE_MIN_WORDS = 8


def _word_excess_significant(valid: int, total: int, chance: float) -> bool:
    q = valid / total
    p = max(chance, 1e-300)
    if q <= p:
        return False
    if q >= 1.0:
        log_odds = total * math.log(1.0 / p)
    else:
        log_odds = total * (
            q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))
        )
    return log_odds >= _WORD_EXCESS_MIN_LOG_ODDS


# Fallback when a lock diagnostic arrives without its block's configured
# threshold; matches cp_symbol_sync's calibrated default (noise ~1300-1600
# permille, real lock >= 2000).
_LOCK_MIN_FALLBACK_PERMILLE = 2000


def sync_evidence(
    census: Sequence[BlockCensus], registry: Mapping[str, Stage[Any, Any]]
) -> list[QualityEvidence]:
    out: list[QualityEvidence] = []
    for row in census:
        stage = registry.get(row.kind)
        if stage is None or not stage.sync_search or row.windows_out is None:
            continue
        expected = row.chance_windows if row.chance_windows is not None else 0.0
        floor = expected + _SYNC_CHANCE_SIGMA * float(np.sqrt(expected))
        if row.windows_out == 0:
            if row.chance_windows is None:
                # the search never ran (stream shorter than the pattern):
                # untestable is not the same as absent
                continue
            assessment: Assessment = "negative"
        elif row.windows_out > floor:
            assessment = "positive"
        else:
            # found something, but no more than random bits would match:
            # indistinguishable from chance, so it proves nothing either way
            continue
        out.append(
            QualityEvidence(
                source=row.block,
                metric="sync_matches",
                value=float(row.windows_out),
                assessment=assessment,
            )
        )
    return out


def survival_evidence(
    census: Sequence[BlockCensus], registry: Mapping[str, Stage[Any, Any]]
) -> list[QualityEvidence]:
    out: list[QualityEvidence] = []
    for row in census:
        stage = registry.get(row.kind)
        if stage is None or not stage.validates_words:
            continue
        if not row.words_total or row.words_valid is None:
            continue
        chance = row.chance_word_rate
        if chance is None or chance > _WORD_CHANCE_MAX:
            continue
        ratio = row.words_valid / row.words_total
        if ratio >= _WORD_VALIDITY_POSITIVE and _word_excess_significant(
            row.words_valid, row.words_total, chance
        ):
            assessment: Assessment = "positive"
        elif (
            ratio <= max(_WORD_VALIDITY_NEGATIVE, 2.0 * chance)
            and row.words_total >= _WORD_NEGATIVE_MIN_WORDS
        ):
            assessment = "negative"
        else:
            continue
        out.append(
            QualityEvidence(
                source=row.block,
                metric="word_validity",
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
    floors = {
        d.block: d.count
        for d in diagnostics
        if d.key == "lock_min_permille" and d.count is not None
    }
    out: list[QualityEvidence] = []
    for d in diagnostics:
        if d.key != "lock_ratio_best_permille" or d.count is None:
            continue
        floor = floors.get(d.block, _LOCK_MIN_FALLBACK_PERMILLE)
        out.append(
            QualityEvidence(
                source=d.block,
                metric="ofdm_lock_ratio",
                value=d.count / 1000.0,
                assessment="positive" if d.count >= floor else "negative",
            )
        )
    return out


_SOFT_MIN_ITEMS = 1000
_SOFT_SAMPLE_ITEMS = 65536
_SOFT_POSITIVE = 6.0
_SOFT_NEGATIVE = 1.45
_SOFT_MIN_POLARITY_FRACTION = 0.02


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
    # A constant or one-sided LLR stream (a CW carrier) can score an
    # arbitrarily high magnitude ratio without carrying any modulation; both
    # polarities must show up in real proportion before that ratio counts.
    both_polarities = (
        min(float((x > 0).mean()), float((x < 0).mean())) >= _SOFT_MIN_POLARITY_FRACTION
    )
    if ratio >= _SOFT_POSITIVE and both_polarities:
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


def assess_quality(
    *,
    registry: Mapping[str, Stage[Any, Any]],
    census: Sequence[BlockCensus],
    diagnostics: Sequence[Diagnostic],
    marks: Sequence[int],
    soft_stream: Path | None,
) -> QualityReport:
    evidence = (
        sync_evidence(census, registry)
        + survival_evidence(census, registry)
        + marks_evidence(marks)
        + lock_evidence(diagnostics)
        + soft_evidence(soft_stream)
    )
    verdict, rationale = verdict_from(evidence)
    return QualityReport(verdict=verdict, evidence=evidence, rationale=rationale)
