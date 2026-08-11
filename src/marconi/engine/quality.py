from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy.ndimage import uniform_filter1d

from marconi.engine.backends.base import (
    BlockCensus,
    Diagnostic,
    DiagnosticRows,
)
from marconi.engine.stages.base import Stage
from marconi.levels import fit_levels

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


# Detection-tier metrics prove a signal is PRESENT (energy/preamble events),
# not that the decode is right: a chirp detector firing on real chirps says
# nothing about the symbols decoded after it. Only decode-tier positives
# (sync matches, word validity, bits-level soft confidence, coherent lock)
# reach "decoded"; detection alone -- a burst mark, or a bare demod's
# per-symbol soft eye (soft_eye) -- stays uncertain.
_DETECTION_METRICS = frozenset({"burst_marks", "soft_eye"})


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
    decode_grade = [e for e in positives if e.metric not in _DETECTION_METRICS]
    if not decode_grade:
        names = ", ".join(sorted({e.metric for e in positives}))
        return "uncertain", (
            f"detection only ({names}): a signal is present but nothing "
            "validated the decoded bits; add a sync/validating/soft stage "
            "to confirm"
        )
    names = ", ".join(sorted({e.metric for e in positives}))
    return "decoded", f"positive evidence: {names}"


# A sync count is only evidence once it clears what pure chance would find on
# random bits: expected chance matches plus a 5-sigma Poisson margin. Short or
# error-tolerant sync words raise the bar on their own; a 32-bit sync keeps
# expecting ~0 so a single hit still counts. Chance is modeled on UNIFORM
# bits: a degenerate low-entropy stream against a matching low-entropy sync
# word can still exceed the floor — measured live (impulsive RFI band):
# 332 found vs 136 chance is 16 sigma yet 100% garbage, so a positive must
# also clear a 3x chance RATIO, which structured interference does not.
_SYNC_CHANCE_SIGMA = 5.0
_SYNC_CHANCE_RATIO = 3.0

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


def _sync_assessment(found: int, expected: float) -> Assessment | None:
    """None = indistinguishable from chance: found something, but no more
    than random bits would match, so it proves nothing either way."""
    if found == 0:
        return "negative"
    if found <= expected + _SYNC_CHANCE_SIGMA * math.sqrt(expected):
        return None
    if found < _SYNC_CHANCE_RATIO * expected:
        return None
    return "positive"


def sync_evidence(
    census: Sequence[BlockCensus], registry: Mapping[str, Stage[Any, Any]]
) -> list[QualityEvidence]:
    out: list[QualityEvidence] = []
    for row in census:
        stage = registry.get(row.kind)
        if stage is None or not stage.sync_search or row.windows_out is None:
            continue
        if row.chance_windows is None:
            # the search never ran (stream shorter than the pattern):
            # untestable is not the same as absent
            continue
        assessment = _sync_assessment(row.windows_out, row.chance_windows)
        if assessment is None:
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


def tag_sync_evidence(diagnostics: Sequence[Diagnostic]) -> list[QualityEvidence]:
    """The GR-side twin of sync_evidence: sync_align's tag_gate counts the
    correlator's sync tags and the chance expectation for the items it
    scanned, so GR-native sync paths contribute the same sync_matches
    evidence the coding-lane sync_word does."""
    rows = DiagnosticRows(diagnostics)
    expected = rows.values("sync_chance")
    scanned = rows.counts("sync_items_scanned")
    out: list[QualityEvidence] = []
    for block, tags in rows.counts("sync_tags").items():
        if not scanned.get(block):
            # the correlator never consumed anything: untestable, not absent
            continue
        assessment = _sync_assessment(tags, expected.get(block, 0.0))
        if assessment is None:
            continue
        out.append(
            QualityEvidence(
                source=block,
                metric="sync_matches",
                value=float(tags),
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
    rows = DiagnosticRows(diagnostics)
    floors = rows.values("lock_min")
    out: list[QualityEvidence] = []
    for block, best in rows.values("lock_ratio_best").items():
        floor = floors.get(block)
        if floor is None:
            # a best-ratio row without its block's configured floor is a
            # malformed pair: untestable, never judged against a default
            continue
        out.append(
            QualityEvidence(
                source=block,
                metric="ofdm_lock_ratio",
                value=best,
                assessment="positive" if best >= floor else "negative",
            )
        )
    return out


# Peak dominance is the CSS analog of soft_confidence: the argmax over the
# dechirped spectrum IS the symbol decision, and peak/median of that vector
# measures how decisively each one was made. The deciding block reports its
# own calibrated tallies and chance ceiling (decision.py holds the measured
# floor); here only the run's fractions are judged, word_validity-style: a
# positive needs a dominant majority plus Chernoff mass against the chance
# ceiling, a negative a near-chance fraction over enough symbols. Like
# soft_confidence this attests decision quality, not symbol identity.
# Measured honest limits (pinned in the css quality tests): around -18 dB
# (symbol errors just beginning) the dominant fraction falls to ~0.08 and
# the run reads uncertain-to-negative, and the worst noise corner (32-bin
# vector at critical sampling, 12% chance tail) reads uncertain rather than
# no_signal — both fail conservative.
_DOMINANCE_POSITIVE = 0.5
_DOMINANCE_NEGATIVE = 0.05
_DOMINANCE_CHANCE_FALLBACK = 0.15


def dominance_evidence(diagnostics: Sequence[Diagnostic]) -> list[QualityEvidence]:
    rows = DiagnosticRows(diagnostics)
    totals = rows.counts("symbols_total")
    chances = rows.values("dominance_chance")
    out: list[QualityEvidence] = []
    for block, dominant in rows.counts("dominant_symbols").items():
        total = totals.get(block)
        if not total:
            # the decider never saw a symbol: untestable, not absent
            continue
        chance = chances.get(block, _DOMINANCE_CHANCE_FALLBACK)
        frac = dominant / total
        if frac >= _DOMINANCE_POSITIVE and _word_excess_significant(
            dominant, total, chance
        ):
            assessment: Assessment = "positive"
        elif frac <= _DOMINANCE_NEGATIVE and total >= _WORD_NEGATIVE_MIN_WORDS:
            assessment = "negative"
        else:
            continue
        out.append(
            QualityEvidence(
                source=block,
                metric="peak_dominance",
                value=frac,
                assessment=assessment,
            )
        )
    return out


_SOFT_MIN_ITEMS = 1000
_SOFT_SAMPLE_ITEMS = 65536
# Evidence values are agent-facing JSON: a degenerate zero-noise stream can
# score an unbounded separation/ratio, so cap what ships (the verdict bars sit
# orders of magnitude below the cap; the cap only tames the reported number).
_SOFT_VALUE_CAP = 1e6

# Measured on a real off-air 4-level FSK capture through a bare demod front
# end: several real inter-burst noise gaps all read order=2, separation
# 2.51-2.78; the capture's real bursts all read order=4, separation
# 5.07-8.12 -- a clean, non-overlapping gap between real noise and real
# signal. The synthetic fixtures corroborate at the extremes (unimodal
# blob ~2.7-3.6, clean synthetic 4-level ~25-40) but say nothing about where
# a REAL demodulated signal lands: the prior 8.0 bar was calibrated against
# synthetics alone and sat above every real burst measured, rejecting all of
# them as no_signal. 4.0 sits with margin above the real noise ceiling
# (~2.8) and below the real signal floor (~5.07).
_SOFT_MULTILEVEL_SEPARATION = 4.0

# Calibrated on measured FSK-discriminator streams (|x| mean/std): clean 19.9,
# SNR 14/9/5/3 dB -> 8.6/5.0/3.1/2.5, demod noise floor 1.6. The positive bar
# admits the whole decodable envelope; the old 6.0 sat above it and starved
# real mid-SNR signals into "uncertain". The bar alone cannot reject a
# wrong-rate decode (2x-oversampled artifact measures 4.6), so a positive
# also demands whitened decisions via consecutive-sign correlation - which
# is ~0 only for SCRAMBLED/random payloads (2x oversample 0.41, tone 1.0).
# HONEST LIMITS, both measured: (a) structured unscrambled data suppresses
# the positive (run-length-8 NRZ +0.87, chip-rate Manchester -0.50, heavy
# 1010 idle <= -0.2) - conservative, the stream reads uncertain, and
# sync/validity evidence still applies; (b) rate errors milder than ~1.2x
# slip under the guard (1.1x measures corr 0.095, ratio 2.85 -> positive),
# and UNDERSAMPLING emits genuinely clean decisions of aliased bits - both
# invisible to every stream statistic. Soft confidence attests decision
# quality, not bit identity; only sync/validity evidence catches aliasing.
_SOFT_POSITIVE = 2.0
_SOFT_NEGATIVE = 1.45
_SOFT_MIN_POLARITY_FRACTION = 0.02
_SOFT_MAX_SIGN_CORR = 0.15

_SOFT_ACTIVE_WINDOW = 64
_SOFT_ACTIVE_HI_PCTILE = 90.0
# Windowed power blurs about half a window of idle into each burst edge, so a
# ~50%-duty stream needs the gate above ~0.3 before that leaked idle drags the
# active-set ratio under _SOFT_POSITIVE.
_SOFT_ACTIVE_FRACTION = 0.35


def _sample_soft(path: Path) -> npt.NDArray[np.float32]:
    """The highest-power contiguous window of the demod stream, never a
    head-only slice (a capture with a noise lead should be judged on the signal
    it decoded, not the lead) and never strided chunks stitched together (which
    break the consecutive decisions the whitening/level statistics rely on). A
    coarse per-block power scan locates the window; it is then read contiguously."""
    total = path.stat().st_size // 4
    with path.open("rb") as f:
        if total <= _SOFT_SAMPLE_ITEMS:
            return np.fromfile(f, dtype=np.float32)
        nblocks = -(-total // _SOFT_SAMPLE_ITEMS)
        powers = np.zeros(nblocks, dtype=np.float64)
        for i in range(nblocks):
            block = np.fromfile(f, dtype=np.float32, count=_SOFT_SAMPLE_ITEMS)
            if block.size:
                powers[i] = float(np.mean(block.astype(np.float64) ** 2))
        start = min(
            int(np.argmax(powers)) * _SOFT_SAMPLE_ITEMS, total - _SOFT_SAMPLE_ITEMS
        )
        f.seek(start * 4)
        return np.fromfile(f, dtype=np.float32, count=_SOFT_SAMPLE_ITEMS)


def _active_mask(x: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    """Sustained-power gate. Idle gaps in a bursty stream sit near zero and drag
    mean|x| under the noise bar; keep only items whose windowed power clears a
    fraction of the stream's high-power level. Flat-power streams (noise,
    continuous signal) pass ~everything, so their statistic is unchanged; only
    bimodal-power (bursty) streams get their idle removed. Keying on windowed
    power (not per-sample magnitude) is what stops it selecting stray high
    noise samples."""
    if x.size <= _SOFT_ACTIVE_WINDOW:
        return np.ones(x.size, dtype=bool)
    power: npt.NDArray[np.float64] = uniform_filter1d(
        (x * x).astype(np.float64), _SOFT_ACTIVE_WINDOW, mode="constant", cval=0.0
    )
    threshold = _SOFT_ACTIVE_FRACTION * float(
        np.percentile(power, _SOFT_ACTIVE_HI_PCTILE)
    )
    return power > threshold


def _emit_soft(
    value: float, assessment: Assessment, decode_grade: bool
) -> list[QualityEvidence]:
    """A bits-level LLR carries a per-bit confidence, so its cleanliness can
    certify OR reject a decode (metric soft_confidence, decode tier). A
    symbols-level demod tap (a bare fsk/msk front end) is a per-symbol eye: it
    attests signal-PRESENT only. It cannot certify the bits -- a mistimed
    matched filter emits a clean eye of wrong symbols (measured: msk on a real
    GMSK capture scores a cleaner eye than the correct fsk decode) -- nor assert
    their ABSENCE -- a correct discriminator decode of a bursty capture reads
    below the noise floor. So it rides the detection tier (metric soft_eye) and
    never contributes a negative."""
    if assessment == "negative" and not decode_grade:
        return []
    return [
        QualityEvidence(
            source="soft_stream",
            metric="soft_confidence" if decode_grade else "soft_eye",
            value=value,
            assessment=assessment,
        )
    ]


def soft_evidence(
    path: Path | None, *, decode_grade: bool = True
) -> list[QualityEvidence]:
    if path is None or not path.is_file():
        return []
    x = _sample_soft(path)
    x = x[np.isfinite(x)]
    if x.size < _SOFT_MIN_ITEMS:
        return []
    active = _active_mask(x)
    if not active.any():
        return _emit_soft(0.0, "negative", decode_grade)
    xa = x[active]
    if xa.size < _SOFT_MIN_ITEMS:
        return []
    fit = fit_levels(xa)
    if fit.order > 2 and fit.separation >= _SOFT_MULTILEVEL_SEPARATION:
        return _emit_soft(
            float(min(fit.separation, _SOFT_VALUE_CAP)), "positive", decode_grade
        )
    mag = np.abs(xa)
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
        min(float((xa > 0).mean()), float((xa < 0).mean()))
        >= _SOFT_MIN_POLARITY_FRACTION
    )
    signs = np.sign(x)
    pair = active[1:] & active[:-1]
    sign_corr = float(np.mean((signs[1:] * signs[:-1])[pair])) if pair.any() else 0.0
    whitened = abs(sign_corr) <= _SOFT_MAX_SIGN_CORR
    if ratio >= _SOFT_POSITIVE and both_polarities and whitened:
        assessment: Assessment = "positive"
    elif ratio <= _SOFT_NEGATIVE:
        assessment = "negative"
    else:
        return []
    return _emit_soft(float(min(ratio, _SOFT_VALUE_CAP)), assessment, decode_grade)


def assess_quality(
    *,
    registry: Mapping[str, Stage[Any, Any]],
    census: Sequence[BlockCensus],
    diagnostics: Sequence[Diagnostic],
    marks: Sequence[int],
    soft_stream: Path | None,
    soft_decode_grade: bool = True,
) -> QualityReport:
    evidence = (
        sync_evidence(census, registry)
        + tag_sync_evidence(diagnostics)
        + survival_evidence(census, registry)
        + marks_evidence(marks)
        + lock_evidence(diagnostics)
        + dominance_evidence(diagnostics)
        + soft_evidence(soft_stream, decode_grade=soft_decode_grade)
    )
    verdict, rationale = verdict_from(evidence)
    return QualityReport(verdict=verdict, evidence=evidence, rationale=rationale)
