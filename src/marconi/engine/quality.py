from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from marconi.deadline import check_deadline
from marconi.engine.backends.base import (
    BlockCensus,
    Diagnostic,
    DiagnosticKey,
    DiagnosticRows,
)
from marconi.engine.stages.base import Stage
from marconi.engine.types.enums import ItemType
from marconi.levelfit import fit_levels, windowed_power


class Assessment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Verdict(StrEnum):
    DECODED = "decoded"
    UNCERTAIN = "uncertain"
    NO_SIGNAL = "no_signal"


class QualityMetric(StrEnum):
    """Every measure that can become evidence. An enum rather than the string
    literals each producer used to spell out, because _TIERS below decides from
    these names which evidence may reach "decoded" — a typo in either half
    silently promoted detection evidence to decode-grade, with nothing to fail."""

    SYNC_MATCHES = "sync_matches"
    WORD_VALIDITY = "word_validity"
    BURST_MARKS = "burst_marks"
    OFDM_LOCK_RATIO = "ofdm_lock_ratio"
    OFDM_PILOT_SCORE = "ofdm_pilot_score"
    PEAK_DOMINANCE = "peak_dominance"
    SOFT_CONFIDENCE = "soft_confidence"
    SOFT_EYE = "soft_eye"


class Tier(StrEnum):
    """DETECTION proves a signal is PRESENT (energy/preamble events), not that
    the decode is right: a chirp detector firing on real chirps says nothing
    about the symbols decoded after it. Only DECODE positives reach "decoded";
    detection alone stays uncertain."""

    DETECTION = "detection"
    DECODE = "decode"


_TIERS: dict[QualityMetric, Tier] = {
    QualityMetric.SYNC_MATCHES: Tier.DECODE,
    QualityMetric.WORD_VALIDITY: Tier.DECODE,
    # the pilot-lattice score measures the equalizer's consistency on the
    # decoded lattice itself; the CP-correlation ratio below is a peak found
    # BEFORE any symbol decision - this file's own tier rule ("a detector
    # firing says nothing about the symbols decoded after it")
    QualityMetric.OFDM_PILOT_SCORE: Tier.DECODE,
    QualityMetric.PEAK_DOMINANCE: Tier.DECODE,
    QualityMetric.SOFT_CONFIDENCE: Tier.DECODE,
    # a burst mark, a bare demod's per-symbol soft eye, and a CP-timing peak
    QualityMetric.BURST_MARKS: Tier.DETECTION,
    QualityMetric.SOFT_EYE: Tier.DETECTION,
    QualityMetric.OFDM_LOCK_RATIO: Tier.DETECTION,
}

_untiered = sorted(m.value for m in QualityMetric if m not in _TIERS)
if _untiered:
    raise RuntimeError(f"quality metrics with no evidence tier: {_untiered}")


class QualityEvidence(BaseModel):
    source: str
    metric: QualityMetric
    value: float
    assessment: Assessment

    @property
    def tier(self) -> Tier:
        return _TIERS[self.metric]


class QualityReport(BaseModel):
    verdict: Verdict
    evidence: list[QualityEvidence] = []
    rationale: str


def verdict_from(evidence: Sequence[QualityEvidence]) -> tuple[Verdict, str]:
    positives = [e for e in evidence if e.assessment is Assessment.POSITIVE]
    negatives = [e for e in evidence if e.assessment is Assessment.NEGATIVE]
    if not evidence:
        return Verdict.UNCERTAIN, (
            "path produces no checkable evidence (no sync matches beyond "
            "chance, no soft stream, no validating decoder); bits out does "
            "not mean signal in"
        )
    # Only evidence of the same grade can rebut a negative. A DETECTION
    # positive says a signal is PRESENT, which is not a claim about the decode
    # and cannot answer a decode-grade refutation — and marks_evidence emits an
    # unconditional positive for a single burst mark, the one producer here
    # with no null hypothesis. Treating that as a conflict let it veto every
    # producer that has one: word_validity NEGATIVE alone read no_signal, and
    # one burst mark turned the same run uncertain.
    decode_positives = [e for e in positives if e.tier is Tier.DECODE]
    if negatives and decode_positives:
        return Verdict.UNCERTAIN, f"conflicting evidence; negative: {_names(negatives)}"
    if negatives:
        detected = _names([e for e in positives if e.tier is Tier.DETECTION])
        aside = f" (detection-only, does not rebut: {detected})" if detected else ""
        return Verdict.NO_SIGNAL, f"negative evidence: {_names(negatives)}{aside}"
    if not any(e.tier is Tier.DECODE for e in positives):
        return Verdict.UNCERTAIN, (
            f"detection only ({_names(positives)}): a signal is present but "
            "nothing validated the decoded bits; add a sync/validating/soft "
            "stage to confirm"
        )
    return Verdict.DECODED, f"positive evidence: {_names(positives)}"


def _names(evidence: Sequence[QualityEvidence]) -> str:
    return ", ".join(sorted({e.metric.value for e in evidence}))


# A sync count is evidence once it clears the expectation its searcher
# measured (ops_bits._surrogate_chance), by a 5-sigma Poisson margin AND a 3x
# ratio. Both bars are only as good as that expectation: a uniform-bit model
# collapses to ~0 for any long pattern, which is why the searcher measures the
# stream it is about to judge rather than assuming one.
_SYNC_CHANCE_SIGMA = 5.0
_SYNC_CHANCE_RATIO = 3.0
# Finding ZERO matches only means "absent" if the search had somewhere to find
# one. It is not a claim the chance model can make — a long pattern drives the
# expectation toward 0, so 0 hits is what the NULL predicts too — so it is
# bounded by search extent instead: below this many scanned items the reading
# is untestable, not negative. Sized as roughly one frame's worth of bit
# positions, the scale at which a periodic sync would have had to appear.
# Lower and a 2-position search over a stream barely longer than the pattern
# certifies "no_signal" (measured: expected=1e-9, found=0, verdict no_signal);
# higher and a genuinely short capture reads untestable — the safe direction.
_SYNC_NEGATIVE_MIN_ITEMS = 1000

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
# The Chernoff bar is cleared by a SINGLE word once chance is small enough
# (measured: 255 bytes of silence through RS(255,223) was 1/1 valid at chance
# 2.6e-14 and the run read "decoded"), so a positive needs mass the same way
# a negative does.
_WORD_POSITIVE_MIN_WORDS = 8


def _word_excess_significant(valid: int, total: int, chance: float) -> bool:
    """Callers guarantee 0 < chance: a non-positive chance is the absence of a
    null, not a strict one, and flooring it here (an old max(chance, 1e-300))
    manufactured exactly the null that makes any observation look infinitely
    significant."""
    q = valid / total
    p = chance
    if q <= p:
        return False
    if q >= 1.0:
        log_odds = total * math.log(1.0 / p)
    else:
        log_odds = total * (
            q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))
        )
    return log_odds >= _WORD_EXCESS_MIN_LOG_ODDS


def _sync_assessment(found: int, expected: float, scanned: int) -> Assessment | None:
    """None = untestable: the hit count is indistinguishable from chance (found
    something, but no more than random bits would match), the search covered too
    little of the stream for zero hits to mean anything, or there is no chance
    model to judge the hits against at all."""
    if found == 0:
        return (
            Assessment.NEGATIVE
            if scanned >= _SYNC_NEGATIVE_MIN_ITEMS
            else None  # searched too little to call it absent
        )
    if expected <= 0.0:
        # A non-positive expectation is not a very strict null, it is the
        # ABSENCE of one: both bars below multiply it, so 0.0 clears them for
        # any hit count and one match would certify the whole decode. Reachable
        # without a malformed harvest — _chance_valid_rate underflows to exactly
        # 0.0 past ~1075 pattern bits, and _surrogate_chance returns 0 when the
        # rotation matches nowhere, so max() of the two is 0.0. Zero hits are
        # still judged above, on search extent: that reading needs no model.
        return None
    if found <= expected + _SYNC_CHANCE_SIGMA * math.sqrt(expected):
        return None
    if found < _SYNC_CHANCE_RATIO * expected:
        return None
    return Assessment.POSITIVE


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
        # a row that does not report what it scanned cannot support "absent"
        # either, so it reaches _sync_assessment as zero extent; a POSITIVE
        # needs no extent — the hits themselves are the evidence
        assessment = _sync_assessment(
            row.windows_out, row.chance_windows, row.items_in or 0
        )
        if assessment is None:
            continue
        out.append(
            QualityEvidence(
                source=row.block,
                metric=QualityMetric.SYNC_MATCHES,
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
    expected = rows.values(DiagnosticKey.SYNC_CHANCE)
    scanned = rows.counts(DiagnosticKey.SYNC_ITEMS_SCANNED)
    out: list[QualityEvidence] = []
    for block, tags in rows.counts(DiagnosticKey.SYNC_TAGS).items():
        chance = expected.get(block)
        items = scanned.get(block)
        if chance is None:
            # tags without the chance expectation they must be judged
            # against: untestable, not absent.
            continue
        # the coding lane's rule holds here too: a POSITIVE needs no extent
        # (the chance expectation already embeds what was scanned), while a
        # zero-hit reading is judged on extent inside _sync_assessment
        assessment = _sync_assessment(tags, chance, items or 0)
        if assessment is None:
            continue
        out.append(
            QualityEvidence(
                source=block,
                metric=QualityMetric.SYNC_MATCHES,
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
        # 0.0 is reachable, not malformed: _chance_valid_rate underflows to
        # exactly 0.0 past ~1075 redundancy bits, and both bars below divide
        # or multiply by it — the sync lane's _sync_assessment carries the
        # same guard for the same generator.
        if chance is None or not 0.0 < chance <= _WORD_CHANCE_MAX:
            continue
        # A constant corrected codeword is free — the all-zero word satisfies
        # every linear code — so it is excluded from both arms: a dead demod's
        # 400/400 becomes 0/0 (untestable), a padded real decode keeps its
        # positive from the varied words that remain.
        constant = row.words_constant or 0
        valid = row.words_valid - constant
        total = row.words_total - constant
        if total <= 0:
            continue
        ratio = valid / total
        if (
            ratio >= _WORD_VALIDITY_POSITIVE
            and valid >= _WORD_POSITIVE_MIN_WORDS
            and _word_excess_significant(valid, total, chance)
        ):
            assessment = Assessment.POSITIVE
        elif (
            ratio <= max(_WORD_VALIDITY_NEGATIVE, 2.0 * chance)
            and total >= _WORD_NEGATIVE_MIN_WORDS
        ):
            assessment = Assessment.NEGATIVE
        else:
            continue
        out.append(
            QualityEvidence(
                source=row.block,
                metric=QualityMetric.WORD_VALIDITY,
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
            metric=QualityMetric.BURST_MARKS,
            value=float(len(marks)),
            assessment=Assessment.POSITIVE,
        )
    ]


def lock_evidence(diagnostics: Sequence[Diagnostic]) -> list[QualityEvidence]:
    rows = DiagnosticRows(diagnostics)
    out: list[QualityEvidence] = []
    ratio_floors = rows.values(DiagnosticKey.LOCK_MIN)
    for block, best in rows.values(DiagnosticKey.LOCK_RATIO_BEST).items():
        floor = ratio_floors.get(block)
        if floor is None:
            # a best-ratio row without its block's configured floor is a
            # malformed pair: untestable, never judged against a default
            continue
        if best < floor:
            # detection grade cannot assert absence: a non-OFDM transmission
            # fails a CP search without being absent (the soft_eye rule)
            continue
        out.append(
            QualityEvidence(
                source=block,
                metric=QualityMetric.OFDM_LOCK_RATIO,
                value=best,
                assessment=Assessment.POSITIVE,
            )
        )
    score_floors = rows.values(DiagnosticKey.LOCK_SCORE_MIN)
    for block, best in rows.values(DiagnosticKey.LOCK_SCORE_BEST).items():
        floor = score_floors.get(block)
        if floor is None:
            continue
        out.append(
            QualityEvidence(
                source=block,
                metric=QualityMetric.OFDM_PILOT_SCORE,
                value=best,
                assessment=(
                    Assessment.POSITIVE if best >= floor else Assessment.NEGATIVE
                ),
            )
        )
    return out


# The CSS analog of soft_confidence: the argmax over the dechirped spectrum IS
# the symbol decision, so peak/median of that vector measures how decisively
# each one was made. The deciding block owns the calibrated tallies and the
# chance ceiling (decision.py); here only the run's fractions are judged,
# word_validity-style. Like soft_confidence it attests decision quality, not
# symbol identity. Bars straddled by test_verdict_constants; the honest limits
# are pinned in the css quality tests.
_DOMINANCE_POSITIVE = 0.5
_DOMINANCE_NEGATIVE = 0.05


def dominance_evidence(diagnostics: Sequence[Diagnostic]) -> list[QualityEvidence]:
    rows = DiagnosticRows(diagnostics)
    totals = rows.counts(DiagnosticKey.SYMBOLS_TOTAL)
    chances = rows.values(DiagnosticKey.DOMINANCE_CHANCE)
    out: list[QualityEvidence] = []
    for block, dominant in rows.counts(DiagnosticKey.DOMINANT_SYMBOLS).items():
        total = totals.get(block)
        if not total:
            # the decider never saw a symbol: untestable, not absent
            continue
        chance = chances.get(block)
        if chance is None:
            # counts without the chance ceiling they are judged against:
            # untestable, not absent. The block that owns the ceiling is the
            # only place it is calibrated, so a copy here could only go stale.
            continue
        frac = dominant / total
        if frac >= _DOMINANCE_POSITIVE and _word_excess_significant(
            dominant, total, chance
        ):
            assessment = Assessment.POSITIVE
        elif frac <= _DOMINANCE_NEGATIVE and total >= _WORD_NEGATIVE_MIN_WORDS:
            assessment = Assessment.NEGATIVE
        else:
            continue
        out.append(
            QualityEvidence(
                source=block,
                metric=QualityMetric.PEAK_DOMINANCE,
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

# Separation above which a multi-level fit is real signal rather than a noise
# blob. Straddled by test_verdict_constants, which is where the calibration
# lives now — restating measurements here left the comment as the only record
# of an experiment nothing re-ran.
_SOFT_MULTILEVEL_SEPARATION = 4.0

# |x| mean/std bars on a demodulated stream. The ratio alone cannot reject a
# wrong-rate decode, so a positive also demands whitened decisions via
# consecutive-sign correlation. Two limits worth knowing, both pinned in
# test_verdict_constants: structured unscrambled data suppresses the positive
# (conservative — the stream reads uncertain and sync/validity evidence still
# applies), and undersampling emits genuinely clean decisions of aliased bits,
# which no stream statistic can see. Soft confidence attests decision quality,
# never bit identity.
_SOFT_POSITIVE = 2.0
_SOFT_NEGATIVE = 1.45
_SOFT_MIN_POLARITY_FRACTION = 0.02
_SOFT_MAX_SIGN_CORR = 0.15
# Whiteness is judged on the worst sign correlation across lags 1..4, not lag
# 1 alone: ++-- repeats have lag-1 correlation exactly 0.0, so a one-lag guard
# certified every even-period repeat up to the same bar it was built to hold.
_SOFT_WHITENESS_LAGS = 4

# A stream of decisions is not a stream of confidences. A demod saturating on
# noise emits constant-magnitude items, which scored mean|x|/std|x| = inf —
# MAXIMUM confidence as the signature of the failure mode — and random signs
# are white at every lag, so no downstream guard could object. Two gates, one
# per flavor of degeneracy: a handful of distinct values is a slicer/limiter
# output (a genuine float32 LLR stream is ~all-distinct), and a magnitude
# spread below 1e-3 of the mean is saturation (measured healthy decodes sit at
# 6e-2 relative spread — 40 dB FSK scored ratio 15.8 — a 60x margin, while a
# limiter's jitter sits at float precision, ~1e-7).
_SOFT_MIN_DISTINCT = 64
_SOFT_MIN_DISPERSION = 1e-3

_SOFT_DISCRETE_CAVEAT = (
    "soft stream carries only {n} distinct values: these are decisions, not "
    "per-item confidences, so they can neither certify nor reject the decode "
    "— tap a soft (LLR) stage output instead of a slicer/limiter"
)
_SOFT_SATURATED_CAVEAT = (
    "soft stream magnitudes are constant to {rel:.1e} relative spread: a "
    "saturated or hard-limited stream carries no confidence information — "
    "remove the limiter or reduce gain ahead of the demod"
)

# A soft stream carrying non-finite items is not the stream it claims to be, so
# it scores nothing at all rather than scoring the survivors. run_rx rejects a
# non-finite INPUT capture outright, so a NaN/Inf reaching here was manufactured
# by the path itself — measured: agc mode='power' divides by the rms of an
# exactly-zero span. Dropping them silently and judging what was left certified
# "decoded" over 1311 real items of a 65536-item stream that was 98% NaN, with
# nothing on the wire saying so. No tolerance band: a healthy path emits none,
# which the whole suite passing at this bar is the measurement for.
_NON_FINITE_CAVEAT = (
    "soft stream unscorable: {bad} of {total} sampled items are non-finite "
    "({pct:.1f}%). run_rx refuses a non-finite capture, so these were produced "
    "by the path — check for an agc over a zero-power span, or a stage dividing "
    "by a level it never established"
)

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
    total = path.stat().st_size // ItemType.F.item_bytes
    with path.open("rb") as f:
        if total <= _SOFT_SAMPLE_ITEMS:
            return np.fromfile(f, dtype=np.float32)
        nblocks = -(-total // _SOFT_SAMPLE_ITEMS)
        powers = np.zeros(nblocks, dtype=np.float64)
        for i in range(nblocks):
            check_deadline()
            block = np.fromfile(f, dtype=np.float32, count=_SOFT_SAMPLE_ITEMS)
            if block.size:
                powers[i] = float(np.mean(block.astype(np.float64) ** 2))
        start = min(
            int(np.argmax(powers)) * _SOFT_SAMPLE_ITEMS, total - _SOFT_SAMPLE_ITEMS
        )
        f.seek(start * ItemType.F.item_bytes)
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
    # zero-padded edges, top-decile reference: survey's two burst gates reflect
    # their edges and reference the top decile's MEDIAN. Three gates, one
    # smoothing (levelfit.windowed_power), three deliberately different bars.
    power = windowed_power(x, _SOFT_ACTIVE_WINDOW, mode="constant")
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
    if assessment is Assessment.NEGATIVE and not decode_grade:
        return []
    return [
        QualityEvidence(
            source="soft_stream",
            metric=(
                QualityMetric.SOFT_CONFIDENCE
                if decode_grade
                else QualityMetric.SOFT_EYE
            ),
            value=value,
            assessment=assessment,
        )
    ]


def soft_evidence(
    path: Path | None, *, decode_grade: bool = True
) -> list[QualityEvidence]:
    return _soft_reading(path, decode_grade=decode_grade)[0]


def _soft_reading(
    path: Path | None, *, decode_grade: bool = True
) -> tuple[list[QualityEvidence], str | None]:
    """The soft evidence, and a caveat naming what had to be discarded to get
    it. Sampled once: the caveat cannot be a second pass over the file."""
    if path is None or not path.is_file():
        return [], None
    x = _sample_soft(path)
    finite = np.isfinite(x)
    n_bad = int(x.size - int(finite.sum()))
    if n_bad:
        return [], _NON_FINITE_CAVEAT.format(
            bad=n_bad, total=int(x.size), pct=100.0 * n_bad / max(x.size, 1)
        )
    x = x[finite]
    # An exact 0.0 is an erasure, not a measurement: depuncture scatters them
    # from its null_source and a hard gate emits them over silence. Scoring
    # them INVERTED the verdict on punctured paths — the erasure spike at 0
    # tightened a k=8 levelfit on pure noise past the separation bar
    # ("decoded"), while the same zeros dragged a genuine 10 dB decode's
    # magnitude ratio under the negative bar ("no_signal").
    x = x[x != 0.0]
    if x.size < _SOFT_MIN_ITEMS:
        return [], None
    active = _active_mask(x)
    if not active.any():
        return _emit_soft(0.0, Assessment.NEGATIVE, decode_grade), None
    xa = x[active]
    if xa.size < _SOFT_MIN_ITEMS:
        return [], None
    distinct = int(np.unique(xa).size)
    if distinct < _SOFT_MIN_DISTINCT:
        return [], _SOFT_DISCRETE_CAVEAT.format(n=distinct)
    mag = np.abs(xa)
    spread = float(mag.std())
    mean = float(mag.mean())
    if spread <= _SOFT_MIN_DISPERSION * mean:
        return [], _SOFT_SATURATED_CAVEAT.format(rel=spread / mean if mean else 0.0)
    # A constant or one-sided LLR stream (a CW carrier) can score an
    # arbitrarily high magnitude ratio without carrying any modulation; both
    # polarities must show up in real proportion before that ratio counts.
    both_polarities = (
        min(float((xa > 0).mean()), float((xa < 0).mean()))
        >= _SOFT_MIN_POLARITY_FRACTION
    )
    signs = np.sign(x)
    corrs = []
    for lag in range(1, _SOFT_WHITENESS_LAGS + 1):
        pair = active[lag:] & active[:-lag]
        if pair.any():
            corrs.append(abs(float(np.mean((signs[lag:] * signs[:-lag])[pair]))))
    whitened = max(corrs, default=0.0) <= _SOFT_MAX_SIGN_CORR
    fit = fit_levels(xa)
    # The multilevel branch exists because a clean M-ary eye's magnitude ratio
    # sits BELOW the binary negative bar (measured 0.998 on clean CPFSK), so
    # it must run before the ratio arms — but it earns no exemption from the
    # polarity and whiteness gates, which is exactly how an all-positive
    # ladder and a stuck alternator read "decoded" through it.
    if (
        fit.order > 2
        and fit.separation >= _SOFT_MULTILEVEL_SEPARATION
        and both_polarities
        and whitened
    ):
        return (
            _emit_soft(
                float(min(fit.separation, _SOFT_VALUE_CAP)),
                Assessment.POSITIVE,
                decode_grade,
            ),
            None,
        )
    ratio = mean / spread
    if ratio >= _SOFT_POSITIVE and both_polarities and whitened:
        assessment = Assessment.POSITIVE
    elif ratio <= _SOFT_NEGATIVE:
        assessment = Assessment.NEGATIVE
    else:
        return [], None
    return (
        _emit_soft(float(min(ratio, _SOFT_VALUE_CAP)), assessment, decode_grade),
        None,
    )


def assess_quality(
    *,
    registry: Mapping[str, Stage[Any, Any]],
    census: Sequence[BlockCensus],
    diagnostics: Sequence[Diagnostic],
    marks: Sequence[int],
    soft_stream: Path | None,
    soft_decode_grade: bool = True,
) -> QualityReport:
    soft, caveat = _soft_reading(soft_stream, decode_grade=soft_decode_grade)
    # A mark-replacing search stage (sync_symbols) makes the run's marks its
    # own pattern hits: judged with no chance model they are not evidence of
    # anything, and marks_evidence is the one extractor without a null - an
    # alternating pattern on an alternating stream read "burst_marks
    # positive" while the stage's description promises it earns no credit.
    searched = any(
        getattr(registry.get(row.kind), "marks_are_search_hits", False)
        for row in census
    )
    evidence = (
        sync_evidence(census, registry)
        + tag_sync_evidence(diagnostics)
        + survival_evidence(census, registry)
        + ([] if searched else marks_evidence(marks))
        + lock_evidence(diagnostics)
        + dominance_evidence(diagnostics)
        + soft
    )
    verdict, rationale = verdict_from(evidence)
    if caveat is not None:
        # rides the rationale the tool docstring already sends the agent to,
        # rather than a new wire field every clean run would pay tokens for.
        # Carried whatever the verdict: independent evidence may still support a
        # decode, and the operator needs to know their stream was poisoned
        # either way. When it is the whole story it REPLACES the no-evidence
        # text, which says "no soft stream" — there is one, and saying both
        # sends the reader looking for a stream the next sentence describes.
        rationale = f"{rationale}. {caveat}" if evidence else caveat
    return QualityReport(verdict=verdict, evidence=evidence, rationale=rationale)
