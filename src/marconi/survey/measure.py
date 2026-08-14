from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis

from marconi.deadline import check_deadline
from marconi.engine.types.bounds import check_sample_rate
from marconi.levelfit import (
    Tail,
    binned_counts,
    fit_levels,
    percentile_span,
    windowed_power,
)
from marconi.survey.iqfile import iter_iq, iter_probes, sample_iq
from marconi.wire import Payload, replace

_SURVEY_NPERSEG = 4096
# Agent-facing arrays stay small: a coarse PSD conveys the spectral shape, and
# the frequency axis is a uniform ramp reconstructed from start+step, never
# shipped as hundreds of derivable floats.
_SPECTRUM_BINS = 64
_SURVEY_ACTIVE_FRACTION = 0.3
_SURVEY_BURST_WINDOW = 64


class EnvelopeStats(Payload):
    const_envelope_ratio: float
    amplitude_kurtosis: float
    mean_amplitude: float
    std_amplitude: float


class SpectrumStats(Payload):
    psd_db: list[float]
    freq_start_hz: float
    freq_step_hz: float
    center_offset_hz: float
    peak_offset_hz: float
    occupied_bw_hz: float
    occupied_lo_hz: float
    occupied_hi_hz: float


class PhaseConcentration(Payload):
    order_2: float
    order_4: float
    order_8: float


class CarrierStats(Payload):
    offset_hz: float
    psk_order: int | None
    phase_concentration: PhaseConcentration
    method: Literal["mpsk", "spectral_centroid"]
    off_center: bool
    offset_ambiguous: bool


# M-th-power phase-fold analysis: raising phase-only z to the m-th power
# collapses an order-m alphabet to a spectral line at m*offset. See the survey()
# MCP docstring for the full rationale (the jump rule, the QAM confound, order 2).
_MPSK_ORDERS = (2, 4, 8)
_MPSK_CONC_THRESH = 25.0
_MPSK_JUMP = 10.0
# Centroid distance from a dealiased candidate (in units of sample_rate / m)
# beyond which the offset is too ambiguous to disambiguate — fallback to centroid.
_DEALIAS_AMBIGUOUS_FRAC = 0.35
# A staggered quaternary alphabet (pi/4-shifted QPSK) folds clean at 8, but its
# 4-fold power alternates between the two component grids: the order-4 line is
# displaced by half the symbol rate, so a naive order-4 claim reports an offset
# biased by Rs/8 with "mpsk" confidence. When the 8-fold line is independently
# strong, the two implied offsets must agree; a displaced 4-line hands the
# claim to order 8 and its true offset. Genuine QPSK/QAM agree within noise —
# the tolerance is a few FFT bins.
_STAGGER_TOL_BINS = 6.0
# Off the analyzed band's centre by a meaningful fraction of the occupied
# bandwidth: the default demod (carrier at DC) will miss, so surface it.
_OFF_CENTER_FRAC = 0.15
# Occupied span reaching past this fraction of the rate means the band wraps
# +-fs/2: every coarse anchor is meaningless there, so the offset is flagged
# ambiguous instead of dealiased confidently wrong (measured: +499 kHz
# reported as -999.9 Hz with every confidence flag clear).
_BAND_EDGE_FRAC = 0.45
# Default-band exclusion of the burst-repetition comb: the comb's measured
# fundamental and its first few harmonics outrank a real symbol line, so the
# default floor clears 4 cadence harmonics. 4 keeps a genuine symbol rate as
# low as 4x the frame rate searchable, which no framed protocol sits under.
_CADENCE_FLOOR_HARMONICS = 4.0
# Cap the record length for carrier M-th-power FFT analysis.
_CARRIER_MAX_SAMPLES = 1 << 20


def _downsample(a: npt.NDArray[np.float64], cap: int) -> npt.NDArray[np.float64]:
    if a.size <= cap:
        return a
    groups = -(-a.size // cap)
    trimmed = a[: (a.size // groups) * groups]
    return trimmed.reshape(-1, groups).mean(axis=1)


def _interp_peak_hz(
    p: npt.NDArray[np.floating[Any]], peak: int, sample_rate: float
) -> float:
    n = p.size
    a, b, c = float(p[(peak - 1) % n]), float(p[peak]), float(p[(peak + 1) % n])
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    freq = float(np.fft.fftfreq(n, d=1.0 / sample_rate)[peak])
    return freq + delta * (sample_rate / n)


def _dealias_offset(
    line_hz: float, m: int, sample_rate: float, coarse_hz: float
) -> float:
    base = line_hz / m
    step = sample_rate / m
    k = round((coarse_hz - base) / step)
    return base + k * step


def _sig(x: float, figs: int = 4) -> float:
    # Round a scale-spanning dimensionless value to `figs` significant figures:
    # keeps a small std and an O(1) ratio equally readable, and never rounds a
    # positive to zero (unlike a fixed decimal place). Hz values round to 0.1
    # instead — an absolute grid the agent matches candidates against.
    if not np.isfinite(x) or x == 0.0:
        return float(x)
    return float(round(x, figs - 1 - int(np.floor(np.log10(abs(x))))))


def _spectrum(x: npt.NDArray[np.complex64], sample_rate: float) -> SpectrumStats:
    nperseg = int(min(_SURVEY_NPERSEG, x.size))
    f, pxx = welch(
        x, fs=sample_rate, nperseg=nperseg, return_onesided=False, detrend=False
    )
    order = np.argsort(f)
    f, pxx = f[order], pxx[order]
    # Moments over the FLOOR-SUBTRACTED PSD: raw moments include the noise
    # floor, which dominates whenever the signal is narrow relative to the
    # capture - measured on a QPSK occupying 6.7% of the band, the centroid
    # drifted from 200 kHz (true) to 80 kHz as SNR fell to 10 dB and
    # occupied_bw inflated from 59 kHz to 983 kHz (pure noise reads 990),
    # which both vetoed the correct mpsk offset AND suppressed off_center.
    # The median of a mostly-noise PSD IS the floor; for a band-filling
    # signal the subtraction costs a constant that cancels in the ratio.
    excess = np.maximum(pxx - float(np.median(pxx)), 0.0)
    total = float(excess.sum()) or 1.0
    centroid = float((f * excess).sum() / total)
    peak = float(f[int(np.argmax(pxx))])
    cum = np.cumsum(excess) / total
    lo = float(f[min(int(np.searchsorted(cum, 0.005)), f.size - 1)])
    hi = float(f[min(int(np.searchsorted(cum, 0.995)), f.size - 1)])
    fd = _downsample(f, _SPECTRUM_BINS)
    pd = 10.0 * np.log10(np.maximum(_downsample(pxx, _SPECTRUM_BINS), 1e-20))
    step = float(fd[1] - fd[0]) if fd.size > 1 else 0.0
    return SpectrumStats(
        psd_db=[round(float(v), 1) for v in pd],
        freq_start_hz=round(float(fd[0]), 1) if fd.size else 0.0,
        freq_step_hz=round(step, 1),
        center_offset_hz=round(centroid, 1),
        peak_offset_hz=round(peak, 1),
        occupied_bw_hz=round(hi - lo, 1),
        occupied_lo_hz=round(lo, 1),
        occupied_hi_hz=round(hi, 1),
    )


def _bounded(x: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    return x[:_CARRIER_MAX_SAMPLES] if x.size > _CARRIER_MAX_SAMPLES else x


def _mpsk_line(
    z: npt.NDArray[np.complexfloating[Any, Any]], sample_rate: float, m: int, dof: int
) -> tuple[float, float]:
    """(score, line_hz) for the m-th-power carrier line of phase-only z. The
    line sits at m x the carrier offset (at DC for a zero-offset carrier, so DC
    is never special-cased); score is the dof-normalized peak-to-mean of the
    x^m power spectrum. line_hz is the raw interpolated frequency, not divided."""
    spec = np.fft.fft(z**m)
    p = spec.real**2 + spec.imag**2
    peak = int(np.argmax(p))
    mean = float(p.mean()) or 1e-20
    score = (float(p[peak]) / mean) / np.log(max(dof, 2))
    return score, _interp_peak_hz(p, peak, sample_rate)


def _fold(
    variant: npt.NDArray[np.complexfloating[Any, Any]],
    active: npt.NDArray[np.bool_],
    sample_rate: float,
    dof: int,
) -> tuple[dict[int, float], dict[int, float]]:
    z = (variant / np.maximum(np.abs(variant), 1e-12)) * active
    scores: dict[int, float] = {}
    lines: dict[int, float] = {}
    for m in _MPSK_ORDERS:
        scores[m], lines[m] = _mpsk_line(z, sample_rate, m, dof)
    return scores, lines


def _claimed_order(scores: dict[int, float]) -> int | None:
    return next(
        (
            m
            for m in (4, 8)
            if scores[m] >= _MPSK_CONC_THRESH
            and scores[m] >= _MPSK_JUMP * scores[m // 2]
        ),
        None,
    )


def _variant_order(
    scores: dict[int, float],
    lines: dict[int, float],
    sample_rate: float,
    anchor_hz: float,
    n_samples: int,
) -> int | None:
    order = _claimed_order(scores)
    if order == 4 and scores[8] >= _MPSK_CONC_THRESH:
        off4 = _dealias_offset(lines[4], 4, sample_rate, anchor_hz)
        off8 = _dealias_offset(lines[8], 8, sample_rate, anchor_hz)
        if abs(off8 - off4) > _STAGGER_TOL_BINS * sample_rate / n_samples:
            order = 8
    return order


def _carrier(
    x: npt.NDArray[np.complex64],
    sample_rate: float,
    occupied_bw_hz: float,
    anchor_hz: float,
    band_edge_wrap: bool = False,
) -> CarrierStats:
    xb = _bounded(x)
    active = _burst_power_gate(xb, _SURVEY_ACTIVE_FRACTION, _SURVEY_BURST_WINDOW)
    dof = int(active.sum()) or xb.size
    # Two fold hypotheses, decided coherently PER VARIANT (mixing per-order
    # maxima broke the jump rule: an unsubtracted DC spike inflates order 2
    # and vetoes a genuine order-4 claim). The DC-subtracted variant decides
    # first - the spur case (LO leakage, universal on RTL-SDR) - and the raw
    # variant only speaks when the subtracted one claims nothing, which is
    # the carrier-tuned-to-EXACTLY-0-Hz case the subtraction annihilated
    # (measured: phase_concentration 1.0 at 0 Hz offset, 75,000 without the
    # subtraction, identical at 100 Hz). Reported concentrations are the
    # per-order max so signal-at-DC stays visible either way.
    xd = xb - _active_mean(xb, active)
    scores, lines = _fold(xd, active, sample_rate, dof)
    raw_scores, raw_lines = _fold(xb, active, sample_rate, dof)
    order = _variant_order(scores, lines, sample_rate, anchor_hz, xb.size)
    if order is None:
        order = _variant_order(raw_scores, raw_lines, sample_rate, anchor_hz, xb.size)
        if order is not None:
            lines = raw_lines
    # The per-order max is REPORT-ONLY, computed after every decision: commit
    # 27eee43 merged it above the stagger gate, and the raw variant's spur-fed
    # 8-line armed a gate whose lines were the subtracted variant's — measured
    # (QPSK +50 kHz, SNR +3 dB, constant spur): order 8 at a noise line's
    # offset with offset_ambiguous False.
    scores = {m: max(scores[m], raw_scores[m]) for m in _MPSK_ORDERS}
    # a signal whose occupied band wraps +-fs/2 has no trustworthy coarse
    # anchor at all: dealiasing against one reported +499 kHz as -999.9 Hz
    # with every confidence flag clear
    ambiguous = band_edge_wrap
    if order is not None and not ambiguous:
        offset = _dealias_offset(lines[order], order, sample_rate, anchor_hz)
        if abs(anchor_hz - offset) / (sample_rate / order) > _DEALIAS_AMBIGUOUS_FRAC:
            offset, ambiguous = anchor_hz, True
    else:
        offset = anchor_hz
    method: Literal["mpsk", "spectral_centroid"] = (
        "mpsk" if order is not None and not ambiguous else "spectral_centroid"
    )
    off_center = occupied_bw_hz > 0 and abs(offset) > _OFF_CENTER_FRAC * occupied_bw_hz
    return CarrierStats(
        offset_hz=round(offset, 1),
        psk_order=order,
        phase_concentration=PhaseConcentration(
            order_2=_sig(scores[2]), order_4=_sig(scores[4]), order_8=_sig(scores[8])
        ),
        method=method,
        off_center=off_center,
        offset_ambiguous=ambiguous,
    )


def _envelope(x: npt.NDArray[np.complex64]) -> EnvelopeStats:
    active = _burst_power_gate(x, _SURVEY_ACTIVE_FRACTION, _SURVEY_BURST_WINDOW)
    a = np.abs(_gate(x, active)).astype(np.float64)
    mean = float(a.mean())
    std = float(a.std())
    kurt = float(kurtosis(a)) if std > 0 else 0.0
    return EnvelopeStats(
        const_envelope_ratio=_sig(std / mean) if mean > 0 else 0.0,
        amplitude_kurtosis=_sig(kurt),
        mean_amplitude=_sig(mean),
        std_amplitude=_sig(std),
    )


_SURVEY_IFREQ_BINS = 65
_SURVEY_RATE_K = 5
_SURVEY_CLOCK_CHUNKS = 16
_SURVEY_RATE_FLOOR_BINS = 3.0
# Burst/TDMA cadence fundamentals are hunted below this ratio of the sample
# rate even when the resolution-derived search floor reaches further down: the
# comb damping must keep seeing slot-cadence fundamentals as fundamentals, not
# as rate candidates crowding the pool.
_SURVEY_CADENCE_HUNT_RATIO = 1000.0
_SURVEY_RATE_POOL_K = 2 * _SURVEY_RATE_K
_SURVEY_FUNDAMENTAL_POOL = 20
_SURVEY_COMB_ORDER_CAP = 20
_SURVEY_COMB_MIN_ORDER = 2
_SURVEY_COMB_REL_TOL = 0.02
# A hunted fundamental must span comfortably more than the +/-2-bin match
# tolerance below, or adjacent comb orders overlap and the mask saturates —
# a ~3-bin "fundamental" reads every peak in the band as its harmonic.
_SURVEY_COMB_FLOOR_BINS = 5.0
_SURVEY_COMB_TOL_FLOOR_BINS = 2.0
_SURVEY_COMB_MAJORITY = 0.5
_SURVEY_COMB_DAMPING = 0.05
_EYE_MIN_SYMBOLS = 64
_EYE_MAX_SYMBOLS = 2000
_EYE_PHASE_STEPS = 8
# The old note here — "wrong-rate smear tops out ~3.6" — is REFUTED: on
# constant-envelope FM the eye at ANY rate samples the modulation's bimodal
# instantaneous frequency, so it measures level structure, not rate
# correctness, and rises with SNR (2FSK h=0.7 sps 8, 5 seeds/point: noise-rate
# lines read 6.9-7.2 at SNR 20 and 10.1-11.3 at SNR 25). 8.0 still earns its
# keep as the SNR gate: at SNR <= 15 every line's eye — true included — reads
# <= 5.8 and promotion must not fire; at SNR 20 the true line reads 8.1-8.4.
# What excludes the noise lines is _EYE_STRENGTH_FRAC below, not this bar.
_CLEAR_EYE = 8.0
# Harmonics of the true rate carry cleaner-looking eyes than the fundamental
# (2R read ~1.2x the true line's eye at every SNR in the sweep: 10.2 vs 8.3,
# 18.1 vs 14.7), so the promotion pool keeps every eligible line whose eye is
# within this fraction of the best and takes the LOWEST rate. Raising it past
# the measured true/2R ratio ~0.8 would hand the fundamental's crown to its
# own harmonic.
_HARMONIC_FRAC = 0.6
# Eye-based promotion is restricted to lines carrying real cyclostationary
# strength: in the same sweep every true/harmonic line measured >= 0.76 of the
# max strength while noise lines measured 0.028-0.050 wherever any eye
# cleared _CLEAR_EYE (their 0.26-0.30 readings occur only at SNR <= 10, where
# every eye is <= 3.6). 0.2 sits 4x over the dangerous-regime noise and keeps
# the rescue of a true line down-ranked to 0.2x by a decoy (the strongest
# measured decoy left the true line at 0.92x).
_EYE_STRENGTH_FRAC = 0.2


class InstFreqStats(Payload):
    hist_counts: list[int]
    hist_start_hz: float
    hist_step_hz: float
    peaks_hz: list[float]
    spread_hz: float


class SymbolRateStats(Payload):
    candidates_hz: list[float]
    strengths: list[float]
    eye_openness: list[float]
    eye_confirmed: bool
    search_lo_hz: float
    search_hi_hz: float
    clock_resolution_hz: float


def _clock_spectrum(
    sig: npt.NDArray[np.float32], sample_rate: float, chunks: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    per = sig.size // chunks
    if per < 2:
        per, chunks = sig.size, 1
    seg = sig[: per * chunks].reshape(chunks, per).astype(np.float64)
    seg = seg - seg.mean(axis=1, keepdims=True)
    mag = np.abs(np.fft.rfft(seg, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(per, d=1.0 / sample_rate)
    return freqs, mag


def _peaks_in_band(
    freqs: npt.NDArray[np.floating[Any]],
    mag: npt.NDArray[np.floating[Any]],
    lo: float,
    hi: float,
    k: int | None,
) -> tuple[npt.NDArray[np.floating[Any]], npt.NDArray[np.floating[Any]]]:
    band = (freqs >= lo) & (freqs <= hi)
    fb, mb = freqs[band], mag[band]
    if mb.size == 0:
        return np.array([]), np.array([])
    idx, _ = find_peaks(mb)
    if idx.size == 0:
        return np.array([]), np.array([])
    top = idx[np.argsort(mb[idx])[::-1][:k]]
    return fb[top], mb[top]


# Survey has TWO "active" definitions and they are not interchangeable, so
# each is named for what it measures rather than both being called active.
# _magnitude_gate is per-sample against the median magnitude: it keeps the
# sample-to-sample structure a clock line lives in, which is why the
# symbol-rate search uses it. _burst_power_gate is windowed power against the
# top-decile level: it cuts whole idle spans, which is what an amplitude or
# tone statistic needs and what would destroy a clock line. Changing which
# block uses which changes the measurement, not just the sample count.
def _magnitude_gate(
    x: npt.NDArray[np.complex64], fraction: float
) -> npt.NDArray[np.bool_]:
    mag = np.abs(x)
    med = float(np.median(mag))
    if med <= 0.0:
        return np.ones(x.size, dtype=bool)
    return mag > fraction * med


def _magnitude_gate_pairs(
    x: npt.NDArray[np.complex64], fraction: float
) -> npt.NDArray[np.bool_]:
    active = _magnitude_gate(x, fraction)
    return active[1:] & active[:-1]


def _burst_power_gate(
    x: npt.NDArray[np.complex64], fraction: float, window: int
) -> npt.NDArray[np.bool_]:
    power = windowed_power(x, window)
    # reference the top-decile median, not the max: a single hot interferer
    # burst inside the span would otherwise set the threshold and gate the
    # actual signal of interest out of the mask entirely
    p90 = float(np.percentile(power, 90))
    top = power[power > p90]
    peak = float(np.median(top)) if top.size else p90
    if peak <= 0.0:
        return np.ones(x.size, dtype=bool)
    return power > fraction * peak


def _burst_power_gate_pairs(
    x: npt.NDArray[np.complex64], fraction: float, window: int
) -> npt.NDArray[np.bool_]:
    active = _burst_power_gate(x, fraction, window)
    return active[1:] & active[:-1]


_G = TypeVar("_G", bound=np.generic)


def _gate(values: npt.NDArray[_G], keep: npt.NDArray[np.bool_]) -> npt.NDArray[_G]:
    gated = values[keep]
    return gated if gated.size else values


def _active_mean(
    x: npt.NDArray[np.complex64], active: npt.NDArray[np.bool_]
) -> complex:
    sel = x[active]
    return complex(sel.mean()) if sel.size else 0j


def _comb_harmonic_mask(
    freqs: npt.NDArray[np.floating[Any]], fundamental: float, bin_hz: float
) -> npt.NDArray[np.bool_]:
    orders: npt.NDArray[np.float64] = np.round(freqs / fundamental)
    tol: npt.NDArray[np.floating[Any]] = np.maximum(
        _SURVEY_COMB_REL_TOL * freqs, _SURVEY_COMB_TOL_FLOOR_BINS * bin_hz
    )
    in_range = (orders >= _SURVEY_COMB_MIN_ORDER) & (orders <= _SURVEY_COMB_ORDER_CAP)
    return in_range & (np.abs(freqs - orders * fundamental) < tol)


def _fundamental_below(
    freqs: npt.NDArray[np.floating[Any]],
    mag: npt.NDArray[np.floating[Any]],
    lo: float,
    targets: npt.NDArray[np.floating[Any]],
    bin_hz: float,
) -> float | None:
    floor = _SURVEY_COMB_FLOOR_BINS * bin_hz
    if floor >= lo or targets.size == 0:
        return None
    candidates, _ = _peaks_in_band(freqs, mag, floor, lo, _SURVEY_FUNDAMENTAL_POOL)
    if candidates.size == 0:
        return None
    required = _SURVEY_COMB_MAJORITY * targets.size
    best_g, best_matches = None, 0
    for g in candidates:
        n = int(_comb_harmonic_mask(targets, float(g), bin_hz).sum())
        if n > required and n > best_matches:
            best_g, best_matches = float(g), n
    return best_g


def _damp_harmonics(
    freqs: npt.NDArray[np.floating[Any]],
    strengths: npt.NDArray[np.floating[Any]],
    fundamental: float | None,
    bin_hz: float,
) -> npt.NDArray[np.floating[Any]]:
    if fundamental is None or freqs.size == 0:
        return strengths
    mask = _comb_harmonic_mask(freqs, fundamental, bin_hz)
    return np.where(mask, strengths * _SURVEY_COMB_DAMPING, strengths)


def _eye_cleared(eyes: list[float]) -> bool:
    return bool(eyes) and max(eyes) >= _CLEAR_EYE


@dataclass(frozen=True)
class RateCandidate:
    """One symbol-rate hypothesis with the two measurements that rank it.
    Kept as one object so the three cannot drift out of correspondence."""

    rate_hz: float
    strength: float
    eye: float


def _rerank_by_eye(
    candidates: list[RateCandidate],
) -> tuple[list[RateCandidate], bool]:
    """Ranked candidates plus the eye_confirmed flag, one decision: True means
    the head of the list was promoted by its eye AND its strength — when False,
    candidates_hz is strength-ranked only and untrustworthy for
    constant-envelope or pulse-shaped (e.g. GMSK) signals."""
    if not candidates:
        return candidates, False
    strength_bar = _EYE_STRENGTH_FRAC * max(c.strength for c in candidates)
    eligible = [c for c in candidates if c.strength >= strength_bar]
    if not _eye_cleared([c.eye for c in eligible]):
        return candidates, False
    thresh = _HARMONIC_FRAC * max(c.eye for c in eligible)
    strong = [c for c in eligible if c.eye >= thresh]
    primary = min(strong, key=lambda c: c.rate_hz)
    rest = sorted(
        (c for c in candidates if c is not primary), key=lambda c: c.eye, reverse=True
    )
    return [primary, *rest], True


def _symbol_rate(
    x: npt.NDArray[np.complex64], sample_rate: float, lo: float, hi: float
) -> SymbolRateStats:
    ya = np.abs(np.diff(np.abs(x)))
    dphi = np.angle(x[1:] * np.conj(x[:-1]))
    yf_full = np.abs(np.diff(dphi))

    active_pairs = _magnitude_gate_pairs(x, _SURVEY_ACTIVE_FRACTION)
    # ZEROED, never deleted: _gate compresses the record, and _clock_spectrum
    # then measured the compressed array against the UNCOMPRESSED sample rate
    # - a periodicity of P samples reported at f/rho (measured +4.5..9.3%
    # high, 20-104 clock bins off a docstring that promises one). Zeros keep
    # the time base; the per-chunk mean subtraction absorbs their DC.
    yf = np.where(active_pairs[1:] & active_pairs[:-1], yf_full, 0.0)

    fa, ma = _clock_spectrum(ya, sample_rate, _SURVEY_CLOCK_CHUNKS)
    fg, mg = _clock_spectrum(yf, sample_rate, _SURVEY_CLOCK_CHUNKS)
    norm = max(float(ma.max(initial=0.0)), float(mg.max(initial=0.0)), 1e-20)
    # every in-band peak enters; ranking happens on DAMPED strength (a raw-
    # magnitude pool cut first would let comb harmonics — about to be damped
    # to nothing — crowd real low-frequency lines out of the pool)
    ca, sa = _peaks_in_band(fa, ma, lo, hi, None)
    cg, sg = _peaks_in_band(fg, mg, lo, hi, None)

    dfa = fa[1] - fa[0] if fa.size > 1 else 1.0
    dfg = fg[1] - fg[0] if fg.size > 1 else 1.0
    # the fundamental hunt keeps the bounded strongest-raw pool as its comb
    # targets: its majority rule is meaningless against every minor peak
    pool_a = ca[np.argsort(sa)[::-1][:_SURVEY_RATE_POOL_K]]
    pool_g = cg[np.argsort(sg)[::-1][:_SURVEY_RATE_POOL_K]]
    hunt_hi = max(lo, sample_rate / _SURVEY_CADENCE_HUNT_RATIO)
    sa = _damp_harmonics(ca, sa, _fundamental_below(fa, ma, hunt_hi, pool_a, dfa), dfa)
    sg = _damp_harmonics(cg, sg, _fundamental_below(fg, mg, hunt_hi, pool_g, dfg), dfg)

    cand = np.concatenate([ca, cg])
    strg = np.concatenate([sa, sg]) / norm
    tol_floor = 2.0 * max(float(dfa), float(dfg))
    rates, strengths = _merge_candidates(cand, strg, tol_floor)
    ranked, confirmed = _rerank_by_eye(
        [
            RateCandidate(
                rate_hz=r,
                strength=s,
                eye=_eye_openness(dphi, active_pairs, sample_rate, r),
            )
            for r, s in zip(rates, strengths)
        ]
    )
    return SymbolRateStats(
        candidates_hz=[round(c.rate_hz, 1) for c in ranked],
        strengths=[_sig(c.strength) for c in ranked],
        eye_openness=[_sig(c.eye) for c in ranked],
        eye_confirmed=confirmed,
        search_lo_hz=round(lo, 1),
        search_hi_hz=round(hi, 1),
        # the COARSER of the two branches, kept as max() even though the
        # phase branch now zero-fills its idle (equal record lengths, so the
        # two bins normally agree): a future branch that shortens its record
        # again must coarsen this number, not silently understate the
        # quantization under a docstring that tells the caller a hypothesis
        # within one bin is a match.
        clock_resolution_hz=round(max(float(dfa), float(dfg)), 1),
    )


def _merge_candidates(
    cand: npt.NDArray[np.floating[Any]],
    strg: npt.NDArray[np.floating[Any]],
    tol_floor: float,
) -> tuple[list[float], list[float]]:
    # dedup at the scale a line can actually smear (its own 2%, the comb
    # tolerance, floored at two clock bins) — a band-proportional tolerance
    # over a wide default search swallows distinct low-frequency lines into
    # their 2x harmonics
    merged_f: list[float] = []
    merged_s: list[float] = []
    for i in np.argsort(strg)[::-1]:
        f_ = float(cand[i])
        if all(
            abs(f_ - m) > max(_SURVEY_COMB_REL_TOL * m, tol_floor) for m in merged_f
        ):
            merged_f.append(f_)
            merged_s.append(float(strg[i]))
        if len(merged_f) == _SURVEY_RATE_K:
            break
    return merged_f, merged_s


def _inst_freq(x: npt.NDArray[np.complex64], sample_rate: float) -> InstFreqStats:
    # float64: numpy takes the bin dtype from the data, and a steady tone's
    # frequency span at tens of kHz is finer than float32 resolves there, so
    # 65 bins across it collapse into a "Too many bins for data range" raise.
    f = np.angle(x[1:] * np.conj(x[:-1])).astype(np.float64) / (2 * np.pi) * sample_rate
    f = _gate(
        f, _burst_power_gate_pairs(x, _SURVEY_ACTIVE_FRACTION, _SURVEY_BURST_WINDOW)
    )
    spread = float(f.std())
    # DROP: these counts are peak-picked below, and a clipped edge spike is
    # found as a tone that is not there
    span = percentile_span(f, bins=_SURVEY_IFREQ_BINS)
    counts, edges = binned_counts(f, _SURVEY_IFREQ_BINS, span, Tail.DROP)
    centers = (edges[:-1] + edges[1:]) / 2
    step = float(edges[1] - edges[0]) if edges.size > 1 else 0.0
    prom = max(float(counts.max()) * 0.05, 1.0)
    padded = np.concatenate(([0], counts, [0]))
    idx, _ = find_peaks(padded, prominence=prom, distance=2)
    return InstFreqStats(
        hist_counts=[int(v) for v in counts],
        hist_start_hz=round(float(centers[0]), 1) if centers.size else 0.0,
        hist_step_hz=round(step, 1),
        peaks_hz=[round(float(centers[i - 1]), 1) for i in idx],
        spread_hz=round(spread, 1),
    )


def _longest_true_run(mask: npt.NDArray[np.bool_]) -> tuple[int, int] | None:
    runs = _find_runs(mask)
    if not runs:
        return None
    return max(runs, key=lambda se: se[1] - se[0])


def _eye_openness(
    instfreq: npt.NDArray[np.floating[Any]],
    active: npt.NDArray[np.bool_],
    sample_rate: float,
    rate: float,
) -> float:
    check_deadline()
    if rate <= 0.0:
        return 0.0
    sps = sample_rate / rate
    if not np.isfinite(sps) or sps < 2.0:
        return 0.0
    run = _longest_true_run(active)
    if run is None:
        return 0.0
    lo, hi = run
    seg = np.asarray(instfreq[lo:hi], dtype=np.float64)
    n_sym = int(seg.size / sps) - 1
    if n_sym < _EYE_MIN_SYMBOLS:
        return 0.0
    n_sym = min(n_sym, _EYE_MAX_SYMBOLS)
    grid = np.arange(seg.size, dtype=np.float64)
    idx = np.arange(n_sym, dtype=np.float64)
    best = 0.0
    for p in range(_EYE_PHASE_STEPS):
        phase = (p + 0.5) * sps / _EYE_PHASE_STEPS
        samples = np.interp(phase + idx * sps, grid, seg)
        sep = fit_levels(samples).separation
        best = max(best, sep)
    return best


_SURVEY_BURST_FLOOR_PCTL = 25.0  # a unimodal probe's floor sample: low enough
# to sit under sparse activity, high enough that pure noise's own spread
# (0.906-0.917 of the median at this percentile under the window-64
# smoothing) stays near the median
_SURVEY_BURST_RATIO = 4.0  # activity bar above that floor; matches the rise
# ratio burst_sampler detects on, so survey and the demod agree on "active"
# A probe whose smoothed power has a genuine quiet SIDE under a loud median —
# the within-probe duty 0.5..~0.92 regime where the 25th-percentile floor
# sample IS the emitter's level.
# Adapted from embedded/burst.py's bimodality rule, re-measured for survey's
# window-64 power smoothing (probe 16384): noise-only probes read p5/med
# 0.779-0.85 (200 probes), 6.2x over the 1/8 bar, and the rule fires for
# emitters >= 9 dB over the quiet side (measured firing at 9 dB, not at 6).
# p5 rather than burst.py's p40: past duty 0.6 the p40 sits ON the signal, and
# the review's cliff (duty 0.74 -> 128 bursts, 0.76 -> 0) needs the quiet side
# found at duties where only the lowest few percent are quiet. Measured
# ceiling at burst period 4096: the p5 still reads the true noise floor
# (0.83-1.08x) through within-probe duty 0.92 and loses it at 0.94, where the
# off-run minus the smoothing window no longer spans 5% of the probe.
_QUIET_SIDE_PCTL = 5.0
_QUIET_SIDE_FRAC = 1.0 / 8.0
# Floor for an exactly-zero quiet side (digital silence smooths to true 0 and
# a 0 floor turns the whole noise band "active"): keeps the bar >= 12 dB under
# the probe's median, and sits under the 1/8 detection bound so a genuinely
# measured noise floor wins whenever the emitter is below ~18 dB over it.
_ZERO_QUIET_GUARD = 1.0 / 64.0
# An always-on emitter leaves NO quiet reference anywhere in the slice, and
# probe-scale smoothed power cannot tell it from dead noise (both unimodal:
# noise reads p5/med ~0.8 smoothed). The raw per-sample envelope can: complex
# noise power is exponential, p10/p50 = ln(10/9)/ln2 = 0.152 measured exactly,
# while a constant-envelope carrier reads 0.912 at 20 dB and 0.738 at 10 dB.
# 0.5 splits them with ~3x margin against noise. Amplitude-modulated always-on
# emitters sit BELOW it (16QAM 0.193, 50%-ones OOK 0.043) — their envelope
# minima are sample-scale indistinguishable from a quiet band, so they keep
# reading duty 0.0; that limitation is documented at the explain surface.
_CARRIER_SHAPE_MIN = 0.5


_MAX_INLINE_SEGMENTS = 512


class CaptureScale(Payload):
    """Maps a burst segment back onto the ORIGINAL capture so it can be
    re-decoded with a targeted slice: capture_offset = offset_samples +
    start*decim, capture_samples = length*decim."""

    offset_samples: int
    decim: int


class BurstStats(Payload):
    """The activity measurement, and — once `for_capture` has placed it — how to
    map a segment back onto the capture it came from. The last two fields are
    the wire's, set by the tool that knows the slice the measurement ran on."""

    # a measured-but-undefined period is a different answer from "not measured"
    always_present = frozenset({"dominant_period_samples"})

    count: int
    duty_cycle: float
    dominant_period_samples: int | None
    segments: list[tuple[int, int]]
    segments_total: int | None = None
    capture_scale: CaptureScale | None = None

    def for_capture(self, *, offset_samples: int, decim: int) -> "BurstStats":
        """Bounded for the agent's context and anchored to the original capture.
        Segments cap with a total and no sidecar: re-window to inspect a busier
        span, which is the answer a truncated list should push toward anyway."""
        # build(), not the plain constructor: segments_total is ABSENT on an
        # uncapped list rather than an explicit null, and dominant_period_samples
        # keeps the null it was measured with (always_present)
        return BurstStats.build(
            count=self.count,
            duty_cycle=self.duty_cycle,
            dominant_period_samples=self.dominant_period_samples,
            segments=list(self.segments[:_MAX_INLINE_SEGMENTS]),
            # the retained list is itself capped now, so truncation is
            # count vs shipped, never the length of what was held in memory
            segments_total=(self.count if self.count > len(self.segments) else None),
            capture_scale=CaptureScale(offset_samples=offset_samples, decim=decim),
        )


def _find_runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    starts = [int(i) for i in np.flatnonzero(d == 1) + 1]
    ends = [int(i) for i in np.flatnonzero(d == -1) + 1]
    if mask[0]:
        starts = [0, *starts]
    if mask[-1]:
        ends = [*ends, int(mask.size)]
    return list(zip(starts, ends))


@dataclass(frozen=True)
class _ProbeStats:
    quiet_side: bool
    floor_sample: float
    median: float
    envelope_shape: float
    high_fraction: float


def _probe_stats(block: npt.NDArray[np.complex64]) -> _ProbeStats:
    power = _smooth_power(block)
    quiet, low, med = (
        float(v)
        for v in np.percentile(
            power, [_QUIET_SIDE_PCTL, _SURVEY_BURST_FLOOR_PCTL, 50.0]
        )
    )
    quiet_side = quiet < _QUIET_SIDE_FRAC * med
    floor_sample = max(quiet, _ZERO_QUIET_GUARD * med) if quiet_side else low
    raw = np.abs(block).astype(np.float64) ** 2
    r10, r50 = (float(v) for v in np.percentile(raw, [10.0, 50.0]))
    return _ProbeStats(
        quiet_side=quiet_side,
        floor_sample=floor_sample,
        median=med,
        envelope_shape=r10 / r50 if r50 > 0.0 else 0.0,
        high_fraction=(
            float(np.mean(power >= med / _SURVEY_BURST_RATIO)) if med > 0.0 else 0.0
        ),
    )


@dataclass(frozen=True)
class _ActivityReference:
    threshold: float
    continuous_duty: float | None


def _activity_reference(path: Path, offset: int, length: int) -> _ActivityReference:
    """Activity bar, referenced to the slice's own noise floor. Referenced
    instead to the loudest emitter's level, every transmission an order of
    magnitude below it fell under the bar and was reported as absent."""
    stats = [_probe_stats(block) for block in iter_probes(path, offset, length)]
    if not stats:
        return _ActivityReference(threshold=0.0, continuous_duty=None)
    # a LOW percentile across probes, never the median: a continuous emitter
    # contaminates every probe it is on in, and at duty >= 50% the median of
    # per-probe floors WAS the emitter - a +200 kHz carrier 20 dB over noise
    # raised the bar 100x and deleted all 57 weak bursts elsewhere in the
    # band ("the band is dead"). The 10th percentile keeps the reference on
    # the cleanest probes when up to ~90% of PROBES are contaminated; a
    # probe's own quiet SIDE covers within-probe duty to the measured ~0.92
    # ceiling; an emitter continuous at BOTH scales masks weaker TOTAL-power
    # activity physically — no time-domain bar can recover that, which is
    # what continuous_duty reports instead of "dead".
    floor = float(np.percentile([s.floor_sample for s in stats], 10.0))
    typical = float(np.median([s.median for s in stats]))
    quiet_evidence = (
        any(s.quiet_side for s in stats) or floor < _QUIET_SIDE_FRAC * typical
    )
    shape = float(np.median([s.envelope_shape for s in stats]))
    continuous_duty = (
        float(np.mean([s.high_fraction for s in stats]))
        if not quiet_evidence and typical > 0.0 and shape >= _CARRIER_SHAPE_MIN
        else None
    )
    return _ActivityReference(
        threshold=_SURVEY_BURST_RATIO * floor, continuous_duty=continuous_duty
    )


def _smooth_power(block: npt.NDArray[np.complex64]) -> npt.NDArray[np.float64]:
    return windowed_power(block, _SURVEY_BURST_WINDOW)


def _bursts(path: Path, offset: int, length: int) -> BurstStats:
    ref = _activity_reference(path, offset, length)
    thr = ref.threshold
    segs: list[tuple[int, int]] = []
    count = 0
    active_total = 0
    pos = 0
    pending: list[int] | None = None

    def commit(seg: list[int]) -> None:
        # count everything, KEEP only what ships: 47,259 segments were held
        # to ship 512 - ~4.9 GB of tuples on a 300 s capture at 20 Msps. The
        # 512 retained starts are ample for the median period estimate.
        nonlocal count
        count += 1
        if len(segs) < _MAX_INLINE_SEGMENTS:
            segs.append((seg[0], seg[1] - seg[0]))

    for block in iter_iq(path, offset, length):
        mask = _smooth_power(block) > thr
        active_total += int(mask.sum())
        for s, e in _find_runs(mask):
            gs, ge = pos + s, pos + e
            if pending is not None and gs - pending[1] <= _SURVEY_BURST_WINDOW:
                pending[1] = ge
            else:
                if pending is not None:
                    commit(pending)
                pending = [gs, ge]
        pos += int(block.size)
    if pending is not None:
        commit(pending)
    starts = [s for s, _ in segs]
    deltas = np.diff(starts)
    dom = int(np.median(deltas)) if deltas.size else None
    duty = _sig(active_total / pos) if pos else 0.0
    if count == 0 and ref.continuous_duty is not None:
        # nothing cleared a bar built with no quiet reference anywhere in the
        # slice, and the raw envelope is carrier-shaped: the emitter is on the
        # whole time, not absent — "always on" and "never on" both stream zero
        # bursts, and only this distinction separates them.
        duty = _sig(ref.continuous_duty)
    return BurstStats(
        count=count,
        duty_cycle=duty,
        dominant_period_samples=dom,
        segments=segs,
    )


class SurveyResult(Payload):
    """survey's measurement AND its wire shape — the same ten fields, so there
    is nothing to keep in step. A separate payload model used to restate all of
    them and reach the wire via a dump/re-validate round trip; a field added
    here then failed `extra="forbid"` over there and reached the agent as
    [invalid_argument], blaming the caller for our own drift."""

    sample_rate: float
    span_samples: int
    analyzed_samples: int
    analyzed_start: int
    spectrum: SpectrumStats
    carrier: CarrierStats
    envelope: EnvelopeStats
    symbol_rate: SymbolRateStats
    inst_freq: InstFreqStats
    bursts: BurstStats

    def for_capture(self, *, offset_samples: int, decim: int) -> "SurveyResult":
        return replace(
            self,
            bursts=self.bursts.for_capture(offset_samples=offset_samples, decim=decim),
        )


def _default_rate_floor(n_samples: int, sample_rate: float) -> float:
    """The finest rate the chunked clock spectrum resolves with margin: a line
    below a few of its bins is indistinguishable from DC leakage, so searching
    there by default would rank noise. Anything above is fair game."""
    per = max(n_samples // _SURVEY_CLOCK_CHUNKS, 1)
    return _SURVEY_RATE_FLOOR_BINS * sample_rate / per


def survey_iq(
    path: Path,
    sample_rate: float,
    *,
    offset: int = 0,
    length: int = 0,
    min_symbol_rate: float | None = None,
    max_symbol_rate: float | None = None,
) -> SurveyResult:
    # named first, and by name: with sample_rate 0/negative/NaN the derived
    # bounds below both collapse and the failure surfaced as "require 0 <
    # min_symbol_rate < max_symbol_rate" — two parameters the caller had not
    # passed, about a value that was not the problem
    check_sample_rate(sample_rate)
    window = sample_iq(path, offset, length)
    x = window.samples
    bursts = _bursts(path, offset, length)
    hi = max_symbol_rate if max_symbol_rate is not None else sample_rate / 2
    if min_symbol_rate is not None:
        lo = min_symbol_rate
    else:
        lo = min(_default_rate_floor(x.size, sample_rate), hi / 2)
        if bursts.dominant_period_samples:
            # a bursty capture puts its burst-repetition comb inside the
            # default band, and the comb outranked the symbol line: measured
            # candidates [74.9 .. 861.2] Hz for a 125 kBd signal, across six
            # burst periods, with the truth NEVER in the list. The measured
            # cadence and its low harmonics are excluded from the default
            # band; an explicit min_symbol_rate still searches anywhere.
            cadence = sample_rate / bursts.dominant_period_samples
            lo = min(max(lo, _CADENCE_FLOOR_HARMONICS * cadence), hi / 2)
    if not 0 < lo < hi:
        raise ValueError(
            f"symbol-rate search band is empty: lo={lo:g} hi={hi:g}; require "
            f"0 < min_symbol_rate < max_symbol_rate (defaults derive from "
            f"sample_rate={sample_rate:g} and the analyzed span)"
        )
    check_deadline()
    spectrum = _spectrum(x, sample_rate)
    edge = _BAND_EDGE_FRAC * sample_rate
    band_edge_wrap = spectrum.occupied_hi_hz > edge or spectrum.occupied_lo_hz < -edge
    return SurveyResult(
        sample_rate=sample_rate,
        span_samples=window.span,
        analyzed_samples=window.analyzed,
        analyzed_start=window.start,
        spectrum=spectrum,
        carrier=_carrier(
            x,
            sample_rate,
            spectrum.occupied_bw_hz,
            spectrum.center_offset_hz,
            band_edge_wrap,
        ),
        envelope=_envelope(x),
        symbol_rate=_symbol_rate(x, sample_rate, lo, hi),
        inst_freq=_inst_freq(x, sample_rate),
        bursts=bursts,
    )
