from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis

from marconi.levels import fit_levels, percentile_span
from marconi.survey.iqfile import iter_iq, sample_iq

_SURVEY_NPERSEG = 4096
# Agent-facing arrays stay small: a coarse PSD conveys the spectral shape, and
# the frequency axis is a uniform ramp reconstructed from start+step, never
# shipped as hundreds of derivable floats.
_SPECTRUM_BINS = 64
_SURVEY_ACTIVE_FRACTION = 0.3
_SURVEY_BURST_WINDOW = 64


class EnvelopeStats(BaseModel):
    const_envelope_ratio: float
    amplitude_kurtosis: float
    mean_amplitude: float
    std_amplitude: float


class SpectrumStats(BaseModel):
    psd_db: list[float]
    freq_start_hz: float
    freq_step_hz: float
    center_offset_hz: float
    peak_offset_hz: float
    occupied_bw_hz: float
    occupied_lo_hz: float
    occupied_hi_hz: float


class PhaseConcentration(BaseModel):
    order_2: float
    order_4: float
    order_8: float


class CarrierStats(BaseModel):
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
    total = float(pxx.sum()) or 1.0
    centroid = float((f * pxx).sum() / total)
    peak = float(f[int(np.argmax(pxx))])
    cum = np.cumsum(pxx) / total
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


def _carrier(
    x: npt.NDArray[np.complex64],
    sample_rate: float,
    occupied_bw_hz: float,
    centroid_hz: float,
) -> CarrierStats:
    xb = _bounded(x)
    active = _slot_active_mask(xb, _SURVEY_ACTIVE_FRACTION, _SURVEY_BURST_WINDOW)
    dof = int(active.sum()) or xb.size
    xd = xb - _active_mean(xb, active)
    z = (xd / np.maximum(np.abs(xd), 1e-12)) * active
    scores: dict[int, float] = {}
    lines: dict[int, float] = {}
    for m in _MPSK_ORDERS:
        scores[m], lines[m] = _mpsk_line(z, sample_rate, m, dof)
    order = next(
        (
            m
            for m in (4, 8)
            if scores[m] >= _MPSK_CONC_THRESH
            and scores[m] >= _MPSK_JUMP * scores[m // 2]
        ),
        None,
    )
    if order == 4 and scores[8] >= _MPSK_CONC_THRESH:
        off4 = _dealias_offset(lines[4], 4, sample_rate, centroid_hz)
        off8 = _dealias_offset(lines[8], 8, sample_rate, centroid_hz)
        if abs(off8 - off4) > _STAGGER_TOL_BINS * sample_rate / xb.size:
            order = 8
    ambiguous = False
    if order is not None:
        offset = _dealias_offset(lines[order], order, sample_rate, centroid_hz)
        if abs(centroid_hz - offset) / (sample_rate / order) > _DEALIAS_AMBIGUOUS_FRAC:
            offset, ambiguous = centroid_hz, True
    else:
        offset = centroid_hz
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
    active = _slot_active_mask(x, _SURVEY_ACTIVE_FRACTION, _SURVEY_BURST_WINDOW)
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
_CLEAR_EYE = 8.0  # measured: unimodal / wrong-rate smear tops out ~3.6; a real eye ~25
_HARMONIC_FRAC = 0.6


class InstFreqStats(BaseModel):
    hist_counts: list[int]
    hist_start_hz: float
    hist_step_hz: float
    peaks_hz: list[float]
    spread_hz: float


class SymbolRateStats(BaseModel):
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


def _active_mask(
    x: npt.NDArray[np.complex64], fraction: float
) -> npt.NDArray[np.bool_]:
    mag = np.abs(x)
    med = float(np.median(mag))
    if med <= 0.0:
        return np.ones(x.size, dtype=bool)
    return mag > fraction * med


def _active_pairs(
    x: npt.NDArray[np.complex64], fraction: float
) -> npt.NDArray[np.bool_]:
    active = _active_mask(x, fraction)
    return active[1:] & active[:-1]


def _slot_active_mask(
    x: npt.NDArray[np.complex64], fraction: float, window: int
) -> npt.NDArray[np.bool_]:
    power: npt.NDArray[np.float64] = uniform_filter1d(
        np.abs(x).astype(np.float64) ** 2, window
    )
    # reference the top-decile median, not the max: a single hot interferer
    # burst inside the span would otherwise set the threshold and gate the
    # actual signal of interest out of the mask entirely
    p90 = float(np.percentile(power, 90))
    top = power[power > p90]
    peak = float(np.median(top)) if top.size else p90
    if peak <= 0.0:
        return np.ones(x.size, dtype=bool)
    return power > fraction * peak


def _slot_active_pairs(
    x: npt.NDArray[np.complex64], fraction: float, window: int
) -> npt.NDArray[np.bool_]:
    active = _slot_active_mask(x, fraction, window)
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
    # A candidate's eye "clears" only above the measured smear ceiling, so this
    # is the single source for both the re-rank trigger and the caller-facing
    # eye_confirmed flag — when False, candidates_hz is strength-ranked only and
    # untrustworthy for constant-envelope or pulse-shaped (e.g. GMSK) signals.
    return bool(eyes) and max(eyes) >= _CLEAR_EYE


@dataclass(frozen=True)
class RateCandidate:
    """One symbol-rate hypothesis with the two measurements that rank it.
    Kept as one object so the three cannot drift out of correspondence."""

    rate_hz: float
    strength: float
    eye: float


def _rerank_by_eye(candidates: list[RateCandidate]) -> list[RateCandidate]:
    eyes = [c.eye for c in candidates]
    if not _eye_cleared(eyes):
        return candidates
    thresh = _HARMONIC_FRAC * max(eyes)
    strong = [c for c in candidates if c.eye >= thresh]
    primary = min(strong, key=lambda c: c.rate_hz)
    rest = sorted(
        (c for c in candidates if c is not primary), key=lambda c: c.eye, reverse=True
    )
    return [primary, *rest]


def _symbol_rate(
    x: npt.NDArray[np.complex64], sample_rate: float, lo: float, hi: float
) -> SymbolRateStats:
    ya = np.abs(np.diff(np.abs(x)))
    dphi = np.angle(x[1:] * np.conj(x[:-1]))
    yf_full = np.abs(np.diff(dphi))

    active_pairs = _active_pairs(x, _SURVEY_ACTIVE_FRACTION)
    yf = _gate(yf_full, active_pairs[1:] & active_pairs[:-1])

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
    ranked = _rerank_by_eye(
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
        eye_confirmed=_eye_cleared([c.eye for c in ranked]),
        search_lo_hz=round(lo, 1),
        search_hi_hz=round(hi, 1),
        clock_resolution_hz=round(float(dfa), 1),
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
    f = np.angle(x[1:] * np.conj(x[:-1])) / (2 * np.pi) * sample_rate
    f = _gate(f, _slot_active_pairs(x, _SURVEY_ACTIVE_FRACTION, _SURVEY_BURST_WINDOW))
    spread = float(f.std())
    lo, hi = percentile_span(f)
    counts, edges = np.histogram(f, bins=_SURVEY_IFREQ_BINS, range=(lo, hi))
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


_SURVEY_BURST_FRACTION = 0.35


class BurstStats(BaseModel):
    count: int
    duty_cycle: float
    dominant_period_samples: int | None
    segments: list[tuple[int, int]]


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


def _bursts(
    sample_x: npt.NDArray[np.complex64], path: Path, offset: int, length: int
) -> BurstStats:
    thr = _SURVEY_BURST_FRACTION * float(np.percentile(np.abs(sample_x) ** 2, 90))
    segs: list[tuple[int, int]] = []
    active_total = 0
    pos = 0
    pending: list[int] | None = None
    for block in iter_iq(path, offset, length):
        p = uniform_filter1d(
            np.abs(block).astype(np.float64) ** 2, _SURVEY_BURST_WINDOW
        )
        mask = p > thr
        active_total += int(mask.sum())
        for s, e in _find_runs(mask):
            gs, ge = pos + s, pos + e
            if pending is not None and gs - pending[1] <= _SURVEY_BURST_WINDOW:
                pending[1] = ge
            else:
                if pending is not None:
                    segs.append((pending[0], pending[1] - pending[0]))
                pending = [gs, ge]
        pos += int(block.size)
    if pending is not None:
        segs.append((pending[0], pending[1] - pending[0]))
    starts = [s for s, _ in segs]
    deltas = np.diff(starts)
    dom = int(np.median(deltas)) if deltas.size else None
    return BurstStats(
        count=len(segs),
        duty_cycle=_sig(active_total / pos) if pos else 0.0,
        dominant_period_samples=dom,
        segments=segs,
    )


class SurveyResult(BaseModel):
    sample_rate: float
    span_samples: int
    analyzed_samples: int
    spectrum: SpectrumStats
    carrier: CarrierStats
    envelope: EnvelopeStats
    symbol_rate: SymbolRateStats
    inst_freq: InstFreqStats
    bursts: BurstStats


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
    x, analyzed, span = sample_iq(path, offset, length)
    hi = max_symbol_rate if max_symbol_rate is not None else sample_rate / 2
    if min_symbol_rate is not None:
        lo = min_symbol_rate
    else:
        lo = min(_default_rate_floor(x.size, sample_rate), hi / 2)
    if not 0 < lo < hi:
        raise ValueError("require 0 < min_symbol_rate < max_symbol_rate")
    spectrum = _spectrum(x, sample_rate)
    return SurveyResult(
        sample_rate=sample_rate,
        span_samples=span,
        analyzed_samples=analyzed,
        spectrum=spectrum,
        carrier=_carrier(
            x, sample_rate, spectrum.occupied_bw_hz, spectrum.center_offset_hz
        ),
        envelope=_envelope(x),
        symbol_rate=_symbol_rate(x, sample_rate, lo, hi),
        inst_freq=_inst_freq(x, sample_rate),
        bursts=_bursts(x, path, offset, length),
    )
