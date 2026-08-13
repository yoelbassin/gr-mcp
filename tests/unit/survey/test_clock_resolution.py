"""clock_resolution_hz describes every candidate it is reported alongside.

symbol_rate merges peaks from two clock spectra — an ungated amplitude branch
and a magnitude-gated phase branch. Gating removes idle, so the phase branch's
record is shorter and its bin WIDER. Reporting the amplitude branch's bin alone
understated the quantization of every candidate the phase branch contributed,
under a survey docstring telling the caller a hypothesis within one bin is a
match.
"""

from __future__ import annotations

import numpy as np

from marconi.survey.measure import (
    _SURVEY_ACTIVE_FRACTION,
    _SURVEY_CLOCK_CHUNKS,
    _clock_spectrum,
    _gate,
    _magnitude_gate_pairs,
    _symbol_rate,
)


def _bursty(n: int = 200_000, duty: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64) * 0.01
    x[: int(n * duty)] *= 100
    return x


def _branch_bins(x: np.ndarray, sample_rate: float) -> tuple[float, float]:
    ya = np.abs(np.diff(np.abs(x)))
    dphi = np.angle(x[1:] * np.conj(x[:-1]))
    pairs = _magnitude_gate_pairs(x, _SURVEY_ACTIVE_FRACTION)
    yf = _gate(np.abs(np.diff(dphi)), pairs[1:] & pairs[:-1])
    fa, _ = _clock_spectrum(ya, sample_rate, _SURVEY_CLOCK_CHUNKS)
    fg, _ = _clock_spectrum(yf, sample_rate, _SURVEY_CLOCK_CHUNKS)
    return float(fa[1] - fa[0]), float(fg[1] - fg[0])


def test_a_bursty_capture_gives_the_two_branches_different_bins() -> None:
    """The premise. Without this the assertion below is vacuous."""
    dfa, dfg = _branch_bins(_bursty(), 1e6)
    assert dfg > dfa, (dfa, dfg)


def test_the_reported_resolution_is_the_coarser_branch() -> None:
    rate = 1e6
    x = _bursty()
    dfa, dfg = _branch_bins(x, rate)
    stats = _symbol_rate(x, rate, 1000.0, rate / 2)
    assert stats.clock_resolution_hz == round(max(dfa, dfg), 1)
    assert stats.clock_resolution_hz > round(dfa, 1)


def test_every_candidate_sits_within_the_reported_resolution_of_a_branch_bin() -> None:
    """What the number is FOR: a caller matching a hypothesis against
    candidates_hz must be able to trust one bin of tolerance."""
    rate = 1e6
    x = _bursty()
    dfa, dfg = _branch_bins(x, rate)
    stats = _symbol_rate(x, rate, 1000.0, rate / 2)
    tol = stats.clock_resolution_hz
    for c in stats.candidates_hz:
        on_a = abs(c - round(c / dfa) * dfa)
        on_g = abs(c - round(c / dfg) * dfg)
        assert min(on_a, on_g) <= tol, (c, on_a, on_g, tol)


def test_a_continuous_capture_still_reports_a_sane_resolution() -> None:
    rate = 1e6
    n = 200_000
    sym = np.repeat(np.random.default_rng(1).integers(0, 2, n // 10) * 2 - 1, 10)
    x = np.exp(1j * np.cumsum(sym * 0.3)).astype(np.complex64)
    stats = _symbol_rate(x, rate, 1000.0, rate / 2)
    assert stats.clock_resolution_hz > 0
    assert stats.clock_resolution_hz < rate
