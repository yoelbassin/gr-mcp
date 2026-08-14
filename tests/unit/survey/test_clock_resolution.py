"""clock_resolution_hz describes every candidate it is reported alongside.

symbol_rate merges peaks from two clock spectra - an ungated amplitude branch
and a phase branch whose idle is ZEROED, never deleted: deleting compressed
the record, and measuring the compressed array against the uncompressed
sample rate shifted every phase-branch candidate high by the kept fraction
(measured +4.5..9.3%, 20-104 clock bins, under a docstring promising one).
The resolution stays max(dfa, dfg) so a future branch that shortens its
record again must coarsen the reported number rather than understate it.
"""

from __future__ import annotations

import numpy as np

from marconi.survey.measure import _SURVEY_CLOCK_CHUNKS, _clock_spectrum, _symbol_rate


def _bursty(n: int = 200_000, duty: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64) * 0.01
    x[: int(n * duty)] *= 100
    return x


def _amplitude_bin(x: np.ndarray, sample_rate: float) -> float:
    ya = np.abs(np.diff(np.abs(x)))
    fa, _ = _clock_spectrum(ya, sample_rate, _SURVEY_CLOCK_CHUNKS)
    return float(fa[1] - fa[0])


def test_gating_no_longer_coarsens_the_phase_branch() -> None:
    # zero-filling keeps both branches on the full record, so the reported
    # resolution equals the amplitude branch's bin even on a 20%-duty capture
    # (the compressed design measured 98.3 Hz against 80.0 here)
    x = _bursty()
    stats = _symbol_rate(x, 1e6, 1000.0, 5e5)
    assert stats.clock_resolution_hz == round(_amplitude_bin(x, 1e6), 1)


def test_a_known_rate_lands_within_the_reported_resolution() -> None:
    """What the number is FOR: a caller matching a hypothesis against
    candidates_hz must be able to trust one bin of tolerance. Pinned against
    an EXTERNAL truth - the previous form checked each candidate against its
    own nearest bin, a distance that is at most half a bin for every real
    number in the universe, so fabricated candidates of 1.7 Hz and 999999 Hz
    passed it."""
    np.random.seed(1)
    fs, true_rate, n_sym = 48_000.0, 1_200.0, 4000
    sps = int(round(fs / true_rate))
    env = np.repeat(np.random.randint(0, 2, n_sym).astype(np.float64), sps)
    w = max(sps // 2, 1)
    env = np.convolve(env, np.ones(w) / w, mode="same")
    stats = _symbol_rate(env.astype(np.complex64), fs, fs / 1000, fs / 2)
    best = min(stats.candidates_hz, key=lambda c: abs(c - true_rate))
    assert abs(best - true_rate) <= stats.clock_resolution_hz, (
        best,
        stats.clock_resolution_hz,
    )
