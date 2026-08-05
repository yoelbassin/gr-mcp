from pathlib import Path

import numpy as np
import pytest

from marconi.survey.measure import (
    _SURVEY_COMB_DAMPING,
    _bursts,
    _damp_harmonics,
    _envelope,
    _fundamental_below,
    _inst_freq,
    _spectrum,
    _symbol_rate,
)


def test_spectrum_locates_offset_tone() -> None:
    np.random.seed(0)
    fs, n, f0 = 48_000.0, 1 << 15, 6_000.0
    t = np.arange(n) / fs
    x = np.exp(2j * np.pi * f0 * t).astype(np.complex64)
    x += (0.01 * (np.random.randn(n) + 1j * np.random.randn(n))).astype(np.complex64)
    s = _spectrum(x, fs)
    assert abs(s.peak_offset_hz - f0) < 200.0
    assert abs(s.center_offset_hz - f0) < 500.0
    assert s.occupied_bw_hz < fs / 4
    assert len(s.freqs_hz) == len(s.psd_db) <= 512


def test_spectrum_zero_power_input() -> None:
    x = np.zeros(8192, dtype=np.complex64)
    s = _spectrum(x, 48_000.0)
    assert s.occupied_bw_hz >= 0.0
    assert np.isfinite(s.occupied_lo_hz) and np.isfinite(s.occupied_hi_hz)
    assert len(s.freqs_hz) == len(s.psd_db) <= 512


def test_envelope_separates_constant_from_varying_amplitude() -> None:
    n = 1 << 14
    const = np.exp(2j * np.pi * 0.05 * np.arange(n)).astype(np.complex64)
    rng = np.random.default_rng(0)
    levels = rng.choice([1.0, 2.5], size=n)
    ask = (levels * np.exp(2j * np.pi * 0.05 * np.arange(n))).astype(np.complex64)
    assert _envelope(const).const_envelope_ratio < 0.1
    assert _envelope(ask).const_envelope_ratio > _envelope(const).const_envelope_ratio
    assert _envelope(const).mean_amplitude > 0.0


def test_envelope_ignores_off_gaps_between_constant_bursts() -> None:
    on, off, reps = 400, 400, 8
    burst = np.exp(2j * np.pi * 0.2 * np.arange(on)).astype(np.complex64)
    slot = np.concatenate([burst, np.zeros(off, np.complex64)])
    x = np.tile(slot, reps)
    assert _envelope(x).const_envelope_ratio < 0.1


def test_envelope_kurtosis_matches_two_level_ground_truth() -> None:
    n = 1 << 14
    amp = np.concatenate([np.ones(n // 2), 3.0 * np.ones(n // 2)])
    x = (amp * np.exp(2j * np.pi * 0.05 * np.arange(n))).astype(np.complex64)
    e = _envelope(x)
    assert abs(e.mean_amplitude - 2.0) < 0.05
    assert abs(e.std_amplitude - 1.0) < 0.05
    assert abs(e.amplitude_kurtosis - (-2.0)) < 0.1


def test_envelope_zero_input_is_safe() -> None:
    e = _envelope(np.zeros(4096, dtype=np.complex64))
    assert e.const_envelope_ratio == 0.0
    assert e.amplitude_kurtosis == 0.0


def _fsk(
    fs: float,
    rate: float,
    dev: float,
    n_sym: int,
    levels: np.ndarray,
    smooth_frac: float = 0.0,
) -> np.ndarray:
    sps = int(round(fs / rate))
    syms = levels[np.random.randint(0, levels.size, n_sym)]
    inst = np.repeat(syms * dev, sps)
    if smooth_frac > 0:
        w = max(round(sps * smooth_frac), 1)
        inst = np.convolve(inst, np.ones(w) / w, mode="same")
    phase = 2 * np.pi * np.cumsum(inst) / fs
    return np.exp(1j * phase).astype(np.complex64)


def test_symbol_rate_recovers_fsk_baud() -> None:
    np.random.seed(0)
    fs, rate = 48_000.0, 2_400.0
    x = _fsk(fs, rate, 1_000.0, 4000, np.array([-1.0, 1.0]), smooth_frac=0.5)
    s = _symbol_rate(x, fs, fs / 1000, fs / 2)
    assert s.candidates_hz, "expected at least one candidate"
    assert min(abs(c - rate) for c in s.candidates_hz) < 0.05 * rate
    assert abs(s.candidates_hz[0] - rate) < 0.05 * rate


def _ook(fs: float, rate: float, n_sym: int) -> np.ndarray:
    sps = int(round(fs / rate))
    bits = np.random.randint(0, 2, n_sym).astype(np.float64)
    env = np.repeat(bits, sps)
    w = max(sps // 2, 1)
    env = np.convolve(env, np.ones(w) / w, mode="same")
    return env.astype(np.complex64)


def test_symbol_rate_recovers_ook_baud() -> None:
    np.random.seed(1)
    fs, rate = 48_000.0, 1_200.0
    x = _ook(fs, rate, 4000)
    s = _symbol_rate(x, fs, fs / 1000, fs / 2)
    assert s.candidates_hz, "expected at least one candidate"
    assert abs(s.candidates_hz[0] - rate) < 0.05 * rate


def test_symbol_rate_pure_noise_is_honest() -> None:
    np.random.seed(2)
    fs = 48_000.0
    x = (np.random.randn(1 << 15) + 1j * np.random.randn(1 << 15)).astype(np.complex64)
    s = _symbol_rate(x, fs, fs / 1000, fs / 2)
    assert len(s.candidates_hz) == len(s.strengths)
    assert len(s.strengths) < 2 or s.strengths[1] > 0.5 * s.strengths[0]


def test_comb_suppression_damps_majority_harmonic_pool() -> None:
    g0, lo, bin_hz = 40.0, 90.0, 1.0
    freqs = np.array([g0 - 2, g0 - 1, g0, g0 + 1, g0 + 2])
    mag = np.array([1.0, 5.0, 50.0, 5.0, 1.0])

    harmonics = np.array([2 * g0, 3 * g0, 4 * g0])
    non_harmonic_line = 25 * g0
    targets = np.concatenate([harmonics, [non_harmonic_line]])
    strengths = np.array([10.0, 8.0, 6.0, 4.0])

    fundamental = _fundamental_below(freqs, mag, lo, targets, bin_hz)
    assert fundamental == g0

    damped = _damp_harmonics(targets, strengths, fundamental, bin_hz)
    assert np.allclose(damped[:3], strengths[:3] * _SURVEY_COMB_DAMPING)
    assert damped[3] == strengths[3]


def test_comb_suppression_does_not_fire_without_a_majority() -> None:
    g0, lo, bin_hz = 40.0, 90.0, 1.0
    freqs = np.array([g0 - 2, g0 - 1, g0, g0 + 1, g0 + 2])
    mag = np.array([1.0, 5.0, 50.0, 5.0, 1.0])

    targets = np.array([2 * g0, 25 * g0, 26 * g0, 27 * g0])
    strengths = np.array([9.0, 7.0, 5.0, 3.0])

    fundamental = _fundamental_below(freqs, mag, lo, targets, bin_hz)
    assert fundamental is None

    damped = _damp_harmonics(targets, strengths, fundamental, bin_hz)
    assert np.array_equal(damped, strengths)


def test_inst_freq_finds_four_fsk_tones() -> None:
    np.random.seed(42)
    fs, rate, dev = 48_000.0, 2_400.0, 1_500.0
    x = _fsk(fs, rate, dev, 6000, np.array([-3.0, -1.0, 1.0, 3.0]))
    s = _inst_freq(x, fs)
    assert len(s.peaks_hz) == 4
    expected = [-4500.0, -1500.0, 1500.0, 4500.0]
    for got, want in zip(sorted(s.peaks_hz), expected):
        assert abs(got - want) < 200.0, (sorted(s.peaks_hz), expected)
    assert len(s.hist_centers_hz) == len(s.hist_counts) == 65
    assert s.spread_hz > dev


def test_inst_freq_spread_is_near_zero_for_constant_frequency() -> None:
    fs = 48_000.0
    x = np.exp(2j * np.pi * 0.05 * np.arange(1 << 14)).astype(np.complex64)
    s = _inst_freq(x, fs)
    assert s.spread_hz < 50.0


def test_envelope_and_inst_freq_agree_whole_file_vs_single_burst() -> None:
    fs, rate, dev = 48_000.0, 2_400.0, 1_500.0
    np.random.seed(7)
    burst = _fsk(fs, rate, dev, 400, np.array([-3.0, -1.0, 1.0, 3.0]), smooth_frac=0.3)
    gap = np.zeros(burst.size, dtype=np.complex64)
    whole = np.concatenate([burst, gap, burst, gap, burst])

    e_whole, e_burst = _envelope(whole), _envelope(burst)
    assert e_whole.const_envelope_ratio < 0.1
    assert abs(e_whole.const_envelope_ratio - e_burst.const_envelope_ratio) < 0.02

    f_whole, f_burst = _inst_freq(whole, fs), _inst_freq(burst, fs)
    assert abs(f_whole.spread_hz - f_burst.spread_hz) < 0.1 * f_burst.spread_hz


def test_bursts_recovers_periodic_cadence(tmp_path: Path) -> None:
    on, off, reps = 400, 400, 30
    carrier = np.exp(2j * np.pi * 0.1 * np.arange(on)).astype(np.complex64)
    slot = np.concatenate([carrier, np.zeros(off, np.complex64)])
    x = np.tile(slot, reps)
    p = tmp_path / "bursty.cf32"
    x.tofile(p)
    b = _bursts(x, p, 0, 0)
    assert abs(b.count - reps) <= 1
    assert 0.4 < b.duty_cycle < 0.6
    assert b.dominant_period_samples is not None
    assert abs(b.dominant_period_samples - (on + off)) < 40


def test_bursts_stitches_across_chunk_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    on, off, reps = 400, 400, 30
    carrier = np.exp(2j * np.pi * 0.1 * np.arange(on)).astype(np.complex64)
    slot = np.concatenate([carrier, np.zeros(off, np.complex64)])
    x = np.tile(slot, reps)
    p = tmp_path / "bursty.cf32"
    x.tofile(p)
    import marconi.survey.measure as survey_mod
    from marconi.survey import iqfile

    monkeypatch.setattr(
        survey_mod,
        "iter_iq",
        lambda path, offset, length: iqfile.iter_iq(path, offset, length, chunk=500),
    )
    b = _bursts(x, p, 0, 0)
    assert abs(b.count - reps) <= 1
    assert b.dominant_period_samples is not None
    assert abs(b.dominant_period_samples - (on + off)) < 40
    assert 0.4 < b.duty_cycle < 0.6
