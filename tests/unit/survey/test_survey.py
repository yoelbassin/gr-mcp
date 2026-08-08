from pathlib import Path

import numpy as np
import pytest

from marconi.survey.measure import (
    _SURVEY_CLOCK_CHUNKS,
    _SURVEY_COMB_DAMPING,
    _SURVEY_RATE_FLOOR_BINS,
    _bursts,
    _damp_harmonics,
    _default_rate_floor,
    _envelope,
    _fundamental_below,
    _inst_freq,
    _spectrum,
    _symbol_rate,
    survey_iq,
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
    assert len(s.psd_db) <= 64
    assert s.freq_step_hz > 0.0 and np.isfinite(s.freq_start_hz)


def test_spectrum_zero_power_input() -> None:
    x = np.zeros(8192, dtype=np.complex64)
    s = _spectrum(x, 48_000.0)
    assert s.occupied_bw_hz >= 0.0
    assert np.isfinite(s.occupied_lo_hz) and np.isfinite(s.occupied_hi_hz)
    assert len(s.psd_db) <= 64
    assert np.isfinite(s.freq_start_hz) and np.isfinite(s.freq_step_hz)


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
    on, off, reps = 4_000, 4_000, 8
    burst = np.exp(2j * np.pi * 0.2 * np.arange(on)).astype(np.complex64)
    slot = np.concatenate([burst, np.zeros(off, np.complex64)])
    x = np.tile(slot, reps)
    assert _envelope(x).const_envelope_ratio < 0.1


def test_envelope_kurtosis_matches_two_level_ground_truth() -> None:
    n = 1 << 14
    rng = np.random.default_rng(0)
    amp = rng.choice([1.0, 3.0], size=n)
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


def test_envelope_detects_rectangular_ook() -> None:
    np.random.seed(1)
    fs, rate = 48_000.0, 1_200.0
    x = _ook(fs, rate, 4000)
    const = np.exp(2j * np.pi * 0.05 * np.arange(x.size)).astype(np.complex64)
    assert _envelope(x).const_envelope_ratio > 0.2
    assert _envelope(x).const_envelope_ratio > _envelope(const).const_envelope_ratio


def test_default_rate_floor_tracks_span() -> None:
    # a longer analyzed span resolves finer clock lines, so its default floor
    # sits lower; the floor is a few bins of the chunked clock spectrum
    fs = 1_000_000.0
    short, long_ = _default_rate_floor(1 << 13, fs), _default_rate_floor(1 << 20, fs)
    assert long_ < short
    per = (1 << 20) // _SURVEY_CLOCK_CHUNKS
    assert long_ == pytest.approx(_SURVEY_RATE_FLOOR_BINS * fs / per)


def test_survey_default_floor_finds_rate_below_permille_ratio(
    tmp_path: Path,
) -> None:
    # 40 baud at 48 ksps sits below the old fixed sample_rate/1000 floor
    # (48 Hz) but well above what this span resolves — the default search
    # must include it
    np.random.seed(4)
    fs, rate = 48_000.0, 40.0
    x = _fsk(fs, rate, 2_000.0, 110, np.array([-1.0, 1.0]), smooth_frac=0.2)
    p = tmp_path / "slow.cf32"
    x.tofile(p)
    s = survey_iq(p, fs).symbol_rate
    assert s.search_lo_hz < rate < s.search_hi_hz
    assert s.clock_resolution_hz > 0.0
    assert min(abs(c - rate) for c in s.candidates_hz) < 1.5 * s.clock_resolution_hz


def test_survey_small_max_rate_clamps_default_floor(tmp_path: Path) -> None:
    # an explicit max below the derived floor must clamp the default lo, not
    # reject the search
    np.random.seed(5)
    x = (np.random.randn(1 << 13) + 1j * np.random.randn(1 << 13)).astype(np.complex64)
    p = tmp_path / "short.cf32"
    x.tofile(p)
    fs = 48_000.0
    assert _default_rate_floor(1 << 13, fs) > 8.0
    s = survey_iq(p, fs, max_symbol_rate=8.0).symbol_rate
    assert 0.0 < s.search_lo_hz < 8.0


def test_symbol_rate_pure_noise_is_honest() -> None:
    np.random.seed(2)
    fs = 48_000.0
    x = (np.random.randn(1 << 15) + 1j * np.random.randn(1 << 15)).astype(np.complex64)
    s = _symbol_rate(x, fs, fs / 1000, fs / 2)
    assert len(s.candidates_hz) == len(s.strengths)
    assert len(s.strengths) < 2 or s.strengths[1] > 0.5 * s.strengths[0]


@pytest.fixture
def _synth_fsk_capture() -> tuple[np.ndarray, float, float]:
    fs, true_baud, n_sym, dev, sfo = 48_000.0, 3_700.0, 4000, 900.0, 5e-5
    smooth_frac, decoy_rate, decoy_depth = 0.2, 2_050.0, 0.4

    rng = np.random.default_rng(0)
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    syms = levels[rng.integers(0, levels.size, n_sym)]

    sps = fs / true_baud
    span = int(n_sym * sps)
    t = np.arange(span, dtype=np.float64)
    sym_index = np.clip(np.floor(t / (sps * (1.0 + sfo))).astype(int), 0, n_sym - 1)
    freq = syms[sym_index] * dev
    window = max(round(sps * smooth_frac), 1)
    freq = np.convolve(freq, np.ones(window) / window, mode="same")

    phase = 2 * np.pi * np.cumsum(freq) / fs
    iq = np.roll(np.exp(1j * phase), 3)

    ripple = 1.0 + decoy_depth * np.sin(2 * np.pi * decoy_rate * t / fs)
    iq = iq * ripple
    return iq.astype(np.complex64), fs, true_baud


def test_symbol_rate_reranks_true_baud_above_spurious_line(
    _synth_fsk_capture: tuple[np.ndarray, float, float],
) -> None:
    # _synth_fsk_capture: a 4-level FSK IQ array at a generic baud with a decoy
    # low-frequency amplitude comb; helper added in this test module.
    x, fs, true_baud = _synth_fsk_capture
    stats = _symbol_rate(x, fs, lo=fs / 1000, hi=fs / 2)
    assert stats.eye_openness  # populated, aligned to candidates
    assert len(stats.eye_openness) == len(stats.candidates_hz)
    assert abs(stats.candidates_hz[0] - true_baud) / true_baud < 0.05


def test_symbol_rate_prefers_fundamental_over_harmonic(
    _synth_fsk_capture: tuple[np.ndarray, float, float],
) -> None:
    x, fs, true_baud = _synth_fsk_capture
    stats = _symbol_rate(x, fs, lo=fs / 1000, hi=fs / 2)
    # a 2x-baud harmonic must never outrank the fundamental
    assert stats.candidates_hz[0] <= true_baud * 1.5


def test_symbol_rate_rerank_promotes_a_non_strongest_candidate(
    _synth_fsk_capture: tuple[np.ndarray, float, float],
) -> None:
    x, fs, true_baud = _synth_fsk_capture
    stats = _symbol_rate(x, fs, lo=fs / 1000, hi=fs / 2)
    assert stats.strengths[0] < max(stats.strengths)


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


def test_comb_hunt_rejects_bin_scale_fundamental() -> None:
    # a "fundamental" only ~3 bins wide has comb spacing inside the +/-2-bin
    # match tolerance: adjacent orders overlap, the mask saturates, and every
    # line in the band reads as a harmonic — such a fundamental must never be
    # hunted (the NELoRa sf7 regression: a 601 Hz 3-bin line wiped the true
    # 1002 Hz candidate as its "order 2")
    bin_hz = 200.0
    g0 = 3.0 * bin_hz
    freqs = np.arange(1, 60) * bin_hz
    mag = np.ones(freqs.size)
    mag[2] = 50.0  # the 3-bin line
    targets = np.array([2 * g0, 3 * g0, 4 * g0, 5 * g0])
    assert _fundamental_below(freqs, mag, 1_000.0, targets, bin_hz) is None


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
    assert len(s.hist_counts) == 65
    assert s.hist_step_hz > 0.0 and np.isfinite(s.hist_start_hz)
    assert s.spread_hz > dev


def test_inst_freq_spread_is_near_zero_for_constant_frequency() -> None:
    fs = 48_000.0
    x = np.exp(2j * np.pi * 0.05 * np.arange(1 << 14)).astype(np.complex64)
    s = _inst_freq(x, fs)
    assert s.spread_hz < 50.0


def test_inst_freq_spread_is_wide_for_unshaped_psk_not_just_fsk() -> None:
    fs, rate = 48_000.0, 2_400.0
    sps = int(round(fs / rate))
    rng = np.random.default_rng(3)
    phases = rng.integers(0, 4, 3000) * (np.pi / 2) + np.pi / 4
    x = np.repeat(np.exp(1j * phases), sps).astype(np.complex64)
    s = _inst_freq(x, fs)
    assert s.spread_hz > rate
    assert _envelope(x).const_envelope_ratio < 0.1


def test_inst_freq_spread_is_wide_for_a_noisy_tone_not_just_fsk() -> None:
    fs = 48_000.0
    rng = np.random.default_rng(5)
    n = 1 << 15
    tone = np.exp(2j * np.pi * 0.05 * np.arange(n))
    noise_power = 1.0 / (10 ** (10.0 / 10))
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    x = (tone + noise).astype(np.complex64)
    s = _inst_freq(x, fs)
    assert s.spread_hz > 1_000.0


def test_inst_freq_all_zero_input_is_safe() -> None:
    s = _inst_freq(np.zeros(4096, dtype=np.complex64), 48_000.0)
    assert s.spread_hz == 0.0
    assert len(s.hist_counts) == 65


def test_envelope_and_inst_freq_agree_whole_file_vs_single_burst() -> None:
    fs, rate, dev = 48_000.0, 2_400.0, 1_500.0
    np.random.seed(7)
    burst = _fsk(
        fs, rate, dev, 4_000, np.array([-3.0, -1.0, 1.0, 3.0]), smooth_frac=0.3
    )
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
