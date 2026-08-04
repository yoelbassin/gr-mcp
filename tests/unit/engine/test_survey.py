import numpy as np

from marconi.engine.survey import _envelope, _inst_freq, _spectrum, _symbol_rate


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
    assert isinstance(s, type(_spectrum(np.zeros(1, dtype=np.complex64), 48_000.0)))
    assert len(s.freqs_hz) == len(s.psd_db) <= 512


def test_envelope_separates_constant_from_ook() -> None:
    n = 1 << 14
    const = np.exp(2j * np.pi * 0.05 * np.arange(n)).astype(np.complex64)
    ook = const.copy()
    ook[: n // 2] = 0.0
    assert _envelope(const).const_envelope_ratio < 0.1
    assert _envelope(ook).const_envelope_ratio > _envelope(const).const_envelope_ratio
    assert _envelope(const).mean_amplitude > 0.0


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


def test_inst_freq_finds_four_fsk_tones() -> None:
    np.random.seed(42)
    fs, rate, dev = 48_000.0, 2_400.0, 1_500.0
    x = _fsk(fs, rate, dev, 6000, np.array([-3.0, -1.0, 1.0, 3.0]))
    s = _inst_freq(x, fs)
    assert 3 <= len(s.peaks_hz) <= 5
    peaks = sorted(s.peaks_hz)
    assert peaks[0] < -dev and peaks[-1] > dev
    assert len(s.hist_centers_hz) == len(s.hist_counts) == 65
