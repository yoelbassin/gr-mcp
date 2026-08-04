import numpy as np

from marconi.engine.survey import _envelope, _spectrum


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
