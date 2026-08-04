import numpy as np

from marconi.engine.survey import _spectrum


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
