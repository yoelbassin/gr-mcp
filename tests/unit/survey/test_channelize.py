from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.survey import iqfile
from marconi.survey.iqfile import channelize_to_file


def _tone(fs: float, f: float, n: int) -> np.ndarray:
    return np.exp(2j * np.pi * f * np.arange(n) / fs).astype(np.complex64)


def test_channelize_shifts_center_to_dc(tmp_path: Path) -> None:
    fs, n, decim, f1 = 48_000.0, 1 << 16, 4, 6_000.0
    src = tmp_path / "in.cf32"
    _tone(fs, f1, n).tofile(src)
    dst = tmp_path / "out.cf32"
    written, out_rate = channelize_to_file(src, dst, fs, center_hz=f1, decim=decim)
    assert out_rate == fs / decim
    y = np.fromfile(dst, np.complex64)
    assert y.size == written
    assert abs(written - n // decim) <= decim
    freqs = np.fft.fftfreq(y.size, d=1.0 / out_rate)
    peak = freqs[int(np.argmax(np.abs(np.fft.fft(y))))]
    assert abs(peak) < 0.02 * out_rate


def test_channelize_rejects_out_of_band_tone(tmp_path: Path) -> None:
    fs, n, decim = 48_000.0, 1 << 16, 6  # out_rate 8k, passband cutoff ~3.6k
    f_ch, f_far = 3_000.0, -3_000.0  # 6 kHz apart -> far tone in the stopband
    src = tmp_path / "in.cf32"
    (_tone(fs, f_ch, n) + _tone(fs, f_far, n)).astype(np.complex64).tofile(src)
    dst = tmp_path / "out.cf32"
    channelize_to_file(src, dst, fs, center_hz=f_ch, decim=decim)
    y = np.fromfile(dst, np.complex64)
    out_rate = fs / decim
    freqs = np.fft.fftfreq(y.size, d=1.0 / out_rate)
    power = np.abs(np.fft.fft(y)) ** 2
    near_dc = np.abs(freqs) < 500.0
    assert power[near_dc].sum() / power.sum() > 0.9


def test_channelize_streaming_matches_single_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fs, n, decim, f1 = 48_000.0, 1 << 15, 4, 5_000.0
    src = tmp_path / "in.cf32"
    _tone(fs, f1, n).tofile(src)
    single = tmp_path / "single.cf32"
    channelize_to_file(src, single, fs, center_hz=f1, decim=decim)
    monkeypatch.setattr(iqfile, "_SURVEY_SAMPLE_ITEMS", 2048)  # force many blocks
    multi = tmp_path / "multi.cf32"
    channelize_to_file(src, multi, fs, center_hz=f1, decim=decim)
    a = np.fromfile(single, np.complex64)
    b = np.fromfile(multi, np.complex64)
    assert a.size == b.size
    assert np.allclose(a, b, atol=1e-4)


def test_channelize_decim_one_is_a_pure_mixer(tmp_path: Path) -> None:
    fs, n, f1 = 48_000.0, 1 << 15, 4_000.0
    src = tmp_path / "in.cf32"
    _tone(fs, f1, n).tofile(src)
    dst = tmp_path / "out.cf32"
    written, out_rate = channelize_to_file(src, dst, fs, center_hz=f1, decim=1)
    assert out_rate == fs
    assert abs(written - n) <= 1
    y = np.fromfile(dst, np.complex64)
    freqs = np.fft.fftfreq(y.size, d=1.0 / out_rate)
    peak = freqs[int(np.argmax(np.abs(np.fft.fft(y))))]
    assert abs(peak) < 100.0
