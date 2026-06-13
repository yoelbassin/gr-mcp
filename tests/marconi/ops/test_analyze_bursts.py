from pathlib import Path

import numpy as np
import pytest

from marconi.ops.analyze import detect_bursts
from marconi.sigmf import write_capture


def _bursty_signal(
    fs: float = 1e6, total: float = 0.05, on_start: float = 0.01, on_dur: float = 0.01
) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(fs * total)
    x = (rng.normal(0, 0.01, n) + 1j * rng.normal(0, 0.01, n)).astype(np.complex64)
    t = np.arange(n) / fs
    tone = np.exp(2j * np.pi * 50e3 * t).astype(np.complex64)
    on = slice(int(on_start * fs), int((on_start + on_dur) * fs))
    x[on] += tone[on]
    return x


def test_detects_single_burst(tmp_path: Path) -> None:
    ref = write_capture(
        _bursty_signal(), tmp_path / "cap", center_freq=0.0, sample_rate=1e6
    )
    bursts = detect_bursts(ref)
    assert len(bursts) == 1
    b = bursts[0]
    assert abs(b.start_time - 0.01) < 2e-3
    assert abs(b.duration - 0.01) < 4e-3


def test_continuous_signal_is_not_bursty(tmp_path: Path, make_iq) -> None:
    ref = write_capture(
        make_iq([(50e3, 1.0)]), tmp_path / "cap", center_freq=0.0, sample_rate=1e6
    )
    # always-on signal: power never drops below threshold -> no distinct bursts
    assert detect_bursts(ref) == []


def test_too_short_capture_raises(tmp_path: Path) -> None:
    """A 1-sample capture must raise ValueError mentioning 'too short'."""
    x = np.array([1 + 0j], dtype=np.complex64)
    ref = write_capture(x, tmp_path / "cap", center_freq=0.0, sample_rate=1e6)
    with pytest.raises(ValueError, match="too short"):
        detect_bursts(ref)


def test_60pct_duty_cycle_detected_as_single_burst(tmp_path: Path) -> None:
    """A 60% duty-cycle burst (on 0.01–0.04 s in a 0.05 s capture) is one burst."""
    # on_dur = 0.03 s out of 0.05 s total → 60% duty cycle
    x = _bursty_signal(on_start=0.01, on_dur=0.03)
    ref = write_capture(x, tmp_path / "cap", center_freq=0.0, sample_rate=1e6)
    bursts = detect_bursts(ref)
    assert len(bursts) == 1
    b = bursts[0]
    assert abs(b.start_time - 0.01) < 2e-3
    assert abs(b.duration - 0.03) < 4e-3
