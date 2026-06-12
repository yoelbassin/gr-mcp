from pathlib import Path

from marconi.ops.analyze import find_signals
from marconi.sigmf import write_capture


def test_finds_two_tones(tmp_path: Path, make_iq) -> None:
    ref = write_capture(
        make_iq([(100e3, 1.0), (-200e3, 0.5)]),
        tmp_path / "cap",
        center_freq=100e6,
        sample_rate=1e6,
    )
    signals = find_signals(ref)
    assert len(signals) == 2
    signals.sort(key=lambda s: s.center_freq)
    assert abs(signals[0].center_freq - 99.8e6) < 2e3
    assert abs(signals[1].center_freq - 100.1e6) < 2e3
    assert all(s.snr_db > 20 for s in signals)
    # stronger tone reported stronger
    assert signals[1].peak_power_db > signals[0].peak_power_db


def test_pure_noise_finds_nothing(tmp_path: Path, make_iq) -> None:
    ref = write_capture(make_iq([]), tmp_path / "cap", center_freq=0.0, sample_rate=1e6)
    assert find_signals(ref) == []
