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


def test_tone_near_nyquist_single_signal(tmp_path: Path, make_iq) -> None:
    """A tone at +499 kHz (near fs/2) must be reported as exactly one signal.

    Spectral leakage wraps power across the ±fs/2 boundary, which previously
    split the tone into two groups (one at array index 0, one at index N-1).
    The merged signal's center should be within 3 kHz of the true frequency.
    """
    ref = write_capture(
        make_iq([(499e3, 1.0)]),
        tmp_path / "cap",
        center_freq=100e6,
        sample_rate=1e6,
    )
    signals = find_signals(ref)
    assert (
        len(signals) == 1
    ), f"expected 1 signal for near-Nyquist tone, got {len(signals)}: {signals}"
    assert (
        abs(signals[0].center_freq - 100.499e6) < 1e3
    ), f"center_freq {signals[0].center_freq:.1f} Hz not within 1 kHz of 100.499 MHz"


def test_two_distinct_tones_near_opposite_edges_stay_separate(
    tmp_path: Path, make_iq
) -> None:
    """Two co-equal tones near +fs/2 and -fs/2 must NOT be merged into one.

    Both touch the wrap edges (index 0 and N-1), but their comparable peak
    powers distinguish them from a single straddling tone (one strong lobe +
    weak wrapped leakage). Regression for the false-merge edge case.
    """
    ref = write_capture(
        make_iq([(499e3, 1.0), (-499e3, 1.0)]),
        tmp_path / "cap",
        center_freq=100e6,
        sample_rate=1e6,
    )
    signals = find_signals(ref)
    assert (
        len(signals) == 2
    ), f"expected 2 distinct edge tones, got {len(signals)}: {signals}"
    centers = sorted(s.center_freq for s in signals)
    assert abs(centers[0] - 99.501e6) < 2e3
    assert abs(centers[1] - 100.499e6) < 2e3


def test_tone_near_negative_nyquist_single_signal(tmp_path: Path, make_iq) -> None:
    """A tone at -499 kHz (near -fs/2) must be reported as exactly one signal.

    This exercises the else branch of the wrap-around merge, where the dominant
    peak is at the low-frequency edge (first_g).  The misalignment bug produced
    a wildly wrong center frequency before the fix.
    """
    ref = write_capture(
        make_iq([(-499e3, 1.0)]),
        tmp_path / "cap",
        center_freq=100e6,
        sample_rate=1e6,
    )
    signals = find_signals(ref)
    assert len(signals) == 1, (
        f"expected 1 signal for near-negative-Nyquist tone, "
        f"got {len(signals)}: {signals}"
    )
    assert (
        abs(signals[0].center_freq - 99.501e6) < 1e3
    ), f"center_freq {signals[0].center_freq:.1f} Hz not within 1 kHz of 99.501 MHz"
