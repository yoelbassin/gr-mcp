from pathlib import Path

from marconi.ops.analyze import measure
from marconi.sigmf import write_capture


def test_measure_tone(tmp_path: Path, make_iq) -> None:
    ref = write_capture(
        make_iq([(100e3, 1.0)]),
        tmp_path / "cap",
        center_freq=100e6,
        sample_rate=1e6,
    )
    m = measure(ref, center_freq=100.1e6, search_bandwidth=100e3)
    assert abs(m.center_freq - 100.1e6) < 1e3
    assert m.snr_db > 20
    # a pure tone occupies very little bandwidth
    assert m.occupied_bw_99 < 10e3


def test_measure_occupied_bandwidth_of_band(tmp_path: Path, make_iq) -> None:
    # a comb of equal tones spanning 50 kHz: the 99% occupied bandwidth should
    # land near that span, not the near-zero of a single tone. The single-tone
    # tests never exercise the OBW containment math; this does.
    tones = [(float(f), 1.0) for f in range(-25_000, 25_001, 5_000)]  # 11 tones
    ref = write_capture(
        make_iq(tones), tmp_path / "cap", center_freq=100e6, sample_rate=1e6
    )
    m = measure(ref, center_freq=100e6, search_bandwidth=200e3)
    assert 40e3 < m.occupied_bw_99 < 60e3


def test_measure_uses_search_window(tmp_path: Path, make_iq) -> None:
    # two tones; measuring around one must not report the other
    ref = write_capture(
        make_iq([(100e3, 0.5), (-200e3, 1.0)]),
        tmp_path / "cap",
        center_freq=100e6,
        sample_rate=1e6,
    )
    m = measure(ref, center_freq=100.1e6, search_bandwidth=100e3)
    assert abs(m.center_freq - 100.1e6) < 1e3
    assert m.snr_db > 20
    m2 = measure(ref, center_freq=99.8e6, search_bandwidth=100e3)
    assert m2.power_db > m.power_db
