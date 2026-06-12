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
