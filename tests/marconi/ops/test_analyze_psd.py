from pathlib import Path

import numpy as np

from marconi.ops.analyze import psd
from marconi.sigmf import write_capture


def test_psd_finds_tone_at_absolute_freq(tmp_path: Path, make_iq) -> None:
    # tone at +100 kHz offset, center 433 MHz -> absolute 433.1 MHz
    ref = write_capture(
        make_iq([(100e3, 1.0)]),
        tmp_path / "cap",
        center_freq=433e6,
        sample_rate=1e6,
    )
    result = psd(ref)

    assert len(result.freqs) == len(result.psd_db)
    assert len(result.peaks) >= 1
    strongest = max(result.peaks, key=lambda p: p.power_db)
    assert abs(strongest.freq - 433.1e6) < 1e3
    assert strongest.power_db > result.noise_floor_db + 20


def test_psd_rejects_empty_capture(tmp_path: Path) -> None:
    import pytest

    ref = write_capture(
        np.array([], dtype=np.complex64),
        tmp_path / "empty",
        center_freq=433e6,
        sample_rate=1e6,
    )
    with pytest.raises(ValueError, match="too short"):
        psd(ref)


def test_psd_noise_floor_sane(tmp_path: Path, make_iq) -> None:
    ref = write_capture(
        make_iq([], noise_amplitude=0.01),
        tmp_path / "cap",
        center_freq=0.0,
        sample_rate=1e6,
    )
    result = psd(ref)
    # pure noise: no strong peaks, floor near median power
    psd_arr = np.array(result.psd_db)
    assert abs(result.noise_floor_db - float(np.median(psd_arr))) < 1.0
    assert result.peaks == []
