import json
from pathlib import Path

import numpy as np

from marconi.sigmf import read_capture, write_capture


def test_write_read_roundtrip(tmp_path: Path, make_iq) -> None:
    samples = make_iq([(100e3, 1.0)])
    ref = write_capture(samples, tmp_path / "cap", center_freq=433e6, sample_rate=1e6)

    assert ref.path == tmp_path / "cap.sigmf-data"
    assert ref.path.exists()
    assert (tmp_path / "cap.sigmf-meta").exists()
    assert ref.num_samples == len(samples)
    assert ref.center_freq == 433e6

    loaded, ref2 = read_capture(ref.path)
    np.testing.assert_array_equal(loaded, samples)
    assert ref2 == ref


def test_meta_is_valid_sigmf(tmp_path: Path, make_iq) -> None:
    write_capture(make_iq([]), tmp_path / "cap", center_freq=1e9, sample_rate=2e6)
    meta = json.loads((tmp_path / "cap.sigmf-meta").read_text())
    assert meta["global"]["core:datatype"] == "cf32_le"
    assert meta["global"]["core:sample_rate"] == 2e6
    assert meta["captures"][0]["core:frequency"] == 1e9


def test_read_accepts_meta_or_base_path(tmp_path: Path, make_iq) -> None:
    samples = make_iq([])
    write_capture(samples, tmp_path / "cap", center_freq=0.0, sample_rate=1e6)
    for p in (tmp_path / "cap.sigmf-meta", tmp_path / "cap"):
        loaded, ref = read_capture(p)
        assert ref.num_samples == len(samples)
