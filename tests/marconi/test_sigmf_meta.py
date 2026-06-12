from pathlib import Path

import numpy as np

from marconi.sigmf import read_meta, read_samples, write_capture, write_meta_for


def test_read_meta_does_not_read_samples(tmp_path: Path, make_iq) -> None:
    samples = make_iq([])
    ref = write_capture(samples, tmp_path / "cap", center_freq=1e9, sample_rate=2e6)
    meta = read_meta(ref.path)
    assert meta == ref  # num_samples derived from data file size


def test_read_samples_matches_written(tmp_path: Path, make_iq) -> None:
    samples = make_iq([(10e3, 1.0)])
    ref = write_capture(samples, tmp_path / "cap", center_freq=0.0, sample_rate=1e6)
    np.testing.assert_array_equal(read_samples(ref), samples)


def test_write_meta_for_existing_raw_file(tmp_path: Path, make_iq) -> None:
    samples = make_iq([])
    data_path = tmp_path / "x.sigmf-data"
    samples.tofile(data_path)
    ref = write_meta_for(data_path, center_freq=433e6, sample_rate=1e6)
    assert ref.path == data_path
    assert ref.num_samples == len(samples)
    assert read_meta(data_path) == ref
