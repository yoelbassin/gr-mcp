import json
from pathlib import Path

import numpy as np
import pytest

from marconi.sigmf import read_capture, read_meta, read_samples, write_capture


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


def test_read_meta_rejects_malformed_meta(tmp_path: Path) -> None:
    # meta missing the 'global' block -> clear ValueError, not a bare KeyError
    # the MCP boundary would otherwise mislabel as "not_found".
    (tmp_path / "cap.sigmf-data").write_bytes(b"")
    (tmp_path / "cap.sigmf-meta").write_text(json.dumps({"captures": [{}]}))
    with pytest.raises(ValueError, match="malformed SigMF metadata"):
        read_meta(tmp_path / "cap")


def test_read_capture_routes_through_read_samples(tmp_path: Path, make_iq) -> None:
    """read_capture must read via read_samples (the single reader) so both
    return identical data and honor CaptureRef.datatype."""
    samples = make_iq([(50e3, 1.0)])
    write_capture(samples, tmp_path / "cap", center_freq=0.0, sample_rate=1e6)
    loaded, ref = read_capture(tmp_path / "cap")
    np.testing.assert_array_equal(loaded, read_samples(ref))
    assert ref.datatype == "cf32_le"


def test_read_accepts_meta_or_base_path(tmp_path: Path, make_iq) -> None:
    samples = make_iq([])
    write_capture(samples, tmp_path / "cap", center_freq=0.0, sample_rate=1e6)
    for p in (
        tmp_path / "cap.sigmf-meta",
        tmp_path / "cap",
        tmp_path / "cap.sigmf-data",
    ):
        loaded, ref = read_capture(p)
        assert ref.num_samples == len(samples)
