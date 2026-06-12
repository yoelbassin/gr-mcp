from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from marconi.ops.capture import load_capture
from marconi.sigmf import read_capture, write_capture
from marconi.workspace import Workspace


def test_load_sigmf_in_place(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    src = write_capture(
        make_iq([]), tmp_path / "ext", center_freq=433e6, sample_rate=1e6
    )
    ref = load_capture(src.path, ws)
    assert ref == src  # referenced in place, not copied


def test_load_raw_cf32(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    samples = make_iq([(10e3, 1.0)])
    raw = tmp_path / "ext.cf32"
    samples.tofile(raw)

    ref = load_capture(raw, ws, sample_rate=1e6, center_freq=433e6)
    assert ref.path.is_relative_to(ws.root / "captures")
    loaded, _ = read_capture(ref.path)
    np.testing.assert_array_equal(loaded, samples)
    assert ref.sample_rate == 1e6


def test_load_raw_cf32_requires_sample_rate(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    raw = tmp_path / "ext.cf32"
    make_iq([]).tofile(raw)
    with pytest.raises(ValueError, match="sample_rate"):
        load_capture(raw, ws)


def test_load_stereo_wav_as_iq(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "project")
    rate = 48000
    t = np.arange(4800) / rate
    i = (np.cos(2 * np.pi * 1000 * t) * 30000).astype(np.int16)
    q = (np.sin(2 * np.pi * 1000 * t) * 30000).astype(np.int16)
    wav = tmp_path / "ext.wav"
    wavfile.write(wav, rate, np.stack([i, q], axis=1))

    ref = load_capture(wav, ws)
    assert ref.sample_rate == rate
    loaded, _ = read_capture(ref.path)
    assert loaded.dtype == np.complex64
    assert len(loaded) == 4800
    # normalized to <= 1.0 magnitude
    assert np.abs(loaded).max() <= 1.0
