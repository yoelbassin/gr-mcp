from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from engine._dsp import channel


def _iq_file(tmp_path: Path, n: int = 2048) -> Path:
    path = tmp_path / "in.cf32"
    np.ones(n, dtype=np.complex64).tofile(path)
    return path


def test_channel_sfo_noop_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no-op"):
        channel(_iq_file(tmp_path), tmp_path / "out.cf32", sfo_ppm=50.0)


def test_channel_sfo_effective_changes_length(tmp_path: Path) -> None:
    out = channel(_iq_file(tmp_path), tmp_path / "out.cf32", sfo_ppm=500.0)
    assert len(np.fromfile(out, dtype=np.complex64)) == 2049
