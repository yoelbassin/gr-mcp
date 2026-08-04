from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.engine.io.iqfile import (
    CaptureTooShort,
    iter_iq,
    sample_iq,
    slice_len,
)


def _write(path: Path, x: np.ndarray) -> None:
    x.astype(np.complex64).tofile(path)


def test_sample_iq_reads_whole_small_slice(tmp_path: Path) -> None:
    x = (np.arange(1 << 14) + 1j * np.arange(1 << 14)).astype(np.complex64)
    p = tmp_path / "s.cf32"
    _write(p, x)
    got, analyzed, span = sample_iq(p, 0, 0)
    assert span == x.size and analyzed == x.size
    np.testing.assert_array_equal(got, x)


def test_sample_iq_strides_when_over_budget(tmp_path: Path) -> None:
    x = (np.arange(3 << 20) + 0j).astype(np.complex64)
    p = tmp_path / "big.cf32"
    _write(p, x)
    got, analyzed, span = sample_iq(p, 0, 0, budget=1 << 20)
    assert span == x.size
    assert got.size == analyzed <= (1 << 20)
    assert got.dtype == np.complex64


def test_iter_iq_reconstructs_offset_slice(tmp_path: Path) -> None:
    x = (np.arange(5000) + 1j).astype(np.complex64)
    p = tmp_path / "s.cf32"
    _write(p, x)
    out = np.concatenate(list(iter_iq(p, offset=100, length=2000, chunk=512)))
    np.testing.assert_array_equal(out, x[100:2100])
    assert slice_len(p, 100, 2000) == 2000


def test_too_short_raises(tmp_path: Path) -> None:
    p = tmp_path / "tiny.cf32"
    _write(p, np.ones(16, np.complex64))
    with pytest.raises(CaptureTooShort):
        sample_iq(p, 0, 0)
