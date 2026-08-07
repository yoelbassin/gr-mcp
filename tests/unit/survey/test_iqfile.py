from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.survey.iqfile import (
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


def test_sample_iq_reads_contiguous_active_window_over_budget(tmp_path: Path) -> None:
    # A low-power lead then a strong contiguous burst. Over budget, sample_iq
    # must return ONE contiguous window that lands on the burst -- never 16
    # stitched chunks (which would fragment the burst into short runs).
    span = 3 << 20
    burst_len = 1 << 19
    amp = np.full(span, 0.01, np.float64)
    amp[1 << 21 : (1 << 21) + burst_len] = 5.0
    p = tmp_path / "big.cf32"
    _write(p, amp.astype(np.complex64))
    got, analyzed, got_span = sample_iq(p, 0, 0, budget=1 << 20)
    assert got_span == span
    assert got.size == analyzed == (1 << 20)
    assert got.dtype == np.complex64
    # the full burst survives as one unbroken run -> the window is contiguous and
    # sits on the active region, not the low-power lead
    flags = np.concatenate(([0], (got.real == 5.0).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(flags))
    assert int((edges[1::2] - edges[0::2]).max()) == burst_len


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


def test_negative_offset_raises(tmp_path: Path) -> None:
    p = tmp_path / "s.cf32"
    _write(p, np.ones(16, np.complex64))
    with pytest.raises(ValueError):
        slice_len(p, -1, 0)
