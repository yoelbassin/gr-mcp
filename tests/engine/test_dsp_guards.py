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


def test_resolved_ser_rejects_max_shift_at_or_below_settle() -> None:
    from engine._dsp import resolved_ser

    rx = np.zeros(64, dtype=np.complex64)
    tx = np.zeros(64, dtype=int)
    pts = np.array([1 + 0j, -1 + 0j])
    with pytest.raises(ValueError, match="max_shift"):
        resolved_ser(rx, tx, pts, M=2, max_shift=64, settle=64)


def test_resolved_ser_hard_rejects_max_shift_at_or_below_settle() -> None:
    from engine._dsp import resolved_ser_hard

    rx = np.zeros(64, dtype=int)
    tx = np.zeros(64, dtype=int)
    pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    with pytest.raises(ValueError, match="max_shift"):
        resolved_ser_hard(rx, tx, pts, settle=100, max_shift=100)


def test_aligned_ber_raises_when_lag_exceeds_window() -> None:
    from engine._dsp import AlignmentNotFound, aligned_ber

    rng = np.random.default_rng(3)
    tx = rng.integers(0, 2, 4096).astype(np.uint8)
    rx = np.concatenate([np.zeros(600, dtype=np.uint8), tx])
    with pytest.raises(AlignmentNotFound, match="max_shift"):
        aligned_ber(rx, tx, max_shift=64)


def test_aligned_ber_still_returns_on_lock() -> None:
    from engine._dsp import aligned_ber

    rng = np.random.default_rng(3)
    tx = rng.integers(0, 2, 4096).astype(np.uint8)
    rx = np.concatenate([np.zeros(32, dtype=np.uint8), tx])
    assert aligned_ber(rx, tx, max_shift=64) == 0.0


def test_aligned_ber_best_raises_when_no_candidate_aligns() -> None:
    from engine._dsp import AlignmentNotFound, aligned_ber_best

    rng = np.random.default_rng(3)
    tx = rng.integers(0, 2, 4096).astype(np.uint8)
    rx = np.concatenate([np.zeros(600, dtype=np.uint8), tx])
    with pytest.raises(AlignmentNotFound, match="hypothesis"):
        aligned_ber_best([rx, 1 - rx], tx, max_shift=64)


def test_aligned_ber_best_returns_the_locking_candidate() -> None:
    from engine._dsp import aligned_ber_best

    rng = np.random.default_rng(3)
    tx = rng.integers(0, 2, 4096).astype(np.uint8)
    rx = np.concatenate([np.zeros(32, dtype=np.uint8), tx])
    assert aligned_ber_best([1 - rx, rx], tx, max_shift=64) == 0.0


def test_fsk_and_ook_accept_open_loop_reject_negative() -> None:
    from pydantic import ValidationError

    from marconi.engine.modulation.fsk.stages import _FskParams
    from marconi.engine.modulation.ook.stages import _OokParams
    from marconi.engine.types.params import OPEN_LOOP

    assert _FskParams(deviation=2400.0, loop_bw=OPEN_LOOP).loop_bw == 0.0
    assert _OokParams(loop_bw=OPEN_LOOP).loop_bw == 0.0
    with pytest.raises(ValidationError):
        _FskParams(deviation=2400.0, loop_bw=-0.01)
    with pytest.raises(ValidationError):
        _OokParams(loop_bw=-0.01)
