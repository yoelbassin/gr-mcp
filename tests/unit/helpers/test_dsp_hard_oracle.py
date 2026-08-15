from pathlib import Path

import numpy as np
import pytest
from helpers._dsp import (
    AlignmentNotFound,
    PayloadTruncated,
    channel,
    resolved_ser,
    resolved_ser_hard,
)


def _iq_file(tmp_path: Path, n: int = 2048) -> Path:
    path = tmp_path / "in.cf32"
    np.ones(n, dtype=np.complex64).tofile(path)
    return path


def _qam16_points() -> np.ndarray:
    from gnuradio import digital  # in-process GR for oracle ground truth (allowed)

    return np.asarray(digital.constellation_16qam().points())


def _relabel_90(points: np.ndarray) -> np.ndarray:
    # index i, when the constellation is rotated 90 deg, is read as this index
    return np.argmin(np.abs((points * 1j)[:, None] - points[None, :]), axis=1)


def test_zero_on_rotated_aligned_stream() -> None:
    pts = _qam16_points()
    rng = np.random.default_rng(0)
    tsi = rng.integers(0, 16, 4000)
    settle = 1500
    # receiver locked to the 90 deg-rotated frame: rx index = relabel_90[true]
    rx = np.concatenate([rng.integers(0, 16, settle), _relabel_90(pts)[tsi]])
    assert resolved_ser_hard(rx, tsi, pts, settle=settle) == 0.0


def test_high_on_random_stream() -> None:
    pts = _qam16_points()
    rng = np.random.default_rng(1)
    tsi = rng.integers(0, 16, 4000)
    rx = rng.integers(0, 16, 4000 + 1500)
    with pytest.raises(AlignmentNotFound):
        resolved_ser_hard(rx, tsi, pts, settle=1500)


def test_max_shift_must_exceed_settle() -> None:
    # the de-risk trap: rx[n] aligns with tx[n], so trimming `settle` needs a
    # shift of `settle` to realign. A max_shift <= settle reads clean as random.
    pts = _qam16_points()
    rng = np.random.default_rng(2)
    tsi = rng.integers(0, 16, 4000)
    rx = _relabel_90(pts)[tsi]
    with pytest.raises(ValueError, match="max_shift"):
        resolved_ser_hard(rx, tsi, pts, settle=1500, max_shift=100)
    assert resolved_ser_hard(rx, tsi, pts, settle=1500) == 0.0  # default settle+700


def test_channel_sfo_noop_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no-op"):
        channel(_iq_file(tmp_path), tmp_path / "out.cf32", sfo_ppm=50.0)


def test_channel_sfo_effective_changes_length(tmp_path: Path) -> None:
    out = channel(_iq_file(tmp_path), tmp_path / "out.cf32", sfo_ppm=500.0)
    assert len(np.fromfile(out, dtype=np.complex64)) == 2049


def _bpsk_stream(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.array([-1 + 0j, 1 + 0j])
    tsi = np.random.default_rng(seed).integers(0, 2, n)
    return pts[tsi], tsi, pts


def test_resolved_ser_refuses_a_silently_truncated_payload() -> None:
    """The laundering the tx-relative floor exists to stop: a PERFECT decode of
    50 of 4096 payload symbols scored SER 0.0 under the old rx-relative floor,
    so a path that dropped 99% of its payload read as flawless."""
    full, tsi, pts = _bpsk_stream(4096, 0)
    rx = np.concatenate([np.zeros(64, dtype=complex), full])
    assert resolved_ser(rx, tsi, pts, M=2) == 0.0
    with pytest.raises(PayloadTruncated, match="payload, not the alignment"):
        resolved_ser(rx[: 64 + 50], tsi, pts, M=2)


def test_resolved_ser_hard_refuses_a_silently_truncated_payload() -> None:
    pts = _qam16_points()
    rng = np.random.default_rng(4)
    tsi = rng.integers(0, 16, 6000)
    rx = np.concatenate([rng.integers(0, 16, 1500), tsi])
    assert resolved_ser_hard(rx, tsi, pts, settle=1500) == 0.0
    with pytest.raises(PayloadTruncated, match="payload, not the alignment"):
        resolved_ser_hard(rx[: 1500 + 50], tsi, pts, settle=1500)


def test_the_overlap_floor_still_admits_a_real_streaming_tail_loss() -> None:
    """The other half of the bar: real paths lose a tail to group delay (every
    consumer of these oracles measures 0.9973-0.9995 payload coverage), so a
    floor that refused those would be a gate nothing could pass. A stream cut
    to 80% of the reachable payload still scores."""
    full, tsi, pts = _bpsk_stream(4096, 5)
    keep = int(0.80 * 4096)
    rx = np.concatenate([np.zeros(64, dtype=complex), full[:keep]])
    assert resolved_ser(rx, tsi, pts, M=2) == 0.0


def test_aligned_ber_names_truncation_rather_than_lost_alignment() -> None:
    from helpers._dsp import aligned_ber

    tx = np.random.default_rng(6).integers(0, 2, 4096).astype(np.uint8)
    with pytest.raises(PayloadTruncated, match="payload, not the alignment"):
        aligned_ber(tx[:50], tx)


def test_resolved_ser_rejects_max_shift_at_or_below_settle() -> None:
    rx = np.zeros(64, dtype=np.complex64)
    tx = np.zeros(64, dtype=int)
    pts = np.array([1 + 0j, -1 + 0j])
    with pytest.raises(ValueError, match="max_shift"):
        resolved_ser(rx, tx, pts, M=2, max_shift=64, settle=64)


def test_resolved_ser_hard_rejects_max_shift_at_or_below_settle() -> None:
    rx = np.zeros(64, dtype=int)
    tx = np.zeros(64, dtype=int)
    pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    with pytest.raises(ValueError, match="max_shift"):
        resolved_ser_hard(rx, tx, pts, settle=100, max_shift=100)


def test_aligned_ber_raises_when_lag_exceeds_window() -> None:
    from helpers._dsp import AlignmentNotFound, aligned_ber

    rng = np.random.default_rng(3)
    tx = rng.integers(0, 2, 4096).astype(np.uint8)
    rx = np.concatenate([np.zeros(600, dtype=np.uint8), tx])
    with pytest.raises(AlignmentNotFound, match="max_shift"):
        aligned_ber(rx, tx, max_shift=64)


def test_aligned_ber_still_returns_on_lock() -> None:
    from helpers._dsp import aligned_ber

    rng = np.random.default_rng(3)
    tx = rng.integers(0, 2, 4096).astype(np.uint8)
    rx = np.concatenate([np.zeros(32, dtype=np.uint8), tx])
    assert aligned_ber(rx, tx, max_shift=64) == 0.0


def test_aligned_ber_best_raises_when_no_candidate_aligns() -> None:
    from helpers._dsp import AlignmentNotFound, aligned_ber_best

    rng = np.random.default_rng(3)
    tx = rng.integers(0, 2, 4096).astype(np.uint8)
    rx = np.concatenate([np.zeros(600, dtype=np.uint8), tx])
    with pytest.raises(AlignmentNotFound, match="hypothesis"):
        aligned_ber_best([rx, 1 - rx], tx, max_shift=64)


def test_aligned_ber_best_returns_the_locking_candidate() -> None:
    from helpers._dsp import aligned_ber_best

    rng = np.random.default_rng(3)
    tx = rng.integers(0, 2, 4096).astype(np.uint8)
    rx = np.concatenate([np.zeros(32, dtype=np.uint8), tx])
    assert aligned_ber_best([1 - rx, rx], tx, max_shift=64) == 0.0
