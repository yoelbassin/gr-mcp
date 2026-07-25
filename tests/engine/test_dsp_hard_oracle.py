import numpy as np
import pytest
from engine._dsp import AlignmentNotFound, resolved_ser_hard


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
