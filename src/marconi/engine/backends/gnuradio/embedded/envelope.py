from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d


def sustained_runs(
    mask: npt.NDArray[np.bool_], run: int
) -> npt.NDArray[np.signedinteger[Any]]:
    """Start indices of every maximal run of >= `run` consecutive True
    values in `mask`, via a vectorized sliding-window sum (cumsum
    difference) - the idle-span hunt is bulk, so this must not be a
    per-sample Python loop the way the (rare, short) in-burst scan is."""
    if run <= 1:
        return np.flatnonzero(mask)
    if len(mask) < run:
        return np.empty(0, dtype=np.int64)
    csum = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    window = csum[run:] - csum[:-run]
    return np.flatnonzero(window >= run)


def trailing_mean(
    block: npt.NDArray[np.floating[Any]], window: int
) -> npt.NDArray[np.floating[Any]]:
    """Trailing moving average, same length as `block` (front-padded with
    `block[0]` so the first few samples of each fixed-size processing block
    do not see a false dip toward zero from implicit zero-padding)."""
    if window <= 1:
        return block
    smoothed: npt.NDArray[np.float64] = uniform_filter1d(
        block.astype(np.float64), window, mode="nearest", origin=(window - 1) // 2
    )
    return smoothed
