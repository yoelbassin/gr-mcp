"""GR-free OFDM numpy DSP. Generic — protocol values are parameters."""

from __future__ import annotations

import numpy as np


def find_null(
    x: np.ndarray,
    *,
    null_len: int,
    frame_len: int,
    sym_len: int,
    win: int = 512,
) -> int:
    """Sample index just after the first contiguous low-energy (null) run.

    ``win`` is clamped to ``null_len`` so the envelope smoother doesn't wash
    out the null in short test buffers (where null_len << default 512).
    After finding the null region via the smoothed envelope, the boundary is
    refined on the raw power so the return value is the exact first high-power
    sample (not the smeared envelope crossing).
    """
    p = np.abs(x) ** 2
    w = min(win, null_len)
    env = np.convolve(p, np.ones(w) / w, mode="same")
    thresh = 0.25 * np.median(env)
    low = env < thresh
    # run detection in one pass - the acquisition ladder re-calls this per
    # frame advance, so a per-sample Python scan multiplies into whole-capture
    # Python time on no-signal input
    padded = np.concatenate(([False], low, [False]))
    edges = np.flatnonzero(np.diff(padded.view(np.int8)))
    starts, ends = edges[0::2], edges[1::2]
    qualifying = np.flatnonzero((ends - starts) > null_len * 0.5)
    if qualifying.size == 0:
        raise ValueError("no null symbol found")
    # Refine: scan forward in raw power from (run end - w) to find the exact
    # first high-energy sample, bypassing convolution smear.
    refined = max(0, int(ends[qualifying[0]]) - w)
    above = np.flatnonzero(p[refined:] > thresh)
    if above.size == 0:
        # Buffer ends inside the null — need more data.
        raise ValueError("no null symbol found")
    return refined + int(above[0])


def qpsk_lock(z: np.ndarray) -> float:
    """|E[(z/|z|)^4]| — 1.0 on four clean clusters, conjugation-invariant."""
    z = z[np.abs(z) > 0]
    return float(np.abs(np.mean((z / np.abs(z)) ** 4)))
