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
    i, n = 0, len(x)
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if (j - i) > null_len * 0.5:
                # Refine: scan forward in raw power from (j - w) to find exact
                # first high-energy sample, bypassing convolution smear.
                refined = max(0, j - w)
                while refined < n and p[refined] <= thresh:
                    refined += 1
                return refined
            i = j
        else:
            i += 1
    raise ValueError("no null symbol found")


def qpsk_lock(z: np.ndarray) -> float:
    """|E[(z/|z|)^4]| — 1.0 on four clean clusters, conjugation-invariant."""
    z = z[np.abs(z) > 0]
    return float(np.abs(np.mean((z / np.abs(z)) ** 4)))
