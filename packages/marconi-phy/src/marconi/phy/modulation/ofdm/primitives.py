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
    """Sample index just after the first contiguous low-energy (null) run."""
    p = np.abs(x) ** 2
    env = np.convolve(p, np.ones(win) / win, mode="same")
    low = env < 0.25 * np.median(env)
    i, n = 0, len(x)
    while i < n - frame_len - sym_len:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if (j - i) > null_len * 0.5:
                return j
            i = j
        else:
            i += 1
    raise ValueError("no null symbol found")


def qpsk_lock(z: np.ndarray) -> float:
    """|E[(z/|z|)^4]| — 1.0 on four clean clusters, conjugation-invariant."""
    z = z[np.abs(z) > 0]
    return float(np.abs(np.mean((z / np.abs(z)) ** 4)))
