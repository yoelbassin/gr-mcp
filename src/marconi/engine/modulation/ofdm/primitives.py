"""GR-free OFDM numpy DSP. Generic — protocol values are parameters."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d

# Lock threshold for cp_symbol_sync's CP-correlation ratio. Measured: noise
# 1.3-1.6, real lock >= 2.0. Lives here because both the stage default and the
# block's own fallback must move together.
LOCK_MIN_RATIO_DEFAULT = 2.0


def step_boundary(
    p: npt.NDArray[np.floating[Any]],
    lo: int,
    hi: int,
    *,
    floor: float,
    sig: float,
) -> int | None:
    """Change-point of a low->high power step within ``p[lo:hi]``: the CUSUM
    argmin of the per-sample null-vs-signal log-likelihood ratio, given the
    caller's ``floor`` (null) and ``sig`` (signal) power levels. Every sample
    votes with its likelihood, so noise inside a received null cannot drag the
    boundary the way a single-sample threshold crossing does; on a clean null
    the boundary is the exact first nonzero sample."""
    seg = p[lo:hi]
    if seg.size < 2 or sig <= 0.0 or floor >= sig:
        return None
    floor = max(floor, 1e-12 * sig)
    llr = math.log(floor / sig) + seg * (1.0 / floor - 1.0 / sig)
    edge = int(np.argmin(np.cumsum(llr))) + 1
    return None if edge >= seg.size else lo + edge


def find_null(
    x: npt.NDArray[np.complexfloating[Any, Any]],
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
    a step fit on raw power around the envelope crossing — exact on clean
    nulls, unbiased when the null is filled with receiver noise.
    """
    p = np.abs(x) ** 2
    w = min(win, null_len)
    env = uniform_filter1d(p, w, mode="constant", cval=0.0)
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
    run_start = int(starts[qualifying[0]])
    run_end = int(ends[qualifying[0]])
    # the env-low run sits strictly inside the null, so its mean is the null
    # floor; everything outside it estimates the signal level
    run_sum = float(np.sum(p[run_start:run_end]))
    floor = run_sum / (run_end - run_start)
    rest = (float(np.sum(p)) - run_sum) / max(1, p.size - (run_end - run_start))
    edge = step_boundary(
        p, max(0, run_end - w), min(p.size, run_end + w), floor=floor, sig=rest
    )
    if edge is None or float(np.mean(p[edge : run_end + w])) <= thresh:
        # Buffer ends inside the null — need more data.
        raise ValueError("no null symbol found")
    return edge


def qpsk_lock(z: npt.NDArray[np.complexfloating[Any, Any]]) -> float:
    """|E[(z/|z|)^4]| — 1.0 on four clean clusters, conjugation-invariant."""
    z = z[np.abs(z) > 0]
    return float(np.abs(np.mean((z / np.abs(z)) ** 4)))
