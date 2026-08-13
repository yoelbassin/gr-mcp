"""Stream statistics shared by the quality layer, the survey blocks and the
paging surface: nearest-centre assignment, Lloyd clustering in one and two
dimensions, and the windowed-power gate. Each lived in two or three copies,
which is how three activity gates came to differ in their filter mode without
anyone comparing them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypeVar

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter1d

_KMEANS_ITERS = 100
_SEPARATION_EPS = 1e-9
_LEVEL_KS = (2, 4, 8)
_CLIP_PERCENTILES = (1.0, 99.0)

_N = TypeVar("_N", np.float64, np.complex128)


def nearest_labels(
    values: npt.NDArray[_N], centers: npt.NDArray[_N]
) -> npt.NDArray[np.intp]:
    """Index of the closest centre per value. Real and complex alike — the
    distance is |value - centre| either way."""
    return np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)


def windowed_power(
    x: npt.NDArray[Any],
    window: int,
    *,
    mode: Literal["reflect", "constant"] = "reflect",
) -> npt.NDArray[np.float64]:
    """Boxcar-smoothed instantaneous power, the input every activity gate
    thresholds. Callers keep their own bar — a noise-floor multiple, a
    top-decile fraction — because those are different measurements; only the
    smoothing is shared. `mode` is explicit for the same reason: the soft-stream
    gate zero-pads its edges, the burst gates reflect."""
    power: npt.NDArray[np.float64] = uniform_filter1d(
        np.abs(x).astype(np.float64) ** 2, window, mode=mode, cval=0.0
    )
    return power


def percentile_span(
    x: npt.NDArray[np.floating[Any]],
    percentiles: tuple[float, float] = (0.5, 99.5),
    bins: int = 1,
) -> tuple[float, float]:
    """Span covering `percentiles` of x, always wide enough to hold `bins`
    distinct edges in x's OWN dtype. A steady signal's span can be positive
    yet finer than float32 resolves at that magnitude — a clean carrier's
    instantaneous frequency sits hundredths of a Hz wide around tens of kHz —
    and numpy rejects the range outright ("Too many bins for data range")
    rather than returning one narrow cluster."""
    lo, hi = (float(v) for v in np.percentile(x, percentiles))
    if hi <= lo:
        hi = lo + 1.0
    scale = np.asarray(max(abs(lo), abs(hi), 1.0), dtype=x.dtype)
    floor = bins * float(np.spacing(scale))
    if hi - lo < floor:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * floor, mid + 0.5 * floor
    return lo, hi


class Tail(StrEnum):
    """What a histogram does with the samples outside its percentile span.

    The two callers need different answers and used to reach them by writing
    different numpy calls, which read as an accident rather than a choice.
    CLIP keeps every sample, so the counts sum to the number summarized — right
    where a caller may total them (stream_stats). DROP discards them, so the
    end bins are not spiked by the 1% living in the tails — right where the
    counts are peak-picked (survey's inst_freq tone readout), because a clipped
    edge spike is found as a tone that is not there."""

    CLIP = "clip"
    DROP = "drop"


def binned_counts(
    x: npt.NDArray[np.floating[Any]],
    bins: int,
    span: tuple[float, float],
    tail: Tail,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """(counts, edges) over `span`. One home for the tail policy, named at
    every call site."""
    lo, hi = span
    edges = np.linspace(lo, hi, bins + 1)
    data = np.clip(x, lo, hi) if tail is Tail.CLIP else x
    counts, _ = np.histogram(data, bins=edges)
    return counts.astype(np.int64), edges


def _clip_range(x: npt.NDArray[np.float64]) -> tuple[float, float]:
    return percentile_span(x, _CLIP_PERCENTILES)


def _seeded_lloyd_1d(
    xc: npt.NDArray[np.float64], k: int, lo: float, hi: float
) -> npt.NDArray[np.float64]:
    if k == 1:
        return np.array([float(xc.mean())])
    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1] * 100.0
    centers = np.unique(np.percentile(xc, quantiles))
    if centers.size < k:
        centers = np.linspace(lo, hi, k)
    for _ in range(_KMEANS_ITERS):
        labels = nearest_labels(xc, centers)
        moved = centers.copy()
        for j in range(centers.size):
            members = xc[labels == j]
            if members.size:
                moved[j] = float(members.mean())
        if np.allclose(moved, centers):
            break
        centers = moved
    return np.sort(centers)


def kmeans_1d(x: npt.ArrayLike, k: int) -> npt.NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or k < 1:
        return np.zeros(0, dtype=np.float64)
    k = min(k, x.size)
    lo, hi = _clip_range(x)
    xc = np.clip(x, lo, hi)
    return _seeded_lloyd_1d(xc, k, lo, hi)


def _farthest_point_seeds(
    points: npt.NDArray[np.complex128], k: int
) -> npt.NDArray[np.complex128]:
    # deterministic greedy k-means++ (no RNG): each seed is the point farthest
    # from every seed chosen so far. Data-driven, not a fixed angular grid, so
    # it has no unlucky-rotation basin (a grid seeded exactly between a
    # constellation's lobes can converge with two lobes merged into one seed).
    seeds = np.empty(k, dtype=np.complex128)
    seeds[0] = points[np.argmax(np.abs(points - points.mean()))]
    dist = np.abs(points - seeds[0])
    for i in range(1, k):
        seeds[i] = points[np.argmax(dist)]
        dist = np.minimum(dist, np.abs(points - seeds[i]))
    return seeds


def kmeans_2d(points: npt.NDArray[np.complex128], k: int) -> npt.NDArray[np.complex128]:
    """Constellation clustering: Lloyd from farthest-point seeds. The 1-D
    sibling seeds from quantiles instead — a real level ladder is ordered and a
    constellation is not — but the iteration is the same one."""
    centers = _farthest_point_seeds(points, k)
    for _ in range(_KMEANS_ITERS):
        labels = nearest_labels(points, centers)
        moved = np.array(
            [
                points[labels == j].mean() if np.any(labels == j) else centers[j]
                for j in range(k)
            ]
        )
        if np.allclose(moved, centers):
            break
        centers = moved
    return centers


@dataclass(frozen=True)
class LevelFit:
    order: int
    centers: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    within: float
    min_gap: float
    separation: float


def _fit_at_k(xc: npt.NDArray[np.float64], k: int, lo: float, hi: float) -> LevelFit:
    centers = _seeded_lloyd_1d(xc, k, lo, hi)
    empty = LevelFit(0, np.zeros(0), np.zeros(0, dtype=np.int64), 0.0, 0.0, 0.0)
    if centers.size == 0:
        return empty
    labels = nearest_labels(xc, centers)
    occ_c: list[float] = []
    occ_n: list[int] = []
    ssq = 0.0
    total = 0
    for j in range(centers.size):
        members = xc[labels == j]
        if members.size == 0:
            continue
        occ_c.append(float(members.mean()))
        occ_n.append(int(members.size))
        ssq += float(((members - members.mean()) ** 2).sum())
        total += int(members.size)
    if not occ_c:
        return empty
    c = np.asarray(occ_c, dtype=np.float64)
    n = np.asarray(occ_n, dtype=np.int64)
    srt = np.argsort(c)
    cen, cnt = c[srt], n[srt]
    within = float(np.sqrt(ssq / total)) if total else 0.0
    if cen.size < 2:
        return LevelFit(int(cen.size), cen, cnt, within, 0.0, 0.0)
    min_gap = float(np.min(np.diff(cen)))
    separation = min_gap / (within + _SEPARATION_EPS)
    return LevelFit(int(cen.size), cen, cnt, within, min_gap, separation)


def fit_levels(x: npt.ArrayLike, ks: tuple[int, ...] = _LEVEL_KS) -> LevelFit:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 2:
        return LevelFit(0, np.zeros(0), np.zeros(0, dtype=np.int64), 0.0, 0.0, 0.0)
    lo, hi = _clip_range(x)
    xc = np.clip(x, lo, hi)
    best: LevelFit | None = None
    for k in ks:
        if k > xc.size:
            continue
        fit = _fit_at_k(xc, k, lo, hi)
        if best is None or fit.separation > best.separation:
            best = fit
    if best is None:
        return LevelFit(0, np.zeros(0), np.zeros(0, dtype=np.int64), 0.0, 0.0, 0.0)
    return best
