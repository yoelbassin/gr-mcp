from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_KMEANS_ITERS = 100
_SEPARATION_EPS = 1e-9
_LEVEL_KS = (2, 4, 8)
_CLIP_PERCENTILES = (1.0, 99.0)


def _assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)


def _clip_range(x: np.ndarray) -> tuple[float, float]:
    lo, hi = (float(v) for v in np.percentile(x, _CLIP_PERCENTILES))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _seeded_lloyd_1d(xc: np.ndarray, k: int, lo: float, hi: float) -> np.ndarray:
    if k == 1:
        return np.array([float(xc.mean())])
    quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1] * 100.0
    centers = np.unique(np.percentile(xc, quantiles))
    if centers.size < k:
        centers = np.linspace(lo, hi, k)
    for _ in range(_KMEANS_ITERS):
        labels = _assign(xc, centers)
        moved = centers.copy()
        for j in range(centers.size):
            members = xc[labels == j]
            if members.size:
                moved[j] = float(members.mean())
        if np.allclose(moved, centers):
            break
        centers = moved
    return np.sort(centers)


def kmeans_1d(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or k < 1:
        return np.zeros(0, dtype=np.float64)
    k = min(k, x.size)
    lo, hi = _clip_range(x)
    xc = np.clip(x, lo, hi)
    return _seeded_lloyd_1d(xc, k, lo, hi)


@dataclass(frozen=True)
class LevelFit:
    order: int
    centers: np.ndarray
    counts: np.ndarray
    within: float
    min_gap: float
    separation: float


def _fit_at_k(xc: np.ndarray, k: int, lo: float, hi: float) -> LevelFit:
    centers = _seeded_lloyd_1d(xc, k, lo, hi)
    empty = LevelFit(0, np.zeros(0), np.zeros(0, dtype=np.int64), 0.0, 0.0, 0.0)
    if centers.size == 0:
        return empty
    labels = _assign(xc, centers)
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


def fit_levels(x: np.ndarray, ks: tuple[int, ...] = _LEVEL_KS) -> LevelFit:
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
