from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_KMEANS_ITERS = 100
_SEPARATION_EPS = 1e-9
_LEVEL_KS = (2, 4, 8)


def _assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)


def kmeans_1d(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or k < 1:
        return np.zeros(0, dtype=np.float64)
    k = min(k, x.size)
    if k == 1:
        return np.array([float(x.mean())])
    quantiles = np.linspace(0.0, 100.0, k + 2)[1:-1]
    centers = np.percentile(x, quantiles)
    for _ in range(_KMEANS_ITERS):
        labels = _assign(x, centers)
        moved = centers.copy()
        for j in range(centers.size):
            members = x[labels == j]
            if members.size:
                moved[j] = float(members.mean())
        if np.allclose(moved, centers):
            break
        centers = moved
    return np.sort(centers)


@dataclass(frozen=True)
class LevelFit:
    order: int
    centers: np.ndarray
    counts: np.ndarray
    within: float
    min_gap: float
    separation: float


def _fit_at_k(x: np.ndarray, k: int) -> LevelFit:
    centers = kmeans_1d(x, k)
    empty = LevelFit(0, np.zeros(0), np.zeros(0, dtype=np.int64), 0.0, 0.0, 0.0)
    if centers.size == 0:
        return empty
    labels = _assign(x, centers)
    occ_c: list[float] = []
    occ_n: list[int] = []
    ssq = 0.0
    total = 0
    for j in range(centers.size):
        members = x[labels == j]
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
    best = _fit_at_k(x, 1)
    for k in ks:
        if k > x.size:
            continue
        fit = _fit_at_k(x, k)
        if fit.separation > best.separation:
            best = fit
    return best
