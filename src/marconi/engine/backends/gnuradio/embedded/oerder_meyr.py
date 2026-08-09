from __future__ import annotations

import numpy as np


def estimate_tau(block: np.ndarray, sps: int) -> float:
    e = (block.real.astype(np.float64) ** 2) + (block.imag.astype(np.float64) ** 2)
    k = np.arange(e.size)
    c = complex(np.sum(e * np.exp(-2j * np.pi * k / sps)))
    return -float(np.angle(c)) / (2.0 * np.pi)


def _cubic(samples: np.ndarray, x: np.ndarray) -> np.ndarray:
    x0 = np.floor(x).astype(np.int64)
    mu = x - x0

    def g(off: int) -> np.ndarray:
        return samples[np.clip(x0 + off, 0, samples.size - 1)]

    m1, p0, p1, p2 = g(-1), g(0), g(1), g(2)
    a0 = -0.5 * m1 + 1.5 * p0 - 1.5 * p1 + 0.5 * p2
    a1 = m1 - 2.5 * p0 + 2.0 * p1 - 0.5 * p2
    a2 = -0.5 * m1 + 0.5 * p1
    return ((a0 * mu + a1) * mu + a2) * mu + p0


def resample_symbols(
    samples: np.ndarray, sps: int, tau: float, start: float = 0.0
) -> np.ndarray:
    nsym = int((samples.size - start) // sps)
    if nsym <= 0:
        return np.empty(0, np.complex64)
    instants = start + np.arange(nsym) * float(sps) + tau * sps
    return _cubic(samples.astype(np.complex128), instants).astype(np.complex64)
