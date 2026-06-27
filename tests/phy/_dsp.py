from __future__ import annotations

from pathlib import Path

import numpy as np


def write_bits(path: Path, bits: np.ndarray) -> Path:
    np.asarray(bits, dtype=np.uint8).tofile(path)
    return path


def read_bits(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8)


def aligned_ber(rx: np.ndarray, tx: np.ndarray, max_shift: int = 256) -> float:
    """Minimum BER of tx against rx over integer shifts 0..max_shift, requiring
    at least half of tx to overlap. Returns 0.0 on an exact (shifted) match."""
    rx = np.asarray(rx, dtype=np.uint8)
    tx = np.asarray(tx, dtype=np.uint8)
    best = 1.0
    for shift in range(min(max_shift, len(rx)) + 1):
        n = min(len(rx) - shift, len(tx))
        if n < len(tx) // 2:
            break
        best = min(best, float(np.mean(rx[shift : shift + n] != tx[:n])))
        if best == 0.0:
            break
    return best
