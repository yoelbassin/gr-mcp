from __future__ import annotations

from pathlib import Path

import numpy as np


def write_bits(path: Path, bits: np.ndarray) -> None:
    np.asarray(bits, dtype=np.uint8).tofile(path)


def read_bits(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8)


def write_llrs(path: Path, llrs: np.ndarray) -> None:
    np.asarray(llrs, dtype=np.float32).tofile(path)


def read_llrs(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def harden(llrs: np.ndarray) -> np.ndarray:
    return (np.asarray(llrs, dtype=np.float32) < 0).astype(np.uint8)
