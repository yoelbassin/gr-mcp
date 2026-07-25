from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.engine.io.bitfile import (
    harden,
    read_bits,
    read_llrs,
    write_bits,
    write_llrs,
)


def test_bits_roundtrip(tmp_path: Path) -> None:
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    p = tmp_path / "b.u8"
    write_bits(p, bits)
    assert np.array_equal(read_bits(p), bits)


def test_llrs_roundtrip_and_harden(tmp_path: Path) -> None:
    llrs = np.array([-3.0, 2.0, -0.1, 5.0], dtype=np.float32)
    p = tmp_path / "l.f32"
    write_llrs(p, llrs)
    back = read_llrs(p)
    assert np.allclose(back, llrs)
    assert np.array_equal(harden(back), np.array([1, 0, 1, 0], dtype=np.uint8))
