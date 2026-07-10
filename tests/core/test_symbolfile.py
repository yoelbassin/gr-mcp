from pathlib import Path

import numpy as np

from marconi.core.bitfile import read_symbols, write_symbols


def test_symbols_roundtrip_int16(tmp_path: Path) -> None:
    syms = np.array([0, 1, 2047, -5, 32767], dtype=np.int16)
    p = tmp_path / "s.i16"
    write_symbols(p, syms)
    back = read_symbols(p)
    assert back.dtype == np.int16
    assert np.array_equal(back, syms)
