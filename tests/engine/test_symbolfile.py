from pathlib import Path

import numpy as np

from marconi.engine.bitfile import read_symbols, write_llrs, write_symbols


def test_symbols_roundtrip_int16(tmp_path: Path) -> None:
    syms = np.array([0, 1, 2047, -5, 32767], dtype=np.int16)
    p = tmp_path / "s.i16"
    write_symbols(p, syms)
    back = read_symbols(p)
    assert back.dtype == np.int16
    assert np.array_equal(back, syms)


def test_symbols_roundtrip_float32(tmp_path: Path) -> None:
    syms = np.array([0.9, -0.3, 0.1, -0.8], dtype=np.float32)
    p = tmp_path / "s.f32"
    write_llrs(p, syms)
    back = read_symbols(p, item_type="f")
    assert back.dtype == np.float32
    assert np.array_equal(back, syms)


def test_symbols_default_item_type_gates_on_byte_width(tmp_path: Path) -> None:
    syms = np.array([0.9, -0.3, 0.1, -0.8], dtype=np.float32)
    p = tmp_path / "s.f32"
    write_llrs(p, syms)
    back = read_symbols(p)
    assert back.dtype == np.int16
    assert len(back) == 8
