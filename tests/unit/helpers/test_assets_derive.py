from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from helpers.assets.derive import DERIVERS, DeriveError

RATE = 128000


def _wav(path: Path, seconds: float) -> None:
    n = int(RATE * seconds)
    i = np.full(n, 200, np.uint8)
    q = np.full(n, 100, np.uint8)
    frames = np.empty(n * 2, np.uint8)
    frames[0::2], frames[1::2] = i, q
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(1)
        w.setframerate(RATE)
        w.writeframes(frames.tobytes())


def test_pocsag_slice_carves_three_to_ten_seconds(tmp_path: Path) -> None:
    src, dst = tmp_path / "in.wav", tmp_path / "out.cf32"
    _wav(src, 12.0)
    DERIVERS["pocsag_slice"](src, dst)
    got = np.fromfile(dst, dtype=np.complex64)
    assert got.size == int(RATE * 7)
    # centred on 127.5 and NOT rescaled - this reproduces the historical asset
    # byte-for-byte; dividing by 128 here would shift every sample by 128x and
    # silently change what the gate decodes
    assert np.allclose(got.real[0], 200 - 127.5)
    assert np.allclose(got.imag[0], 100 - 127.5)


def test_a_short_source_is_refused_rather_than_truncated(tmp_path: Path) -> None:
    src, dst = tmp_path / "short.wav", tmp_path / "out.cf32"
    _wav(src, 5.0)
    with pytest.raises(DeriveError, match="short"):
        DERIVERS["pocsag_slice"](src, dst)
    assert not dst.exists()


def test_every_deriver_is_callable() -> None:
    assert set(DERIVERS) >= {"pocsag_slice"}
    for name, fn in DERIVERS.items():
        assert callable(fn), name
