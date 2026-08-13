from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from helpers.assets.derive import DERIVERS, DeriveError

RATE = 128000


def _stream(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=np.uint32)
    i = (idx % 251).astype(np.uint8)
    q = ((idx // 251) % 251).astype(np.uint8)
    return i, q


def _wav(path: Path, seconds: float) -> None:
    n = int(RATE * seconds)
    i, q = _stream(n)
    frames = np.empty(n * 2, np.uint8)
    frames[0::2], frames[1::2] = i, q
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(1)
        w.setframerate(RATE)
        w.writeframes(frames.tobytes())


def _raw_wav(
    path: Path, *, nchannels: int, sampwidth: int, framerate: int, n: int
) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(nchannels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(bytes(n * nchannels * sampwidth))


def test_pocsag_slice_carves_three_to_ten_seconds(tmp_path: Path) -> None:
    src, dst = tmp_path / "in.wav", tmp_path / "out.cf32"
    n = int(RATE * 12.0)
    _wav(src, 12.0)
    DERIVERS["pocsag_slice"](src, dst)
    got = np.fromfile(dst, dtype=np.complex64)

    start, end = int(RATE * 3.0), int(RATE * 10.0)
    want_i, want_q = _stream(n)
    want = (
        want_i[start:end].astype(np.float32)
        - 127.5
        + 1j * (want_q[start:end].astype(np.float32) - 127.5)
    ).astype(np.complex64)

    assert got.size == int(RATE * 7)
    # the fixture is time-varying (i, q derived from sample index), so this
    # equality only holds if the deriver carved exactly [start:end] - a wrong
    # offset, even by one sample, produces a different array and fails
    assert np.array_equal(got, want)
    # centred on 127.5 and NOT rescaled - this reproduces the historical asset
    # byte-for-byte; dividing by 128 here would shift every sample by 128x and
    # silently change what the gate decodes
    assert np.allclose(got.real[0], want_i[start] - 127.5)
    assert np.allclose(got.imag[0], want_q[start] - 127.5)


def test_a_short_source_is_refused_rather_than_truncated(tmp_path: Path) -> None:
    src, dst = tmp_path / "short.wav", tmp_path / "out.cf32"
    _wav(src, 5.0)
    with pytest.raises(DeriveError, match="short"):
        DERIVERS["pocsag_slice"](src, dst)
    assert not dst.exists()


def test_mono_source_is_refused(tmp_path: Path) -> None:
    src, dst = tmp_path / "mono.wav", tmp_path / "out.cf32"
    _raw_wav(src, nchannels=1, sampwidth=1, framerate=RATE, n=100)
    with pytest.raises(DeriveError, match="channel"):
        DERIVERS["pocsag_slice"](src, dst)
    assert not dst.exists()


def test_wrong_sample_width_is_refused(tmp_path: Path) -> None:
    src, dst = tmp_path / "wide.wav", tmp_path / "out.cf32"
    _raw_wav(src, nchannels=2, sampwidth=2, framerate=RATE, n=100)
    with pytest.raises(DeriveError, match="8-bit"):
        DERIVERS["pocsag_slice"](src, dst)
    assert not dst.exists()


def test_wrong_frame_rate_is_refused(tmp_path: Path) -> None:
    src, dst = tmp_path / "slow.wav", tmp_path / "out.cf32"
    _raw_wav(src, nchannels=2, sampwidth=1, framerate=64000, n=100)
    with pytest.raises(DeriveError, match="128000"):
        DERIVERS["pocsag_slice"](src, dst)
    assert not dst.exists()


def test_every_deriver_is_callable() -> None:
    assert set(DERIVERS) >= {"pocsag_slice"}
    for name, fn in DERIVERS.items():
        assert callable(fn), name
