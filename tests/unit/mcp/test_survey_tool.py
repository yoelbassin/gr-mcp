from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marconi.mcp.tools import survey
from marconi.survey import CaptureTooShort


def _fsk4_cf32(
    path: Path, fs: float, rate: float, dev: float, n_sym: int, seed: int = 0
) -> None:
    sps = int(round(fs / rate))
    rng = np.random.default_rng(seed)
    syms = np.array([-3.0, -1.0, 1.0, 3.0])[rng.integers(0, 4, n_sym)]
    phase = 2 * np.pi * np.cumsum(np.repeat(syms * dev, sps)) / fs
    np.exp(1j * phase).astype(np.complex64).tofile(path)


def test_survey_returns_all_sections(tmp_path: Path) -> None:
    p = tmp_path / "s.cf32"
    _fsk4_cf32(p, 48_000.0, 2_400.0, 1_200.0, 8000)
    out = survey(str(p), 48_000.0)
    for key in ("spectrum", "envelope", "symbol_rate", "inst_freq", "bursts"):
        assert key in out
    symbol_rate = cast(dict[str, object], out["symbol_rate"])
    assert symbol_rate["candidates_hz"]


def test_survey_missing_file_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        survey(str(tmp_path / "nope.cf32"), 48_000.0)


def test_survey_too_short_raises(tmp_path: Path) -> None:
    p = tmp_path / "tiny.cf32"
    np.ones(64, np.complex64).tofile(p)
    with pytest.raises(CaptureTooShort):
        survey(str(p), 48_000.0)


def test_survey_ci16_dtype_path(tmp_path: Path) -> None:
    p = tmp_path / "s.ci16"
    fs, rate = 48_000.0, 2_400.0
    sps = int(round(fs / rate))
    rng = np.random.default_rng(1)
    syms = np.array([-1.0, 1.0])[rng.integers(0, 2, 8000)]
    phase = 2 * np.pi * np.cumsum(np.repeat(syms * 1_000.0, sps)) / fs
    iq = np.exp(1j * phase)
    inter = np.empty(iq.size * 2, np.int16)
    inter[0::2] = (iq.real * 3000).astype(np.int16)
    inter[1::2] = (iq.imag * 3000).astype(np.int16)
    inter.tofile(p)
    out = survey(str(p), fs, capture_dtype="ci16")
    symbol_rate = cast(dict[str, object], out["symbol_rate"])
    assert symbol_rate["candidates_hz"]


def test_survey_bursts_segments_are_capped(tmp_path: Path) -> None:
    p = tmp_path / "bursts.cf32"
    on_len, off_len, repeats = 200, 400, 600
    on = np.ones(on_len, np.complex64)
    off = np.zeros(off_len, np.complex64)
    pulse = np.concatenate([on, off])
    np.tile(pulse, repeats).tofile(p)
    out = survey(str(p), 48_000.0)
    bursts = cast(dict[str, object], out["bursts"])
    segments = cast(list[object], bursts["segments"])
    assert len(segments) == 512
    assert bursts["segments_total"] == repeats
    assert bursts["count"] == repeats
