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


def _two_tone_cf32(path: Path, fs: float, n: int) -> None:
    t = np.arange(n) / fs
    channel = 2.0 * np.exp(2j * np.pi * 30_000.0 * t)  # strong, at +30 kHz
    interferer = np.exp(2j * np.pi * -25_000.0 * t)  # elsewhere in the band
    (channel + interferer).astype(np.complex64).tofile(path)


def test_survey_channelize_isolates_a_subband(tmp_path: Path) -> None:
    fs, n = 96_000.0, 1 << 16
    p = tmp_path / "wide.cf32"
    _two_tone_cf32(p, fs, n)
    wide = survey(str(p), fs)
    ch = survey(str(p), fs, center_hz=30_000.0, decim=4)
    wide_spec = cast(dict[str, float], wide["spectrum"])
    ch_spec = cast(dict[str, float], ch["spectrum"])
    # aggregate survey: dominant tone sits at its +30 kHz offset, not at DC
    assert abs(wide_spec["peak_offset_hz"] - 30_000.0) < 2_000.0
    # channelized survey: the +30 kHz tone is shifted to DC, interferer rejected
    assert abs(ch_spec["peak_offset_hz"]) < 1_500.0
    assert ch["sample_rate"] == fs / 4  # decimated rate is reported


def test_survey_channelize_leaves_no_workspace_files(tmp_path: Path) -> None:
    fs, n = 96_000.0, 1 << 16
    p = tmp_path / "wide.cf32"
    _two_tone_cf32(p, fs, n)
    runs = tmp_path / "marconi-runs"
    monkey_env = {"MARCONI_WORKSPACE": str(tmp_path)}
    import os

    os.environ.update(monkey_env)
    try:
        survey(str(p), fs, center_hz=30_000.0, decim=4)
    finally:
        os.environ.pop("MARCONI_WORKSPACE", None)
    leftover = list(runs.glob("survey-*")) if runs.exists() else []
    assert leftover == []


def test_survey_reports_capture_scale_for_burst_targeting(tmp_path: Path) -> None:
    fs, n = 96_000.0, 1 << 16
    p = tmp_path / "wide.cf32"
    _two_tone_cf32(p, fs, n)
    # non-channelized survey over an offset slice: scale maps 1:1 from the offset
    agg = survey(str(p), fs, capture_offset=1000)
    assert cast(dict, agg["bursts"])["capture_scale"] == {
        "offset_samples": 1000,
        "decim": 1,
    }
    # channelized survey: segment indices are at the decimated rate, so decim
    # scales them back to original capture samples
    ch = survey(str(p), fs, center_hz=30_000.0, decim=4, capture_offset=1000)
    assert cast(dict, ch["bursts"])["capture_scale"] == {
        "offset_samples": 1000,
        "decim": 4,
    }


def test_survey_capture_scale_survives_dtype_conversion(tmp_path: Path) -> None:
    # converted (ci16/ci8/cu8) captures bake the requested slice into a cached
    # cf32 whose own offset is 0 — the reported offset_samples must still be
    # the caller's capture_offset or the documented burst re-decode formula
    # targets a region capture_offset samples away from the surveyed burst
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
    out = survey(str(p), fs, capture_dtype="ci16", capture_offset=500)
    scale = cast(dict, cast(dict, out["bursts"])["capture_scale"])
    assert scale == {"offset_samples": 500, "decim": 1}


def test_survey_rejects_bad_decim(tmp_path: Path) -> None:
    p = tmp_path / "s.cf32"
    _fsk4_cf32(p, 48_000.0, 2_400.0, 1_200.0, 8000)
    with pytest.raises(ValueError):
        survey(str(p), 48_000.0, decim=0)


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
