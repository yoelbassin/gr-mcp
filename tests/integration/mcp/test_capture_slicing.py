from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from integration.engine.quality._capture import make_clean_capture

from marconi.mcp.tools import run_rx_tool

_SPEC = {
    "symbol_rate": 1.0,
    "path": [{"conv": "fsk", "deviation": 1.0}, {"conv": "slice"}],
}


def _items(out: dict[str, object]) -> int:
    stream = cast(dict[str, Any], out["stream"])
    return int(stream["items"])


def test_capture_offset_decodes_a_bounded_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the streaming answer to CaptureTooLarge: the GR file source starts at
    # the offset, so no copy of the capture is ever written
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    iq = make_clean_capture(tmp_path)  # 4096 bits at 4 sps
    full = run_rx_tool(_SPEC, sample_rate=4.0, capture_path=str(iq))
    half = run_rx_tool(
        _SPEC, sample_rate=4.0, capture_path=str(iq), capture_offset=8192
    )
    assert full["status"] == "ok" and half["status"] == "ok"
    assert _items(full) >= 4000
    assert 1700 <= _items(half) <= 2300


def test_capture_samples_bounds_the_slice_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """capture_offset moves the start; capture_samples bounds the LENGTH, and
    only the start half was ever exercised. Three tool docstrings send the agent
    to this parameter instead of raising timeout ("the answer to a capture too
    large for one run"), and it also keys the conversion cache and gates the
    non-finite scan — the whole length half rested on nothing."""
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    iq = make_clean_capture(tmp_path)  # 4096 bits at 4 sps
    full = run_rx_tool(_SPEC, sample_rate=4.0, capture_path=str(iq))
    quarter = run_rx_tool(
        _SPEC, sample_rate=4.0, capture_path=str(iq), capture_samples=4096
    )
    to_eof = run_rx_tool(
        _SPEC, sample_rate=4.0, capture_path=str(iq), capture_samples=0
    )
    assert quarter["status"] == "ok" and to_eof["status"] == "ok"
    assert _items(quarter) == pytest.approx(1024, abs=64)
    # 0 samples means "to EOF", not "no samples"
    assert _items(to_eof) == _items(full)


def test_capture_samples_bounds_a_converted_capture_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cf32 path slices while streaming; a ci16/ci8/cu8 capture is converted
    into the cache first, with the slice BAKED IN at offset 0. Two different
    mechanisms reaching the same answer, and only the cf32 one was covered."""
    import numpy as np

    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    sig = np.fromfile(make_clean_capture(tmp_path), np.complex64)
    raw = tmp_path / "capture.ci16"
    inter = np.empty(sig.size * 2, np.float32)
    inter[0::2], inter[1::2] = sig.real, sig.imag
    (inter * 32767.0).astype(np.int16).tofile(raw)
    kw: dict[str, Any] = {"capture_path": str(raw), "capture_dtype": "ci16"}
    whole = run_rx_tool(_SPEC, sample_rate=4.0, **kw)
    quarter = run_rx_tool(_SPEC, sample_rate=4.0, capture_samples=4096, **kw)
    assert whole["status"] == "ok" and quarter["status"] == "ok"
    assert _items(quarter) == pytest.approx(1024, abs=64)
    assert _items(quarter) < _items(whole)


def test_corruption_after_the_slice_length_does_not_reject_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The offset half of this was covered; the length half was not. A glitch
    PAST capture_samples is outside what the run reads, so the non-finite scan
    must stop at the slice end rather than walking to EOF."""
    import numpy as np

    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    sig = np.fromfile(make_clean_capture(tmp_path), np.complex64)
    sig[12000] = np.nan + 0j
    bad = tmp_path / "tail_glitch.cf32"
    sig.tofile(bad)
    whole = run_rx_tool(_SPEC, sample_rate=4.0, capture_path=str(bad))
    assert whole["status"] == "error" and "non-finite" in str(whole["error"])
    sliced = run_rx_tool(
        _SPEC, sample_rate=4.0, capture_path=str(bad), capture_samples=8192
    )
    assert sliced["status"] == "ok", sliced


def test_corruption_outside_the_slice_does_not_reject_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # huge off-air captures are exactly the ones with glitch regions you
    # slice around: the non-finite scan must cover only the requested slice
    import numpy as np

    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    iq = make_clean_capture(tmp_path)
    sig = np.fromfile(iq, np.complex64)
    sig[100] = np.nan + 0j
    bad = tmp_path / "glitched.cf32"
    sig.tofile(bad)
    whole = run_rx_tool(_SPEC, sample_rate=4.0, capture_path=str(bad))
    assert whole["status"] == "error"
    assert "non-finite" in str(whole["error"])
    sliced = run_rx_tool(
        _SPEC, sample_rate=4.0, capture_path=str(bad), capture_offset=8192
    )
    assert sliced["status"] == "ok", sliced
    assert 1700 <= _items(sliced) <= 2300


def test_first_time_dtype_conversion_honors_the_hard_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ensure_cf32's ci16/ci8/cu8 conversion is real, unbounded-by-default work
    # that runs before engine_run_rx's own deadline starts; it must count
    # against timeout too, not run to completion for free first
    import numpy as np
    from fastmcp.exceptions import ToolError

    from marconi.mcp.tools import TOOLS

    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    raw = tmp_path / "capture.ci16"
    np.zeros(20000, np.int16).tofile(raw)  # never cf32 -> a real, first-time convert
    with pytest.raises(ToolError, match=r"\[deadline_exceeded\]"):
        TOOLS["run_rx"](
            _SPEC,
            sample_rate=4.0,
            capture_path=str(raw),
            capture_dtype="ci16",
            timeout=0.0,
        )
    # the sole, decisive check: a hollow version of this fix still raises
    # [deadline_exceeded] (engine_run_rx's OWN later deadline catches a
    # timeout=0.0 run right after conversion completes, whether or not
    # conversion itself was ever bounded) - so the exception alone does not
    # prove ensure_cf32 was interrupted. Only an empty cache dir does: with
    # the fix, check_deadline() fires on ensure_cf32's very first loop
    # iteration, before any chunk is read and long before os.replace(tmp,
    # dest) publishes the converted file into the cache.
    cache_dir = tmp_path / "marconi-runs" / "conversions"
    assert list(cache_dir.glob("*.cf32")) == []
