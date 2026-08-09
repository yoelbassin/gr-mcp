from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marconi.mcp.tools import run_rx_tool

_TONE = np.exp(1j * 2 * np.pi * 0.01 * np.arange(200_000)).astype(np.complex64)


def _capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    cap = tmp_path / "iq.cf32"
    _TONE.tofile(cap)
    return str(cap)


def test_conditioning_only_path_returns_iq_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = _capture(tmp_path, monkeypatch)
    spec = {"symbol_rate": 1000.0, "path": [{"conv": "agc"}]}
    res = run_rx_tool(spec, 8000.0, capture_path=cap, timeout=60.0)
    assert res["status"] == "ok"
    stream = cast(dict[str, object], res["stream"])
    assert stream["item_type"] == "c"  # conditioned IQ, not a demod output
    items = cast(int, stream["items"])
    assert items == _TONE.size
    assert Path(cast(str, stream["path"])).stat().st_size // 8 == items


def test_trace_taps_every_gr_stage_with_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = _capture(tmp_path, monkeypatch)
    spec = {
        "symbol_rate": 1000.0,
        "path": [{"conv": "agc"}, {"conv": "psk_demod", "order": 4}],
    }
    res = run_rx_tool(spec, 8000.0, capture_path=cap, trace=True, timeout=60.0)
    assert res["status"] == "ok"
    trace = cast(list[dict[str, object]], res["trace"])
    assert [row["after"] for row in trace] == ["agc[0]", "psk_demod[1]"]
    for row in trace:
        assert row["level"] in ("iq", "symbols")
        assert cast(int, row["items"]) > 0
        p = Path(cast(str, row["path"]))
        assert p.is_file()
        stats = cast(dict[str, object], row["stats"])
        assert "constant_modulus_ratio" in stats  # a "c" wire's compact summary


def test_trace_absent_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = _capture(tmp_path, monkeypatch)
    spec = {
        "symbol_rate": 1000.0,
        "path": [{"conv": "agc"}, {"conv": "psk_demod", "order": 4}],
    }
    res = run_rx_tool(spec, 8000.0, capture_path=cap, timeout=60.0)
    assert "trace" not in res
