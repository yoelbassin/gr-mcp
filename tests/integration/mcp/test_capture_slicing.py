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
