from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marconi.mcp.tools import run_rx_tool


def test_bare_psk_demod_yields_complex_symbolstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    cap = tmp_path / "iq.cf32"
    np.exp(1j * 2 * np.pi * 0.01 * np.arange(200_000)).astype(np.complex64).tofile(cap)
    spec = {
        "symbol_rate": 1000.0,
        "path": [{"conv": "agc"}, {"conv": "psk_demod", "order": 4}],
    }
    res = run_rx_tool(spec, 8000.0, capture_path=str(cap), timeout=60.0)
    stream = cast(dict[str, object], res["stream"])
    assert stream["item_type"] == "c"
    items = cast(int, stream["items"])
    assert items > 0
    assert Path(cast(str, stream["path"])).stat().st_size // 8 == items
