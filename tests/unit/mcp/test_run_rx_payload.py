from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marconi.mcp.tools import read_stream, run_rx_tool


def test_long_window_lists_are_truncated_with_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a segment step over a long stream yields thousands of windows; the tool
    # payload must stay bounded for the LLM context and report the true count
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    bits = np.random.default_rng(0).integers(0, 2, 60_000).astype(np.uint8)
    src = tmp_path / "in.u8"
    bits.tofile(src)
    out = run_rx_tool(
        {"symbol_rate": 1.0, "path": [{"conv": "segment", "frame_body_len": 8}]},
        sample_rate=1.0,
        input_path=str(src),
        input_item_type="b",
    )
    assert out["status"] == "ok"
    assert len(cast(list[int], out["windows"])) == 512
    assert out["windows_total"] == 7500
    # the entries past the cap must stay reachable: full list in an int64
    # sidecar, pageable through read_stream
    sidecar = Path(cast(str, out["windows_path"]))
    full = np.fromfile(sidecar, np.int64)
    assert full.size == 7500
    assert full[:512].tolist() == cast(list[int], out["windows"])
    page = read_stream(str(sidecar), offset=7000, count=500)
    assert page["count"] == 500
    assert cast(list[int], page["values"])[0] == int(full[7000])


def test_capture_slice_params_reject_input_path_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    src = tmp_path / "in.u8"
    np.zeros(16, np.uint8).tofile(src)
    with pytest.raises(ValueError, match="capture_path only"):
        run_rx_tool(
            {"symbol_rate": 1.0, "path": [{"conv": "segment", "frame_body_len": 8}]},
            sample_rate=1.0,
            input_path=str(src),
            input_item_type="b",
            capture_offset=5,
        )
