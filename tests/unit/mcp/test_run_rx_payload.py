from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from marconi.mcp.tools import run_rx_tool


def test_uniform_window_tiling_compresses_to_ramp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a segment step over a long stream yields thousands of windows;
    # when they form a perfect arithmetic ramp, the tool compresses to {key}_ramp
    # and ships no sidecar; irregular long lists still use truncation + sidecar
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
    # uniform tiling → ramp compression: empty inline list, ramp formula
    assert cast(list[int], out["windows"]) == []
    assert out["windows_total"] == 7500
    assert "windows_ramp" in out
    ramp = cast(dict[str, int], out["windows_ramp"])
    assert ramp["count"] == 7500
    assert ramp["stride"] == 8
    assert ramp["start"] == 0
    # no sidecar for a ramp: it's fully derivable from the formula
    assert "windows_path" not in out
    # hard-bit coding-only run: no soft stream anywhere in the path
    assert out["soft_stream"] is None


def test_soft_stream_summary_states_the_sign_convention_per_level() -> None:
    from marconi.engine.run import PipelineResult
    from marconi.engine.types.models import Softstream
    from marconi.mcp.tools import _soft_summary

    def result(level: Literal["symbols", "bits"]) -> PipelineResult:
        return PipelineResult(
            status="ok",
            softstream=Softstream(
                path=Path("run/soft_tap.f32"), num_items=512, level=level
            ),
        )

    assert _soft_summary(result("bits")) == {
        "path": "run/soft_tap.f32",
        "item_type": "f",
        "items": 512,
        "level": "bits",
        "bit1_sign": "negative",
    }
    assert _soft_summary(result("symbols")) == {
        "path": "run/soft_tap.f32",
        "item_type": "f",
        "items": 512,
        "level": "symbols",
        "bit1_sign": "positive",
    }
    assert _soft_summary(PipelineResult(status="ok")) is None


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
