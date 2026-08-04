from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.mcp.tools import TOOLS, stream_stats


def test_tool_is_registered() -> None:
    assert "stream_stats" in TOOLS


def test_tool_returns_levels(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rails = rng.choice([-3.0, -1.0, 1.0, 3.0], size=6000)
    p = tmp_path / "s.f32"
    np.asarray(rails + rng.normal(0, 0.15, 6000), np.float32).tofile(p)
    out = stream_stats(str(p), clusters=4)
    levels = out["levels"]
    assert isinstance(levels, list)
    assert len(levels) == 4
    assert out["item_type"] == "f"
