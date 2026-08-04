from __future__ import annotations

from pathlib import Path

import pytest

from marconi.mcp.streams import render_page


def test_render_page_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="auto-cleaned"):
        render_page(tmp_path / "gone.f32", offset=0, count=10, item_type=None)
