"""A uniform window tiling (segment output) is a derivable ramp: shipping 512
literal entries burns agent context for zero information."""

from pathlib import Path

import numpy as np

from marconi.mcp.tools import _cap_list


def test_ramp_compresses_without_sidecar(tmp_path: Path) -> None:
    payload = {"windows": list(range(0, 112 * 1558, 112))}
    _cap_list(payload, "windows", tmp_path / "windows.i64")
    assert payload["windows"] == []
    assert payload["windows_ramp"] == {"start": 0, "stride": 112, "count": 1558}
    assert payload["windows_total"] == 1558
    assert not (tmp_path / "windows.i64").exists()


def test_irregular_long_list_keeps_sidecar_behavior(tmp_path: Path) -> None:
    vals = list(range(0, 600)) + [10_000]
    payload = {"windows": list(vals)}
    _cap_list(payload, "windows", tmp_path / "windows.i64")
    assert payload["windows"] == vals[:512]
    assert payload["windows_total"] == 601
    assert np.fromfile(tmp_path / "windows.i64", np.int64).tolist() == vals


def test_short_lists_untouched(tmp_path: Path) -> None:
    payload = {"windows": [0, 7, 9]}
    _cap_list(payload, "windows", tmp_path / "windows.i64")
    assert payload == {"windows": [0, 7, 9]}
