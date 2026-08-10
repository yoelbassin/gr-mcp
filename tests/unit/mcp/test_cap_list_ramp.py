"""A uniform window tiling (segment output) is a derivable ramp: shipping 512
literal entries burns agent context for zero information."""

from pathlib import Path
from typing import cast

import numpy as np

from marconi.mcp.payload import capped_int_list
from marconi.mcp.tools import read_stream


def test_ramp_compresses_without_sidecar(tmp_path: Path) -> None:
    payload: dict[str, object] = {}
    payload.update(
        capped_int_list(
            "windows", list(range(0, 112 * 1558, 112)), tmp_path / "windows.i64"
        )
    )
    assert payload["windows"] == []
    assert payload["windows_ramp"] == {"start": 0, "stride": 112, "count": 1558}
    assert payload["windows_total"] == 1558
    assert not (tmp_path / "windows.i64").exists()


def test_irregular_long_list_keeps_sidecar_behavior(tmp_path: Path) -> None:
    vals = list(range(0, 600)) + [10_000]
    payload: dict[str, object] = {}
    payload.update(capped_int_list("windows", list(vals), tmp_path / "windows.i64"))
    assert payload["windows"] == vals[:512]
    assert payload["windows_total"] == 601
    # full list preserved in sidecar for non-ramp long lists
    full = np.fromfile(tmp_path / "windows.i64", np.int64).tolist()
    assert full == vals
    # sidecar must be pageable via read_stream (agent context budget mgmt)
    page = read_stream(str(tmp_path / "windows.i64"), offset=550, count=51)
    assert page["count"] == 51
    assert cast(list[int], page["values"])[0] == vals[550]
    assert page["total_items"] == 601


def test_short_lists_untouched(tmp_path: Path) -> None:
    payload: dict[str, object] = {}
    payload.update(capped_int_list("windows", [0, 7, 9], tmp_path / "windows.i64"))
    assert payload == {"windows": [0, 7, 9]}
