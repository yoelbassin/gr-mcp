from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.mcp.streams import ensure_cf32, parse_bits, render_page


def test_parse_bits_roundtrips() -> None:
    assert parse_bits("0110").tolist() == [0, 1, 1, 0]


def test_parse_bits_rejects_non_binary_and_empty() -> None:
    with pytest.raises(ValueError):
        parse_bits("01x0")
    with pytest.raises(ValueError):
        parse_bits("")


def test_render_bits_page_with_suffix_inference(tmp_path: Path) -> None:
    p = tmp_path / "out.u8"
    np.array([1, 0, 1, 1, 0], np.uint8).tofile(p)
    page = render_page(p, offset=1, count=3, item_type=None)
    assert page["item_type"] == "b"
    assert page["bits"] == "011"
    assert page["total_items"] == 5


def test_render_llr_and_symbol_pages(tmp_path: Path) -> None:
    f = tmp_path / "out.f32"
    np.array([-1.5, 2.25], np.float32).tofile(f)
    s = tmp_path / "out.i16"
    np.array([3, -1], np.int16).tofile(s)
    assert render_page(f, offset=0, count=10, item_type=None)["llrs"] == [-1.5, 2.25]
    assert render_page(s, offset=0, count=10, item_type=None)["symbols"] == [3, -1]


def test_render_unknown_suffix_needs_item_type(tmp_path: Path) -> None:
    p = tmp_path / "seam.dat"
    np.zeros(4, np.uint8).tofile(p)
    with pytest.raises(ValueError, match="item_type"):
        render_page(p, offset=0, count=4, item_type=None)
    assert render_page(p, offset=0, count=4, item_type="b")["bits"] == "0000"


def test_negative_offset_raises_value_error(tmp_path: Path) -> None:
    p = tmp_path / "out.u8"
    np.zeros(4, np.uint8).tofile(p)
    with pytest.raises(ValueError, match="offset"):
        render_page(p, offset=-1, count=4, item_type=None)


def test_page_count_is_clamped(tmp_path: Path) -> None:
    p = tmp_path / "big.u8"
    np.zeros(70000, np.uint8).tofile(p)
    page = render_page(p, offset=0, count=1 << 20, item_type=None)
    assert page["count"] == 65536


def test_ensure_cf32_converts_ci16(tmp_path: Path) -> None:
    src = tmp_path / "cap.iq"
    np.array([32767, 0, -32768, 16384], np.int16).tofile(src)
    out = ensure_cf32(src, "ci16", tmp_path / "runs")
    x = np.fromfile(out, np.complex64)
    assert x.size == 2
    assert abs(x[0].real - 1.0) < 1e-3
    assert abs(x[1].imag - 0.5) < 1e-3


def test_ensure_cf32_passthrough_and_unknown(tmp_path: Path) -> None:
    src = tmp_path / "cap.cf32"
    np.zeros(2, np.complex64).tofile(src)
    assert ensure_cf32(src, "cf32", tmp_path) == src
    with pytest.raises(ValueError, match="cu8"):
        ensure_cf32(src, "wat", tmp_path)
