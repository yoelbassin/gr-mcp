from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.engine.io.source import SourceSlice
from marconi.mcp.streams import (
    _DEFAULT_PAGE_ITEMS,
    _MAX_PAGE_ITEMS,
    ensure_cf32,
    parse_bits,
    render_page,
)
from marconi.mcp.tools import read_stream


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
    assert render_page(f, offset=0, count=10, item_type=None)["values"] == [-1.5, 2.25]
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


def test_ensure_cf32_converts_ci16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    src = tmp_path / "cap.iq"
    np.array([32767, 0, -32768, 16384], np.int16).tofile(src)
    slc = ensure_cf32(src, "ci16")
    assert (slc.offset, slc.length) == (0, 0)
    x = np.fromfile(slc.path, np.complex64)
    assert x.size == 2
    assert abs(x[0].real - 1.0) < 1e-3
    assert abs(x[1].imag - 0.5) < 1e-3


def test_ensure_cf32_conversion_is_cached_per_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # iterating specs against one capture must not write a fresh copy per run
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    src = tmp_path / "cap.iq"
    np.arange(8, dtype=np.int16).tofile(src)
    first = ensure_cf32(src, "ci16").path
    stamp = first.stat().st_mtime_ns
    again = ensure_cf32(src, "ci16").path
    assert again == first
    assert again.stat().st_mtime_ns == stamp


def test_ensure_cf32_converts_only_the_requested_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    src = tmp_path / "cap.iq"
    np.array([100, 200, 300, 400, 500, 600, 700, 800], np.int16).tofile(src)
    slc = ensure_cf32(src, "ci16", offset=1, samples=2)
    assert (slc.offset, slc.length) == (0, 0)
    x = np.fromfile(slc.path, np.complex64)
    assert x.size == 2
    assert abs(x[0].real - 300 / 32768.0) < 1e-6
    assert abs(x[1].imag - 600 / 32768.0) < 1e-6


def test_ensure_cf32_failure_leaves_no_tmp_in_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    src = tmp_path / "cap.iq"
    np.arange(8, dtype=np.int16).tofile(src)

    def boom(*a: object, **k: object) -> np.ndarray:
        raise OSError("disk full")

    monkeypatch.setattr(np, "fromfile", boom)
    with pytest.raises(OSError):
        ensure_cf32(src, "ci16")
    leftovers = list((tmp_path / "marconi-runs" / "conversions").glob("*.tmp"))
    assert leftovers == []


def test_render_int64_sidecar_page(tmp_path: Path) -> None:
    p = tmp_path / "windows.i64"
    np.array([0, 8192, 1 << 40], np.int64).tofile(p)
    page = render_page(p, offset=1, count=5, item_type=None)
    assert page["item_type"] == "l"
    assert page["values"] == [8192, 1 << 40]


def test_ensure_cf32_passthrough_and_unknown(tmp_path: Path) -> None:
    src = tmp_path / "cap.cf32"
    np.zeros(2, np.complex64).tofile(src)
    # cf32 passes through with the slice delegated to the GR file source
    assert ensure_cf32(src, "cf32", offset=5, samples=7) == SourceSlice(
        path=src, offset=5, length=7
    )
    with pytest.raises(ValueError, match="cu8"):
        ensure_cf32(src, "wat")


def test_page_reports_the_ceiling_that_clamped_it(tmp_path: Path) -> None:
    p = tmp_path / "sym.cf32"
    np.zeros(_MAX_PAGE_ITEMS["c"] + 32, np.complex64).tofile(p)
    over = render_page(p, offset=0, count=_MAX_PAGE_ITEMS["c"] + 32, item_type=None)
    assert over["count"] == _MAX_PAGE_ITEMS["c"]
    assert over["capped_at"] == _MAX_PAGE_ITEMS["c"]
    # a page short because the stream ended must NOT claim a cap
    tail = render_page(p, offset=_MAX_PAGE_ITEMS["c"], count=None, item_type=None)
    assert tail["count"] == 32 and "capped_at" not in tail


def test_read_stream_docstring_states_every_page_bound() -> None:
    doc = read_stream.__doc__ or ""
    for table in (_DEFAULT_PAGE_ITEMS, _MAX_PAGE_ITEMS):
        for kind, bound in table.items():
            assert str(bound) in doc, f"{kind}={bound} missing from read_stream doc"
