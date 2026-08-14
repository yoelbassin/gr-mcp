from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marconi.engine.io.source import SourceSlice
from marconi.mcp.streams import (
    _MAX_PAGE_BYTES,
    _PAGE_SPECS,
    PageType,
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
    assert page["count"] == _PAGE_SPECS[PageType.B].max_items


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
    # iterating specs against one capture must not write a fresh copy per run.
    # A hit re-stamps mtime on purpose (that stamp is the LRU recency the cache
    # evicts by), so "not rewritten" is measured on the inode, not the clock.
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    src = tmp_path / "cap.iq"
    np.arange(8, dtype=np.int16).tofile(src)
    first = ensure_cf32(src, "ci16").path
    inode = first.stat().st_ino
    again = ensure_cf32(src, "ci16").path
    assert again == first
    assert again.stat().st_ino == inode
    assert len(list(first.parent.glob("*.cf32"))) == 1


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
    np.zeros(_PAGE_SPECS[PageType.C].max_items + 32, np.complex64).tofile(p)
    over = render_page(
        p, offset=0, count=_PAGE_SPECS[PageType.C].max_items + 32, item_type=None
    )
    assert over["count"] == _PAGE_SPECS[PageType.C].max_items
    assert over["capped_at"] == _PAGE_SPECS[PageType.C].max_items
    # a page short because the stream ended must NOT claim a cap
    tail = render_page(
        p, offset=_PAGE_SPECS[PageType.C].max_items, count=None, item_type=None
    )
    assert tail["count"] == 32 and "capped_at" not in tail


@pytest.mark.parametrize(
    "suffix,dtype",
    [
        (".u8", np.uint8),
        (".i16", np.int16),
        (".f32", np.float32),
        (".i64", np.int64),
        (".cf32", np.complex64),
    ],
)
def test_a_full_page_of_every_type_fits_the_context_budget(
    tmp_path: Path, suffix: str, dtype: Any
) -> None:
    # The requirement is a SIZE, so the test measures size - at the WORST
    # legal values, not a friendly sample. Asserting the clamp equals the
    # spec ceiling (both sides imported from the source) let any retune
    # through, and sampling half-range ints / unit-scale floats pinned the
    # budget against a benign page: the i16 cap validated -32768 (m_slice's
    # own bounds test permits it) yet a full page of them measured 57,434
    # bytes against the 49,152 budget.
    p = tmp_path / f"page{suffix}"
    n = max(s.max_items for s in _PAGE_SPECS.values()) + 32
    if np.issubdtype(dtype, np.complexfloating):
        worst = -1234.567891 - 1234.567891j
        raw = np.full(n, worst).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        raw = np.full(n, -12345.6789).astype(dtype)
    else:
        info = np.iinfo(dtype)
        raw = np.full(n, info.min).astype(dtype)
    raw.tofile(p)
    page = render_page(p, offset=0, count=n, item_type=None)
    size = len(json.dumps(page, separators=(",", ":")))
    assert size <= _MAX_PAGE_BYTES, f"{suffix}: {size} bytes > {_MAX_PAGE_BYTES}"
    kind = PageType(str(page["item_type"]))
    assert page["capped_at"] == _PAGE_SPECS[kind].max_items


def test_read_stream_docstring_states_every_page_bound() -> None:
    doc = read_stream.__doc__ or ""
    for kind, spec in _PAGE_SPECS.items():
        for bound in (spec.default_items, spec.max_items):
            assert str(bound) in doc, f"{kind}={bound} missing from read_stream doc"


def test_failed_read_does_not_touch_the_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the workspace membership check called a getter with a mkdir inside, so
    # a FAILED read of a missing path created ./marconi-runs/conversions in
    # whatever directory the server was launched from - as a plugin, the
    # user's own project
    ws = tmp_path / "ws"
    monkeypatch.setenv("MARCONI_WORKSPACE", str(ws))
    with pytest.raises(Exception, match="not found|re-run"):
        render_page(
            tmp_path / "definitely-not-here.u8", offset=0, count=None, item_type=None
        )
    assert not ws.exists(), list(ws.rglob("*"))


def test_capped_at_reports_truncation_not_the_request(tmp_path: Path) -> None:
    # count=2**40 over a 4,096-item stream reported capped_at=16384 while
    # the docstring says "a short page is truncation, not end of stream" -
    # teaching the agent the opposite of the truth
    p = tmp_path / "s.u8"
    np.zeros(4096, np.uint8).tofile(p)
    page = render_page(p, offset=0, count=1 << 40, item_type=None)
    assert page["count"] == 4096
    assert "capped_at" not in page  # nothing was withheld: that IS the stream
    big = tmp_path / "big.u8"
    np.zeros(20_000, np.uint8).tofile(big)
    capped = render_page(big, offset=0, count=1 << 40, item_type=None)
    assert capped["capped_at"] == 16384  # here items genuinely were withheld
