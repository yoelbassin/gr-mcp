from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
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


def _read_stream_doc() -> str:
    # the docstring is wrapped for the agent's eye; the claims it makes are
    # sentences, not lines
    return " ".join((read_stream.__doc__ or "").split())


# the test's own page header carries a 4-digit total_items; a 100 GB stream
# carries 12-digit offset/total_items and one more key, so the ceilings are
# priced with this much room left for the header they ship under
_HEADER_HEADROOM = 256


def _float32_candidates() -> npt.NDArray[np.float32]:
    rng = np.random.default_rng(20260815)
    bits = rng.integers(0, 2**32, size=60_000, dtype=np.uint64).astype(np.uint32)
    sampled = bits.view(np.float32)
    # |x| in [1e15, 1e16) is the band python renders in FIXED notation (16
    # digits + ".0"), wider than any exponential rendering - and uniform bit
    # patterns land there about once in 300k draws, so it is sampled directly
    band = rng.uniform(1e13, 1e17, size=20_000) * rng.choice([-1.0, 1.0], size=20_000)
    edges = np.array(
        [
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            np.finfo(np.float32).tiny,
            np.finfo(np.float32).smallest_subnormal,
            -np.finfo(np.float32).smallest_subnormal,
        ],
        np.float32,
    )
    every = np.concatenate([sampled, band.astype(np.float32), edges])
    return cast("npt.NDArray[np.float32]", every[np.isfinite(every)])


def _widest_rendered_float() -> int:
    rendered = _PAGE_SPECS[PageType.F].render(_float32_candidates())["values"]
    return max(len(json.dumps(v)) for v in cast(list[object], rendered))


def _widest_float32() -> float:
    candidates = _float32_candidates()
    rendered = cast(list[object], _PAGE_SPECS[PageType.F].render(candidates)["values"])
    widths = [len(json.dumps(v)) for v in rendered]
    return float(candidates[int(np.argmax(widths))])


def _worst_legal_fill(dtype: Any, n: int) -> npt.NDArray[Any]:
    """The most expensive page of `n` legal items of `dtype`. One item is
    non-finite on purpose: that is the true worst for float/complex, adding
    the non_finite_items key to the header while costing only one wide
    value."""
    if np.issubdtype(dtype, np.complexfloating):
        widest = _widest_float32()
        raw = np.full(n, complex(-abs(widest), -abs(widest))).astype(dtype)
        raw[0] = complex(np.inf, np.nan)
        return raw
    if np.issubdtype(dtype, np.floating):
        raw = np.full(n, _widest_float32()).astype(dtype)
        raw[0] = -np.inf
        return raw
    return np.full(n, np.iinfo(dtype).min).astype(dtype)


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
    # bytes against the 49,152 budget. The float/complex fills used to be
    # hand-picked "big-looking" numbers (-12345.6789) under this same
    # comment, and a full page of legal float32 measured 92,235 bytes
    # against the budget they claimed to pin - so the fill is now SEARCHED,
    # not assumed.
    p = tmp_path / f"page{suffix}"
    n = max(s.max_items for s in _PAGE_SPECS.values()) + 32
    _worst_legal_fill(dtype, n).tofile(p)
    page = render_page(p, offset=0, count=n, item_type=None)
    size = len(json.dumps(page, separators=(",", ":")))
    assert (
        size + _HEADER_HEADROOM <= _MAX_PAGE_BYTES
    ), f"{suffix}: {size} bytes > {_MAX_PAGE_BYTES} - {_HEADER_HEADROOM}"
    kind = PageType(str(page["item_type"]))
    assert page["capped_at"] == _PAGE_SPECS[kind].max_items


def test_read_stream_docstring_states_every_page_bound() -> None:
    doc = _read_stream_doc()
    for kind, spec in _PAGE_SPECS.items():
        for bound in (spec.default_items, spec.max_items):
            assert str(bound) in doc, f"{kind}={bound} missing from read_stream doc"


def test_the_float_page_caps_are_priced_from_the_measured_worst_item() -> None:
    # the arithmetic the ceilings were chosen by, executed: worst rendered
    # width + 1 separator, times the ceiling, must fit the budget - and the
    # docstring must name the same per-item cost it sells to the agent.
    per_float = _widest_rendered_float() + 1
    doc = _read_stream_doc()
    assert f"a float up to {per_float}" in doc, per_float
    assert f"a complex pair up to {2 * per_float}" in doc
    for kind, cost in ((PageType.F, per_float), (PageType.C, 2 * per_float)):
        spent = _PAGE_SPECS[kind].max_items * cost
        assert spent + _HEADER_HEADROOM <= _MAX_PAGE_BYTES, (kind, spent)


def test_non_finite_page_items_are_named_not_nulled(tmp_path: Path) -> None:
    # they used to reach the wire as bare NaN/Infinity tokens in-process
    # (which json.dumps(allow_nan=False) rejects outright) and as `null`
    # through the server's serializer - unmarked, and with nan-vs-inf lost
    p = tmp_path / "poison.f32"
    np.array([1.5, np.nan, np.inf, -np.inf, 0.25], np.float32).tofile(p)
    page = render_page(p, offset=0, count=None, item_type=None)
    assert page["values"] == [1.5, "nan", "inf", "-inf", 0.25]
    assert page["non_finite_items"] == 3
    json.dumps(page, allow_nan=False)  # RFC 8259 has no NaN/Infinity literal
    doc = _read_stream_doc()
    for claim in ('"nan"', '"inf"', '"-inf"', "non_finite_items"):
        assert claim in doc, f"the page ships {claim}; the docstring never says so"


def test_a_complex_page_names_the_non_finite_half(tmp_path: Path) -> None:
    p = tmp_path / "poison.cf32"
    np.array(
        [complex(np.nan, 0.5), complex(1.0, -np.inf), complex(0.25, 0.5)], np.complex64
    ).tofile(p)
    page = render_page(p, offset=0, count=None, item_type=None)
    assert page["real"] == ["nan", 1.0, 0.25]
    assert page["imag"] == [0.5, "-inf", 0.5]
    assert page["non_finite_items"] == 2
    json.dumps(page, allow_nan=False)


def test_a_finite_page_carries_no_non_finite_key(tmp_path: Path) -> None:
    # absent, not zero: a reader never pays for a measurement with nothing
    # to report
    p = tmp_path / "clean.f32"
    np.array([1.5, 0.25], np.float32).tofile(p)
    page = render_page(p, offset=0, count=None, item_type=None)
    assert "non_finite_items" not in page


def test_small_page_values_survive_their_own_rounding(tmp_path: Path) -> None:
    # a fixed 4-DECIMAL round zeroed every soft value below ~1e-4 - a ci16
    # capture's own LSB is 3.05e-5, so a weak stream paged as all zeros
    p = tmp_path / "tiny.f32"
    np.array([1.2345678e-8, -3.05e-5, 1.23456789], np.float32).tofile(p)
    page = render_page(p, offset=0, count=None, item_type=None)
    values = cast(list[float], page["values"])
    assert values[0] == pytest.approx(1.2345678e-8, rel=1e-5)
    assert values[1] == pytest.approx(-3.05e-5, rel=1e-5)
    # the docstring sells 6 significant figures - the width the ceilings are
    # priced from, so it is the promise that has to hold at every magnitude
    assert values[2] == 1.23457
    assert "6 significant figures" in _read_stream_doc()


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
