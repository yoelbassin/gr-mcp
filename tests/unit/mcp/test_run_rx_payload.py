from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from helpers import _synth as synth

from marconi.deadline import check_deadline, set_deadline
from marconi.engine import run as run_module
from marconi.engine.quality import QualityReport
from marconi.engine.run import TraceStage
from marconi.engine.types.enums import RunStatus
from marconi.engine.types.levels import Level
from marconi.mcp.payload import _trace_row
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
    from marconi.mcp.payload import soft_summary as _soft_summary

    def result(level: Level) -> PipelineResult:
        return PipelineResult(
            status=RunStatus.OK,
            softstream=Softstream(
                path=Path("run/soft_tap.f32"), num_items=512, level=level
            ),
        )

    bits = _soft_summary(result(Level.BITS))
    assert bits is not None and bits.model_dump(mode="json") == {
        "path": "run/soft_tap.f32",
        "item_type": "f",
        "items": 512,
        "level": "bits",
        "bit1_sign": "negative",
    }
    symbols = _soft_summary(result(Level.SYMBOLS))
    assert symbols is not None and symbols.model_dump(mode="json") == {
        "path": "run/soft_tap.f32",
        "item_type": "f",
        "items": 512,
        "level": "symbols",
        "bit1_sign": "positive",
    }
    assert _soft_summary(PipelineResult(status=RunStatus.OK)) is None


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


@pytest.mark.parametrize(
    "content, expect",
    [
        (np.full(4096, np.nan, np.float32), "EmptyStreamError"),
        (np.zeros(0, np.float32), None),
    ],
)
def test_an_unreadable_trace_tap_costs_one_row_not_the_whole_call(
    tmp_path: Path, content: "np.ndarray[Any, Any]", expect: str | None
) -> None:
    """trace=True is what the run_rx docstring offers for finding WHERE a path
    breaks, so it must survive a broken path. EmptyStreamError is deliberately
    NOT a ValueError (it classifies failed_precondition), so an all-non-finite
    tap escaped _trace_stats' guard, propagated out of pipeline_payload, and
    took the whole run_rx call with it — census, quality, stream and hints —
    on exactly the run trace exists for. A tap can still hold non-finites the
    engine did not make: a CMA equalizer that diverges, or a caller's own soft
    stream. One bad tap must cost one row."""
    tap = tmp_path / "tap.f32"
    content.tofile(tap)
    st = TraceStage(
        after="fsk[0]",
        level="symbols",
        item_type="f",
        sample_rate=1.0,
        path=str(tap),
        items=int(content.size),
    )
    row = _trace_row(st)
    assert row.after == "fsk[0]"
    if expect is None:
        assert row.stats is None and row.stats_error is None
    else:
        assert row.stats is None
        assert expect in str(row.stats_error)


def test_soft_stream_path_is_the_file_the_run_actually_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An all-GR path renames its sink to the item-type suffix, so the bare
    seam name is stale afterwards. soft_stream.path went on carrying it, and
    read_stream answered [not_found] with the hint require_file reserves for a
    run whose directory was deleted — sending the operator to re-run a spec
    whose output was never called that."""
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    cap = tmp_path / "in.cf32"
    np.zeros(300, np.complex64).tofile(cap)
    out = run_rx_tool(
        {"symbol_rate": 10.0, "path": [{"conv": "fsk", "deviation": 5.0}]},
        sample_rate=48000.0,
        capture_path=str(cap),
    )
    assert out["status"] == RunStatus.EMPTY
    soft = cast("dict[str, object]", out["soft_stream"])
    assert Path(str(soft["path"])).is_file(), soft["path"]


def test_a_deadline_after_the_gr_run_is_reported_with_what_it_decoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate on the run_rx docstring's timeout paragraph: once the GR
    pipeline has started, a deadline is REPORTED — status "timeout" carrying
    the census and the stream already decoded — not raised. Scoring is the
    branch after the segment delivered; the mid-run branch is gated in
    tests/integration/engine/test_run_timeout.py."""

    def _scoring_whose_clock_expired(**kwargs: object) -> QualityReport:
        with set_deadline(0.0):
            check_deadline()
        raise AssertionError("check_deadline let an expired deadline by")

    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(run_module, "assess_quality", _scoring_whose_clock_expired)
    rate, sym_rate = 48_000.0, 4800.0
    bits = np.random.default_rng(3).integers(0, 2, 400).astype(np.uint8)
    cap = synth.write(
        tmp_path / "fsk.cf32",
        synth.cpfsk(
            bits, sps=int(rate // sym_rate), deviation=1200.0, sample_rate=rate
        ),
    )
    out = run_rx_tool(
        {"symbol_rate": sym_rate, "path": [{"conv": "fsk", "deviation": 1200.0}]},
        sample_rate=rate,
        capture_path=str(cap),
    )
    assert out["status"] == RunStatus.TIMEOUT
    stream = cast("dict[str, object]", out["stream"])
    assert stream is not None and int(cast(int, stream["items"])) > 0
    assert Path(str(stream["path"])).is_file()
    assert cast("list[object]", out["census"])


def test_a_coding_lane_search_reports_what_it_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sync_symbols' only product is marks. The wire reported the ENTRY marks
    instead, so a path ending there shipped "marks": [] while the engine held
    its hits — the stage was write-only unless a mark_frame consumed them
    inside the same coding program."""
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    src = tmp_path / "syms.f32"
    rng = np.random.default_rng(11)
    s = rng.standard_normal(4000).astype(np.float32)
    pattern = [1, -1, -1, 1, 1, -1, 1, -1]
    planted = (100, 900, 2500)
    for off in planted:
        s[off : off + len(pattern)] = np.array(pattern, np.float32) * 3.0
    s.tofile(src)
    out = run_rx_tool(
        {
            "symbol_rate": 1.0,
            "path": [{"conv": "sync_symbols", "pattern": pattern, "max_errors": 0}],
        },
        sample_rate=1.0,
        input_path=str(src),
        input_item_type="f",
        input_level="symbols",
    )
    assert out["status"] == RunStatus.OK
    marks = cast("list[int]", out["marks"])
    assert marks, "a search that found something must report it"
    assert set(planted) <= set(marks)
