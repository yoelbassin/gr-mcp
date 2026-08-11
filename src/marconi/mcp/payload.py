from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from marconi.engine.backends.base import BlockCensus, Diagnostic, RunResult
from marconi.engine.compile.compiler import CompiledPipeline
from marconi.engine.run import PipelineResult, TraceStage
from marconi.engine.types.enums import ItemType
from marconi.engine.types.models import Modem
from marconi.engine.types.step import stage_label
from marconi.mcp.streams import stream_stats as _compute_stats
from marconi.survey import SurveyResult

_MAX_INLINE_LIST = 512
_RAMP_MIN_LEN = 64

_SOFT_BIT1_SIGN = {"symbols": "positive", "bits": "negative"}

_TRACE_STAT_KEYS: dict[str, tuple[str, ...]] = {
    "c": ("mean_magnitude", "std_magnitude", "constant_modulus_ratio"),
    "f": ("min", "max", "mean", "std"),
    "s": ("min", "max", "mean", "std"),
    "b": ("ones_fraction",),
}


def as_ramp(seq: Sequence[int]) -> dict[str, int] | None:
    if len(seq) <= _RAMP_MIN_LEN:
        return None
    stride = seq[1] - seq[0]
    arr = np.asarray(seq, np.int64)
    if not bool(np.all(np.diff(arr) == stride)):
        return None
    return {"start": int(seq[0]), "stride": int(stride), "count": len(seq)}


def capped_int_list(
    key: str, seq: Sequence[int], sidecar: Path | None = None
) -> dict[str, object]:
    ramp = as_ramp(seq)
    if ramp is not None:
        # a uniform tiling is fully derivable: ship the formula, not the list
        return {key: [], f"{key}_total": ramp["count"], f"{key}_ramp": ramp}
    if len(seq) > _MAX_INLINE_LIST:
        fields: dict[str, object] = {
            key: list(seq[:_MAX_INLINE_LIST]),
            f"{key}_total": len(seq),
        }
        if sidecar is not None:
            # entries past the cap have no other home; the sidecar keeps the
            # full list reachable (a >512-frame carve needs every offset)
            np.asarray(seq, np.int64).tofile(sidecar)
            fields[f"{key}_path"] = str(sidecar)
        return fields
    return {key: list(seq)}


def slim_census(rows: Sequence[BlockCensus]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for c in rows:
        row: dict[str, object] = c.model_dump(mode="json", exclude_none=True)
        if c.block.startswith(c.kind):
            del row["kind"]
        out.append(row)
    return out


def slim_diagnostics(
    rows: Sequence[Diagnostic], run_dir: Path | None = None
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for d in rows:
        row: dict[str, object] = d.model_dump(mode="json", exclude_none=True)
        if d.marks is not None:
            sidecar = (
                run_dir / f"diag_{d.block}_{d.key}.i64" if run_dir is not None else None
            )
            row.update(capped_int_list("marks", d.marks, sidecar))
        out.append(row)
    return out


class RunTxPayload(BaseModel):
    status: str
    iq_path: str
    num_samples: int
    sample_rate: float
    census: list[dict[str, object]]
    error: str | None = None


def run_tx_payload(r: RunResult, out: Path, sample_rate: float) -> dict[str, object]:
    n = out.stat().st_size // ItemType.C.item_bytes if out.is_file() else 0
    return RunTxPayload(
        status=r.status,
        iq_path=str(out),
        num_samples=n,
        sample_rate=sample_rate,
        census=slim_census(r.census),
        error=r.error,
    ).model_dump(mode="json", exclude_none=True)


class ErrorRow(BaseModel):
    code: str
    message: str
    at: str | None = None


class SpecTraceRow(BaseModel):
    after: str
    level: str
    item_type: str
    carrier: str
    amplitude: str
    order: int | None = None
    frame_len: int | None = None
    sample_rate: float


def spec_trace_rows(modem: Modem, cp: CompiledPipeline) -> list[dict[str, object]]:
    labels = ["<start>"] + [stage_label(i, s.conv) for i, s in enumerate(modem.path)]
    return [
        SpecTraceRow(
            after=label,
            level=desc.level.value,
            item_type=desc.item_type.value,
            carrier=desc.carrier.value,
            amplitude=desc.amplitude.value,
            order=desc.order,
            frame_len=desc.frame_len,
            sample_rate=rate,
        ).model_dump(mode="json", exclude_none=True)
        for label, desc, rate in zip(labels, cp.boundaries, cp.rates)
    ]


def error_rows(rows: Sequence[ErrorRow]) -> list[dict[str, object]]:
    return [r.model_dump(mode="json", exclude_none=True) for r in rows]


class StreamSummary(BaseModel):
    path: str
    item_type: str
    items: int


class SoftSummary(BaseModel):
    path: str
    item_type: Literal["f"] = "f"
    items: int
    level: str
    bit1_sign: str


class TraceRow(BaseModel):
    after: str
    level: str
    item_type: str
    sample_rate: float
    path: str
    items: int
    stats: dict[str, object] | None = None
    stats_error: str | None = None


def stream_summary(result: PipelineResult) -> StreamSummary | None:
    if result.bitstream is not None:
        return StreamSummary(
            path=str(result.bitstream.path),
            item_type="b",
            items=result.bitstream.num_bits,
        )
    if result.symbolstream is not None:
        return StreamSummary(
            path=str(result.symbolstream.path),
            item_type=result.symbolstream.item_type,
            items=result.symbolstream.num_symbols,
        )
    return None


def soft_summary(result: PipelineResult) -> SoftSummary | None:
    if result.softstream is None:
        return None
    return SoftSummary(
        path=str(result.softstream.path),
        items=result.softstream.num_items,
        level=result.softstream.level,
        bit1_sign=_SOFT_BIT1_SIGN[result.softstream.level],
    )


def _trace_row(st: TraceStage) -> TraceRow:
    row = TraceRow(
        after=st.after,
        level=st.level,
        item_type=st.item_type,
        sample_rate=round(st.sample_rate, 3),
        path=st.path,
        items=st.items,
    )
    keys = _TRACE_STAT_KEYS.get(st.item_type)
    if not (st.items and keys):
        return row
    try:
        full = _compute_stats(Path(st.path), item_type=st.item_type, clusters=0)
    except (ValueError, FileNotFoundError) as exc:
        return row.model_copy(update={"stats_error": f"{type(exc).__name__}: {exc}"})
    stat = {k: full[k] for k in keys if full.get(k) is not None}
    return row.model_copy(update={"stats": stat}) if stat else row


def trace_payload(result: PipelineResult) -> list[dict[str, object]]:
    return [
        _trace_row(st).model_dump(mode="json", exclude_none=True) for st in result.trace
    ]


def pipeline_payload(result: PipelineResult, run_dir: Path) -> dict[str, object]:
    # null-valued fields are omitted product-wide (a census row lists only the
    # ports/measures its block has); stream/soft_stream stay explicit below
    payload: dict[str, Any] = result.model_dump(
        mode="json",
        exclude_none=True,
        exclude={
            "bitstream",
            "symbolstream",
            "softstream",
            "census",
            "diagnostics",
            "trace",
            "windows",
            "marks",
        },
    )
    payload["census"] = slim_census(result.census)
    payload["diagnostics"] = slim_diagnostics(result.diagnostics, run_dir)
    payload.update(capped_int_list("windows", result.windows, run_dir / "windows.i64"))
    payload.update(capped_int_list("marks", result.marks, run_dir / "marks.i64"))
    if result.trace:
        payload["trace"] = trace_payload(result)
    stream = stream_summary(result)
    soft = soft_summary(result)
    # both keys stay present even when null: their absence would read as "this
    # run has no such stream" rather than "this field was omitted"
    payload["stream"] = stream.model_dump(mode="json") if stream else None
    payload["soft_stream"] = soft.model_dump(mode="json") if soft else None
    return payload


def survey_payload(
    result: SurveyResult, *, offset_samples: int, decim: int
) -> dict[str, object]:
    payload: dict[str, Any] = result.model_dump(mode="json")
    bursts: dict[str, Any] = result.bursts.model_dump(mode="json")
    segments = result.bursts.segments
    if len(segments) > _MAX_INLINE_LIST:
        bursts["segments_total"] = len(segments)
        bursts["segments"] = bursts["segments"][:_MAX_INLINE_LIST]
    # map segment [start, length] back to original capture samples so a burst
    # can be re-decoded with a targeted slice: capture_offset = offset_samples +
    # start*decim, capture_samples = length*decim.
    bursts["capture_scale"] = {"offset_samples": offset_samples, "decim": decim}
    payload["bursts"] = bursts
    return payload
