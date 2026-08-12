from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from marconi.engine.backends.base import BlockCensus, Diagnostic, RunResult
from marconi.engine.compile.compiler import CompiledPipeline
from marconi.engine.quality import QualityReport
from marconi.engine.run import PipelineResult, TraceStage
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.models import Modem
from marconi.engine.types.step import stage_label
from marconi.mcp.streams import stream_stats_payload as _stats_model
from marconi.mcp.wire import Payload, Ramp
from marconi.survey import (
    CarrierStats,
    EnvelopeStats,
    InstFreqStats,
    SpectrumStats,
    SurveyResult,
    SymbolRateStats,
)

_MAX_INLINE_LIST = 512
_RAMP_MIN_LEN = 64

_SOFT_BIT1_SIGN = {"symbols": "positive", "bits": "negative"}

# The compact per-stage stats a trace row carries, by the item type the tap
# wrote. Keyed by the enum so a new wire type fails the check below rather
# than silently tracing with no stats at all.
_TRACE_STAT_KEYS: dict[ItemType, tuple[str, ...]] = {
    ItemType.C: ("mean_magnitude", "std_magnitude", "constant_modulus_ratio"),
    ItemType.F: ("min", "max", "mean", "std"),
    ItemType.S: ("min", "max", "mean", "std"),
    ItemType.B: ("ones_fraction",),
}

_untraced = sorted(t.value for t in ItemType if t not in _TRACE_STAT_KEYS)
if _untraced:
    raise RuntimeError(f"ItemType members with no trace stat keys: {_untraced}")


def as_ramp(seq: Sequence[int]) -> Ramp | None:
    if len(seq) <= _RAMP_MIN_LEN:
        return None
    stride = seq[1] - seq[0]
    arr = np.asarray(seq, np.int64)
    if not bool(np.all(np.diff(arr) == stride)):
        return None
    return Ramp(start=int(seq[0]), stride=int(stride), count=len(seq))


class CappedIntList(Payload):
    """A run's offset list, bounded for the agent's context. The whole list
    inline when it is short; the formula alone when it is an exact arithmetic
    tiling; otherwise a truncated head plus the total and, where the caller
    gave somewhere to put it, an int64 sidecar holding all of it."""

    values: list[int]
    total: int | None = None
    ramp: Ramp | None = None
    path: str | None = None

    def under(self, key: str) -> dict[str, object]:
        """Flattened onto the response under `key`, `key`_total, `key`_ramp,
        `key`_path — one place that knows the naming convention, instead of
        every producer spelling the four suffixes out."""
        base = self.as_payload()
        renamed = {key: base.pop("values")}
        renamed.update({f"{key}_{name}": v for name, v in base.items()})
        return renamed


def capped_int_list(
    key: str, seq: Sequence[int], sidecar: Path | None = None
) -> dict[str, object]:
    ramp = as_ramp(seq)
    if ramp is not None:
        # a uniform tiling is fully derivable: ship the formula, not the list
        return CappedIntList.build(values=[], total=ramp.count, ramp=ramp).under(key)
    if len(seq) <= _MAX_INLINE_LIST:
        return CappedIntList.build(values=list(seq)).under(key)
    path: str | None = None
    if sidecar is not None:
        # entries past the cap have no other home; the sidecar keeps the
        # full list reachable (a >512-frame carve needs every offset)
        np.asarray(seq, np.int64).tofile(sidecar)
        path = str(sidecar)
    return CappedIntList.build(
        values=list(seq[:_MAX_INLINE_LIST]), total=len(seq), path=path
    ).under(key)


class CensusRow(Payload):
    """A census row on the wire. `kind` is dropped when the block id already
    starts with it — the row would otherwise repeat itself once per block."""

    block: str
    kind: str | None = None
    items_in: int | None = None
    items_out: int | None = None
    windows_in: int | None = None
    windows_out: int | None = None
    chance_windows: float | None = None
    words_valid: int | None = None
    words_total: int | None = None
    chance_word_rate: float | None = None


def census_rows(rows: Sequence[BlockCensus]) -> list[CensusRow]:
    out: list[CensusRow] = []
    for c in rows:
        fields = c.model_dump(exclude_none=True)
        if c.block.startswith(c.kind):
            del fields["kind"]
        out.append(CensusRow.model_validate(fields))
    return out


def slim_census(rows: Sequence[BlockCensus]) -> list[dict[str, object]]:
    return [r.as_payload() for r in census_rows(rows)]


class DiagnosticRow(Payload):
    block: str
    key: str
    count: int | None = None
    value: float | None = None
    # the four keys CappedIntList.under("marks") produces, declared so a
    # capped list that does not match this shape fails validation here
    marks: list[int] | None = None
    marks_total: int | None = None
    marks_ramp: Ramp | None = None
    marks_path: str | None = None


def diagnostic_rows(
    rows: Sequence[Diagnostic], run_dir: Path | None = None
) -> list[DiagnosticRow]:
    out: list[DiagnosticRow] = []
    for d in rows:
        # exclude marks from the dump: it is capped below, and materializing
        # the whole list here only to overwrite the key builds a second
        # full-size Python int list per row (the worker already built one)
        fields = d.model_dump(exclude_none=True, exclude={"marks"})
        if d.marks is not None:
            sidecar = (
                run_dir / f"diag_{d.block}_{d.key}.i64" if run_dir is not None else None
            )
            fields.update(capped_int_list("marks", d.marks, sidecar))
        out.append(DiagnosticRow.model_validate(fields))
    return out


def slim_diagnostics(
    rows: Sequence[Diagnostic], run_dir: Path | None = None
) -> list[dict[str, object]]:
    return [r.as_payload() for r in diagnostic_rows(rows, run_dir)]


class RunTxPayload(Payload):
    status: str
    iq_path: str
    num_samples: int
    sample_rate: float
    census: list[CensusRow]
    error: str | None = None


def run_tx_payload(r: RunResult, out: Path, sample_rate: float) -> dict[str, object]:
    n = out.stat().st_size // ItemType.C.item_bytes if out.is_file() else 0
    return RunTxPayload.build(
        status=r.status,
        iq_path=str(out),
        num_samples=n,
        sample_rate=sample_rate,
        census=census_rows(r.census),
        error=r.error,
    ).as_payload()


class ErrorRow(Payload):
    code: str
    message: str
    at: str | None = None


class SpecTraceRow(Payload):
    after: str
    level: str
    item_type: str
    carrier: str
    amplitude: str
    order: int | None = None
    frame_len: int | None = None
    sample_rate: float


def _spec_trace_row(label: str, desc: Descriptor, rate: float) -> SpecTraceRow:
    # order/frame_len are LEFT OUT when a boundary pins neither, rather than
    # passed as None: an unset field is omitted from the row, an explicit null
    # would ship (marconi.mcp.wire.Payload)
    fields: dict[str, object] = {
        "after": label,
        "level": desc.level.value,
        "item_type": desc.item_type.value,
        "carrier": desc.carrier.value,
        "amplitude": desc.amplitude.value,
        "sample_rate": rate,
    }
    if desc.order is not None:
        fields["order"] = desc.order
    if desc.frame_len is not None:
        fields["frame_len"] = desc.frame_len
    return SpecTraceRow.model_validate(fields)


def spec_trace_rows(modem: Modem, cp: CompiledPipeline) -> list[dict[str, object]]:
    labels = ["<start>"] + [stage_label(i, s.conv) for i, s in enumerate(modem.path)]
    return [
        _spec_trace_row(label, desc, rate).as_payload()
        for label, desc, rate in zip(labels, cp.boundaries, cp.rates)
    ]


def error_rows(rows: Sequence[ErrorRow]) -> list[dict[str, object]]:
    return [r.as_payload() for r in rows]


class StreamSummary(Payload):
    path: str
    item_type: str
    items: int


class SoftSummary(Payload):
    path: str
    item_type: Literal["f"] = "f"
    items: int
    level: str
    bit1_sign: str


class TraceStats(Payload):
    """The compact per-stage numbers a trace row carries. Which subset appears
    is set by the tapped item type (_TRACE_STAT_KEYS)."""

    mean_magnitude: float | None = None
    std_magnitude: float | None = None
    constant_modulus_ratio: float | None = None
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    ones_fraction: float | None = None


class TraceRow(Payload):
    after: str
    level: str
    item_type: str
    sample_rate: float
    path: str
    items: int
    stats: TraceStats | None = None
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
    fields: dict[str, object] = {
        "after": st.after,
        "level": st.level,
        "item_type": st.item_type,
        "sample_rate": round(st.sample_rate, 3),
        "path": st.path,
        "items": st.items,
    }
    keys = _TRACE_STAT_KEYS.get(ItemType(st.item_type))
    if not (st.items and keys):
        return TraceRow.model_validate(fields)
    try:
        full = _stats_model(Path(st.path), item_type=st.item_type, clusters=0)
    except (ValueError, FileNotFoundError) as exc:
        fields["stats_error"] = f"{type(exc).__name__}: {exc}"
        return TraceRow.model_validate(fields)
    measured = {k: getattr(full, k) for k in keys if getattr(full, k) is not None}
    if measured:
        fields["stats"] = TraceStats.model_validate(measured)
    return TraceRow.model_validate(fields)


def trace_payload(result: PipelineResult) -> list[dict[str, object]]:
    return [_trace_row(st).as_payload() for st in result.trace]


class PipelinePayload(Payload):
    """run_rx's wire shape. Distinct from PipelineResult on purpose: the engine
    result carries stream objects and full offset lists, this carries what an
    agent can afford to read. The windows_*/marks_* quartets are the flattened
    CappedIntList; stream and soft_stream stay present even when null, because
    their absence would read as "this run has no such stream" rather than
    "this field was omitted"."""

    status: str
    stalled_at: str | None = None
    error: str | None = None
    hints: list[str] = []
    quality: QualityReport | None = None
    census: list[CensusRow] = []
    diagnostics: list[DiagnosticRow] = []
    windows: list[int] = []
    windows_total: int | None = None
    windows_ramp: Ramp | None = None
    windows_path: str | None = None
    marks: list[int] = []
    marks_total: int | None = None
    marks_ramp: Ramp | None = None
    marks_path: str | None = None
    trace: list[TraceRow] | None = None
    stream: StreamSummary | None = None
    soft_stream: SoftSummary | None = None


def pipeline_payload(result: PipelineResult, run_dir: Path) -> dict[str, object]:
    fields: dict[str, object] = {
        "status": result.status,
        "hints": result.hints,
        "census": census_rows(result.census),
        "diagnostics": diagnostic_rows(result.diagnostics, run_dir),
        # explicitly set even when null — see PipelinePayload
        "stream": stream_summary(result),
        "soft_stream": soft_summary(result),
    }
    if result.stalled_at is not None:
        fields["stalled_at"] = result.stalled_at
    if result.error is not None:
        fields["error"] = result.error
    if result.quality is not None:
        fields["quality"] = result.quality
    fields.update(capped_int_list("windows", result.windows, run_dir / "windows.i64"))
    fields.update(capped_int_list("marks", result.marks, run_dir / "marks.i64"))
    if result.trace:
        fields["trace"] = [_trace_row(st) for st in result.trace]
    return PipelinePayload.model_validate(fields).as_payload()


class CaptureScale(Payload):
    """Maps a burst segment back onto the ORIGINAL capture so it can be
    re-decoded with a targeted slice: capture_offset = offset_samples +
    start*decim, capture_samples = length*decim."""

    offset_samples: int
    decim: int


class BurstsPayload(Payload):
    count: int
    duty_cycle: float
    dominant_period_samples: int | None = None
    segments: list[tuple[int, int]]
    segments_total: int | None = None
    capture_scale: CaptureScale


class SurveyPayload(Payload):
    sample_rate: float
    span_samples: int
    analyzed_samples: int
    analyzed_start: int
    spectrum: SpectrumStats
    carrier: CarrierStats
    envelope: EnvelopeStats
    symbol_rate: SymbolRateStats
    inst_freq: InstFreqStats
    bursts: BurstsPayload


def survey_payload(
    result: SurveyResult, *, offset_samples: int, decim: int
) -> dict[str, object]:
    segments = result.bursts.segments
    bursts: dict[str, object] = {
        **result.bursts.model_dump(exclude={"segments"}),
        "segments": list(segments[:_MAX_INLINE_LIST]),
        "capture_scale": CaptureScale(offset_samples=offset_samples, decim=decim),
    }
    if len(segments) > _MAX_INLINE_LIST:
        bursts["segments_total"] = len(segments)
    fields = result.model_dump(exclude={"bursts"}) | {"bursts": bursts}
    return SurveyPayload.model_validate(fields).as_payload()
