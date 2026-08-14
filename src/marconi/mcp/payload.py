from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from marconi.engine.backends.base import BlockCensus, Diagnostic
from marconi.engine.compile.compiler import CompiledPipeline
from marconi.engine.quality import QualityReport
from marconi.engine.run import PipelineResult, TraceStage
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType, RunStatus
from marconi.engine.types.levels import Level
from marconi.engine.types.models import SOFT_LEVELS, Modem
from marconi.engine.types.step import stage_label
from marconi.mcp.streams import EmptyStreamError
from marconi.mcp.streams import stream_stats_payload as _stats_model
from marconi.survey import SurveyResult
from marconi.wire import Payload, Ramp
from marconi.wire import replace as wire_replace

_MAX_INLINE_LIST = 512
_RAMP_MIN_LEN = 64

# Which way a soft value slices to bit 1, per rung. Keyed by the enum and
# checked for completeness at import against the rungs a Softstream may claim,
# so a new soft-bearing level cannot reach the wire as a bare KeyError inside a
# tool call (which the agent only ever sees as an opaque [internal_error]).
_SOFT_BIT1_SIGN: dict[Level, str] = {
    Level.SYMBOLS: "positive",
    Level.BITS: "negative",
}

_unsigned_levels = sorted(lv.value for lv in SOFT_LEVELS if lv not in _SOFT_BIT1_SIGN)
if _unsigned_levels:
    raise RuntimeError(f"soft levels with no bit-1 sign rule: {_unsigned_levels}")

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
    words_constant: int | None = None
    chance_word_rate: float | None = None


def census_rows(rows: Sequence[BlockCensus]) -> list[CensusRow]:
    return [
        CensusRow.build(
            block=c.block,
            kind=None if c.block.startswith(c.kind) else c.kind,
            items_in=c.items_in,
            items_out=c.items_out,
            windows_in=c.windows_in,
            windows_out=c.windows_out,
            chance_windows=c.chance_windows,
            words_valid=c.words_valid,
            words_total=c.words_total,
            words_constant=c.words_constant,
            chance_word_rate=c.chance_word_rate,
        )
        for c in rows
    ]


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
        marks: dict[str, object] = {}
        if d.marks is not None:
            sidecar = (
                run_dir / f"diag_{d.block}_{d.key}.i64" if run_dir is not None else None
            )
            marks = capped_int_list("marks", d.marks, sidecar)
        out.append(
            DiagnosticRow.build(
                block=d.block, key=d.key, count=d.count, value=d.value, **marks
            )
        )
    return out


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
    # build() drops the None fields: order/frame_len are omitted where a
    # boundary pins neither, rather than shipping as explicit nulls
    return SpecTraceRow.build(
        after=label,
        level=desc.level.value,
        item_type=desc.item_type.value,
        carrier=desc.carrier.value,
        amplitude=desc.amplitude.value,
        sample_rate=rate,
        order=desc.order,
        frame_len=desc.frame_len,
    )


def spec_trace_rows(modem: Modem, cp: CompiledPipeline) -> list[SpecTraceRow]:
    labels = ["<start>"] + [stage_label(i, s.conv) for i, s in enumerate(modem.path)]
    return [
        _spec_trace_row(label, desc, rate)
        for label, desc, rate in zip(labels, cp.boundaries, cp.rates)
    ]


class StreamSummary(Payload):
    """The run's output stream. `level` is the rung its items sit on, and it
    is not inferable from item_type: "f" is a per-symbol demod value at symbols
    and an LLR (opposite sign convention) at bits, and "c" is a constellation
    symbol at symbols but conditioned IQ at iq. Both used to send the reader to
    a different field — soft_stream, or the validate_modem trace — to find out
    which stream they were holding."""

    path: str
    item_type: ItemType
    level: Level
    items: int


class SoftSummary(Payload):
    path: str
    item_type: ItemType = ItemType.F
    items: int
    level: Level
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


def _rounded_quality(quality: QualityReport | None) -> QualityReport | None:
    """Evidence values round like every other float this module owns: the
    engine's report passed through verbatim, and quality.evidence[].value was
    the ONE full-precision float64 on the whole surface (17 characters where
    its neighbors ship 9)."""
    if quality is None:
        return quality
    return wire_replace(
        quality,
        margin=None if quality.margin is None else round(quality.margin, 4),
        evidence=[wire_replace(e, value=round(e.value, 6)) for e in quality.evidence],
    )


def stream_summary(result: PipelineResult) -> StreamSummary | None:
    if result.bitstream is not None:
        return StreamSummary(
            path=str(result.bitstream.path),
            item_type=ItemType.B,
            level=Level.BITS,
            items=result.bitstream.num_bits,
        )
    if result.symbolstream is not None:
        return StreamSummary(
            path=str(result.symbolstream.path),
            item_type=result.symbolstream.item_type,
            level=result.symbolstream.level,
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


def _trace_stats(st: TraceStage) -> tuple[TraceStats | None, str | None]:
    keys = _TRACE_STAT_KEYS.get(ItemType(st.item_type))
    if not (st.items and keys):
        return None, None
    try:
        full = _stats_model(Path(st.path), item_type=st.item_type, clusters=0)
    # EmptyStreamError is NOT a ValueError (it is deliberately its own type so
    # it can classify failed_precondition), so an all-non-finite tap escaped
    # this guard and took the whole run_rx call with it — census, quality,
    # stream and hints — on exactly the run trace=True is turned on for. The
    # stats_error field below exists so one unreadable tap costs one row.
    except (ValueError, FileNotFoundError, EmptyStreamError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    measured = {k: getattr(full, k) for k in keys if getattr(full, k) is not None}
    return (TraceStats.model_validate(measured) if measured else None), None


def _trace_row(st: TraceStage) -> TraceRow:
    stats, stats_error = _trace_stats(st)
    return TraceRow.build(
        after=st.after,
        level=st.level,
        item_type=st.item_type,
        sample_rate=round(st.sample_rate, 3),
        path=st.path,
        items=st.items,
        stats=stats,
        stats_error=stats_error,
    )


class PipelinePayload(Payload):
    """run_rx's wire shape. Distinct from PipelineResult on purpose: the engine
    result carries stream objects and full offset lists, this carries what an
    agent can afford to read. The windows_*/marks_* quartets are the flattened
    CappedIntList; stream and soft_stream stay present even when null, because
    their absence would read as "this run has no such stream" rather than
    "this field was omitted"."""

    always_present = frozenset({"stream", "soft_stream"})

    status: RunStatus
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
    return PipelinePayload.build(
        status=result.status,
        hints=result.hints,
        census=census_rows(result.census),
        diagnostics=diagnostic_rows(result.diagnostics, run_dir),
        stream=stream_summary(result),
        soft_stream=soft_summary(result),
        stalled_at=result.stalled_at,
        error=result.error,
        quality=_rounded_quality(result.quality),
        trace=[_trace_row(st) for st in result.trace] or None,
        **capped_int_list("windows", result.windows, run_dir / "windows.i64"),
        **capped_int_list("marks", result.marks, run_dir / "marks.i64"),
    ).as_payload()


def survey_payload(
    result: SurveyResult, *, offset_samples: int, decim: int
) -> dict[str, object]:
    return result.for_capture(offset_samples=offset_samples, decim=decim).as_payload()
