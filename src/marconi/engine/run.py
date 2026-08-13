from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from marconi.deadline import check_deadline, remaining, set_deadline
from marconi.engine.backends.base import (
    Backend,
    BlockCensus,
    Diagnostic,
    DiagnosticKey,
    DiagnosticRows,
    find_diagnostic,
)
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.program import run_coding
from marconi.engine.compile.compiler import (
    CompiledPipeline,
    compile_pipeline,
    trace_item_type,
    trace_sink_path,
)
from marconi.engine.compile.errors import CompileError
from marconi.engine.io.bitfile import (
    read_bits,
    read_symbols,
    write_bits,
    write_llrs,
    write_symbols,
)
from marconi.engine.io.source import SourceSlice
from marconi.engine.quality import QualityReport, Verdict, assess_quality
from marconi.engine.stages.base import Stage
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType, RunStatus
from marconi.engine.types.levels import Level
from marconi.engine.types.models import (
    SOFT_LEVELS,
    Bitstream,
    Modem,
    Softstream,
    Symbolstream,
)
from marconi.engine.types.step import stage_label
from marconi.wire import replace


class TraceStage(BaseModel):
    """One GR-segment stage's runtime output, tapped when run_rx(trace=True).
    `after` is the stage label (conv[index]); path is the sidecar the tap wrote,
    inspectable with read_stream/stream_stats exactly like any output stream."""

    after: str
    level: str
    item_type: str
    sample_rate: float
    path: str
    items: int


class PipelineResult(BaseModel):
    status: RunStatus
    bitstream: Bitstream | None = None
    symbolstream: Symbolstream | None = None
    softstream: Softstream | None = None
    windows: list[int] = []
    marks: list[int] = []
    census: list[BlockCensus] = []
    diagnostics: list[Diagnostic] = []
    trace: list[TraceStage] = []
    stalled_at: str | None = None
    error: str | None = None
    quality: QualityReport | None = None
    hints: list[str] = []

    def diagnostic(self, block: str, key: str) -> Diagnostic | None:
        return find_diagnostic(self.diagnostics, block, key)

    def as_empty(
        self, *, error: str | None, stalled_at: str | None
    ) -> "PipelineResult":
        """The same run, reported as having decoded nothing. The streams drop
        with the status: "empty" means no product, while the census, marks,
        diagnostics and quality report stay — a zero-decode run is exactly where
        the agent needs the gradient."""
        return replace(
            self,
            status=RunStatus.EMPTY,
            error=error,
            stalled_at=stalled_at,
            bitstream=None,
            symbolstream=None,
        )

    def with_report(
        self,
        *,
        softstream: Softstream | None,
        quality: QualityReport,
        hints: list[str],
        trace: list[TraceStage],
    ) -> "PipelineResult":
        return replace(
            self,
            softstream=softstream,
            quality=quality,
            hints=hints,
            trace=trace,
        )


def _harvest_trace(
    modem: Modem,
    cp: CompiledPipeline,
    trace_dir: Path,
    registry: Mapping[str, Stage[Any, Any]],
) -> list[TraceStage]:
    rows: list[TraceStage] = []
    # the true item-rate ladder: compose the compiled rate model onto a base
    # corrected by each stage's own output_item_rate, so rows at and after an
    # internally-decimating demod report the sidecar's real rate
    rate = cp.rates[0]
    for i in range(cp.gr_steps):
        step = modem.path[i]
        conv = step.conv
        out = cp.boundaries[i + 1]
        stage = registry.get(conv)
        model_rate = rate * (cp.rates[i + 1] / cp.rates[i])
        true_rate = (
            stage.output_item_rate(step, rate, modem.symbol_rate)
            if stage is not None
            else None
        )
        rate = model_rate if true_rate is None else true_rate
        it_enum = trace_item_type(out)
        it = it_enum.value
        path = trace_sink_path(trace_dir, i, conv, it_enum)
        size = path.stat().st_size if path.is_file() else 0
        rows.append(
            TraceStage(
                after=stage_label(i, conv),
                level=out.level.value,
                item_type=it,
                sample_rate=rate,
                path=str(path),
                items=size // it_enum.item_bytes,
            )
        )
    return rows


def _harvest_marks(diagnostics: Sequence[Diagnostic]) -> list[int]:
    # a burst probe may re-detect the same position (re-lock); marks are
    # strictly-increasing offsets (types/models.py), and latest_marks dedups
    # here where raw GR output enters the typed lane
    return DiagnosticRows(diagnostics).latest_marks(DiagnosticKey.BURSTS)


def _entry_carrier(boundary: Descriptor, path: Path, marks: list[int]) -> CodingCarrier:
    if boundary.item_type is ItemType.B:
        return CodingCarrier(bits=read_bits(path), marks=tuple(marks))
    item_type = boundary.item_type.require_symbol()
    return CodingCarrier(
        bits=np.zeros(0, np.uint8),
        symbols=read_symbols(path, item_type),
        marks=tuple(marks),
    )


def _ok(
    stream: Bitstream | Symbolstream,
    seg: "GrSegment",
    marks: list[int],
    windows: list[int] | None = None,
) -> PipelineResult:
    """The one constructor for a run that produced a stream. Both tails reach
    it, so a new per-run field lands on both at once — it used to take a
    RunOutputs carrier through a per-tail _wrap_* wrapper each, four layers to
    put a path and a count in a model."""
    return PipelineResult(
        status=RunStatus.OK,
        bitstream=stream if isinstance(stream, Bitstream) else None,
        symbolstream=stream if isinstance(stream, Symbolstream) else None,
        windows=windows or [],
        marks=marks,
        census=seg.census,
        diagnostics=seg.diagnostics,
    )


def _items_on_disk(path: Path, item_type: ItemType) -> int:
    """An output stream's length, from the size the filesystem already knows.
    Reading the file to take its .size was the same answer at whole-file cost —
    and read_bits enforces the bits-layer memory budget, so a GR-only decode
    that wrote more than that raised CaptureTooLarge ("decode a bounded slice")
    at the REPORTING step of a run whose output was on disk and pageable."""
    return int(path.stat().st_size // item_type.item_bytes)


def _gr_only_stream(
    cp: CompiledPipeline, path: Path, marks: list[int]
) -> Bitstream | Symbolstream:
    if cp.final.item_type is ItemType.B and cp.final.level is Level.SYMBOLS:
        # hard symbol indices on the u8 wire (a qam-class demod final): a
        # Bitstream label would invite bitwise parsing of symbol indices.
        # The only branch that must read — it converts the wire, and the
        # bits-layer budget genuinely applies to holding it.
        symbols = read_bits(path).astype(np.int16)
        sym_path = path.with_suffix(".i16")
        write_symbols(sym_path, symbols)
        path.unlink()
        return Symbolstream(
            path=sym_path,
            num_symbols=int(symbols.size),
            item_type=ItemType.S,
            level=Level.SYMBOLS,
            marks=marks,
        )
    if cp.final.item_type is ItemType.B:
        return Bitstream(path=path, num_bits=_items_on_disk(path, ItemType.B))
    item_type = cp.final.item_type
    return Symbolstream(
        path=path,
        num_symbols=_items_on_disk(path, item_type),
        item_type=item_type,
        level=cp.final.level,
        marks=marks,
    )


def _coded_stream(
    final: Descriptor, carrier: CodingCarrier, workdir: Path
) -> Bitstream | Symbolstream:
    if final.level is Level.BITS and final.item_type is ItemType.B:
        path = workdir / "out.u8"
        write_bits(path, carrier.bits)
        return Bitstream(path=path, num_bits=int(carrier.bits.size))
    symbols = carrier.symbols if carrier.symbols is not None else np.zeros(0, np.int16)
    item_type = final.item_type.require_symbol()
    if item_type is ItemType.F:
        path = workdir / "out.f32"
        write_llrs(path, symbols)
    else:
        path = workdir / "out.i16"
        write_symbols(path, symbols)
    return Symbolstream(
        path=path,
        num_symbols=int(np.asarray(symbols).size),
        item_type=item_type,
        level=final.level,
        marks=list(carrier.marks),
    )


_SCAN_CHUNK = 1 << 22


def _first_nonfinite(
    path: Path, item_type: ItemType, offset: int = 0, length: int = 0
) -> int | None:
    """Bounded-memory scan of a float-format input file; integer formats
    cannot hold non-finite values and are never scanned. offset/length (items,
    0 length = to EOF) bound the scan to the slice the run will read: a glitch
    outside the slice must not reject it, and a sliced run over a huge capture
    must not pay a whole-file scan."""
    dtype = item_type.np_dtype
    if not np.issubdtype(dtype, np.inexact) or not path.is_file():
        return None
    itemsize = int(dtype.itemsize)
    pos = offset
    end = offset + length if length > 0 else None
    with path.open("rb") as f:
        f.seek(offset * itemsize)
        while True:
            check_deadline()
            want = _SCAN_CHUNK if end is None else min(_SCAN_CHUNK, end - pos)
            if want <= 0:
                return None
            block: npt.NDArray[np.complex64] | npt.NDArray[np.float32] = np.fromfile(
                f, dtype=dtype, count=want
            )
            if block.size == 0:
                return None
            bad = ~np.isfinite(block)
            if bad.any():
                return pos + int(bad.argmax())
            pos += int(block.size)


def _nonfinite_input(
    path: Path, item_type: ItemType, offset: int = 0, length: int = 0
) -> PipelineResult | None:
    bad = _first_nonfinite(path, item_type, offset, length)
    if bad is None:
        return None
    return PipelineResult(
        status=RunStatus.ERROR,
        error=f"input contains non-finite samples (first at item {bad}); the "
        "capture is corrupt — repair or re-record it before decoding",
    )


def _flag_empty_stream(result: PipelineResult) -> PipelineResult:
    """The engine's mirror of the GR worker's empty-sink flag: a run whose final
    stream holds zero items decoded nothing, and 'ok' would be a lie. Name the
    first census row whose items or windows fell to zero — the gradient a
    parameter search needs.

    Applies to BOTH tails. It used to guard the coding tail only, so a GR-only
    run leaned entirely on the worker's check — which returns early when the
    census is empty, and _harvest_census returns [] on any exception. A harvest
    failure therefore reported status 'ok' over a zero-item stream."""
    if result.status is not RunStatus.OK:
        return result
    items = (
        result.bitstream.num_bits
        if result.bitstream is not None
        else result.symbolstream.num_symbols if result.symbolstream is not None else 0
    )
    if items > 0:
        return result
    stall = next(
        (c for c in result.census if c.items_out == 0 or c.windows_out == 0), None
    )
    where = ""
    if stall is not None:
        what = "windows" if stall.windows_out == 0 and stall.items_out else "items"
        where = f" (no {what} past {stall.kind})"
    return result.as_empty(
        error=f"coding tail produced 0 items{where}",
        stalled_at=stall.block if stall else None,
    )


_POLARITY_HINT = (
    "a coherent demod in this path (e.g. msk) recovers carrier phase only up to a "
    "rotational ambiguity, so the bit polarity may be inverted; if a downstream "
    'sync or CRC fails, retry with the bits flipped — append descramble {"sequence": '
    '"ff"}, or prepend invert if the capture is spectrally mirrored.'
)


def composition_warnings(
    modem: Modem, registry: Mapping[str, Stage[Any, Any]]
) -> list[str]:
    """Spec-shape smells that compile fine but silently do the wrong thing.
    a window-seeding stage after another seeder
    discards the earlier stage's windows (measured trap: sync_word -> segment
    re-tiled from 0 and the sync marks vanished)."""
    warnings: list[str] = []
    prior_seeder: str | None = None
    for s in modem.path:
        stage = registry.get(s.conv)
        if stage is None or not stage.seeds_windows:
            continue
        if prior_seeder is not None:
            warnings.append(
                f"'{s.conv}' seeds windows AFTER '{prior_seeder}' already did: "
                f"the earlier windows are DISCARDED and '{s.conv}' re-seeds from "
                "stream position 0. If you meant to frame at each sync hit, "
                "either drop the second seeder (window-scoped coding stages "
                "consume the sync windows directly) or gate the stream first "
                "with sync_align and then re-tile with segment."
            )
        prior_seeder = s.conv
    return warnings


def _hints(
    modem: Modem,
    registry: Mapping[str, Stage[Any, Any]],
    final: Descriptor,
    verdict: Verdict = Verdict.DECODED,
) -> list[str]:
    hints: list[str] = list(composition_warnings(modem, registry))
    stages = [registry.get(s.conv) for s in modem.path]
    if final.level in (Level.BITS, Level.SYMBOLS) and any(
        st is not None and st.polarity_ambiguous for st in stages
    ):
        hints.append(_POLARITY_HINT)
    if verdict is not Verdict.DECODED:
        path_convs = frozenset(s.conv for s in modem.path)
        for s in modem.path:
            stage = registry.get(s.conv)
            if stage is not None:
                hints.extend(stage.retry_hints(s, path_convs))
    return hints


def _soft_file(path: Path, level: Level | None) -> Softstream:
    return Softstream(
        path=path,
        num_items=(
            path.stat().st_size // ItemType.F.item_bytes if path.is_file() else 0
        ),
        level=level if level in SOFT_LEVELS else Level.SYMBOLS,
    )


def _soft_tap(
    cp: CompiledPipeline,
    result: PipelineResult,
    seam: Path,
    input_stream: Bitstream | Symbolstream | None,
) -> Softstream | None:
    """The soft stream to score and surface, tagged with its level. A bits-level
    LLR stream (a soft-demap output) is decode-grade: its magnitude is a per-bit
    confidence that can certify or reject a decode. A symbols-level demod tap (a
    bare fsk/msk front end) is a per-symbol eye -- signal-present evidence only
    (see marconi.engine.quality._emit_soft)."""
    soft_symbols = (
        result.symbolstream is not None and result.symbolstream.item_type is ItemType.F
    )
    if soft_symbols and result.symbolstream is not None:
        return _soft_file(result.symbolstream.path, cp.final.level)
    if cp.soft_seam is not None and cp.soft_seam.is_file():
        tapped_level = next(
            (b.level for b in reversed(cp.boundaries[1:]) if b.item_type is ItemType.F),
            None,
        )
        return _soft_file(cp.soft_seam, tapped_level)
    if cp.boundary.item_type is ItemType.F:
        if cp.gr is not None:
            return _soft_file(seam, cp.boundary.level)
        if input_stream is not None:
            return _soft_file(input_stream.path, cp.boundary.level)
    return None


def _validate_entry(
    cp: CompiledPipeline, input_stream: Bitstream | Symbolstream | None
) -> None:
    if cp.final.level is Level.AUDIO:
        raise CompileError(
            "rx pipeline ends at AUDIO; run_rx extracts symbol/bit streams — "
            "continue the path with a demodulation stage (for audio output "
            "use compile_modem and run the backend directly)"
        )
    if cp.gr is not None and input_stream is not None:
        raise CompileError(
            "this path starts with a signal-processing stage, so it must be "
            "fed an IQ capture; an existing bit/symbol stream only enters a "
            "path that starts with a coding stage. Pass the capture instead "
            "of the stream, or drop the leading stages"
        )
    if cp.gr is None and input_stream is None:
        raise CompileError(
            "this modem's path starts with a coding stage, so no GR "
            "segment writes the seam file; supply input_stream"
        )
    if isinstance(input_stream, Bitstream) and cp.boundary.item_type is not ItemType.B:
        raise CompileError(
            f"input_stream is a Bitstream (item_type 'b') but the entry "
            f"boundary is item_type {cp.boundary.item_type.value!r}"
        )
    if isinstance(input_stream, Symbolstream):
        if cp.boundary.item_type not in (ItemType.S, ItemType.F):
            raise CompileError(
                f"input_stream is a Symbolstream (item_type "
                f"{input_stream.item_type!r}) but the entry boundary is "
                f"item_type {cp.boundary.item_type.value!r}"
            )
        if input_stream.item_type != cp.boundary.item_type.value:
            raise CompileError(
                f"input_stream item_type {input_stream.item_type!r} does "
                "not match the entry boundary item_type "
                f"{cp.boundary.item_type.value!r}"
            )


def _reject_nonfinite(
    cp: CompiledPipeline,
    start: Descriptor,
    source: SourceSlice | None,
    input_stream: Bitstream | Symbolstream | None,
) -> PipelineResult | None:
    soft_in = (
        isinstance(input_stream, Symbolstream) and input_stream.item_type is ItemType.F
    )
    if soft_in and input_stream is not None:
        return _nonfinite_input(input_stream.path, ItemType.F)
    if cp.gr is not None and source is not None:
        return _nonfinite_input(
            source.path, start.item_type, source.offset, source.length
        )
    return None


def run_rx(
    modem: Modem,
    registry: Mapping[str, Stage[Any, Any]],
    *,
    sample_rate: float,
    start: Descriptor,
    workdir: Path,
    source: SourceSlice | None = None,
    input_stream: Bitstream | Symbolstream | None = None,
    backend: Backend | None = None,
    trace: bool = False,
    timeout: float = 180.0,
) -> PipelineResult:
    with set_deadline(timeout):
        seam = workdir / "seam.dat"
        trace_dir: Path | None = None
        if trace:
            trace_dir = workdir / "trace"
            trace_dir.mkdir(parents=True, exist_ok=True)
        cp = compile_pipeline(
            modem,
            registry,
            direction="rx",
            sample_rate=sample_rate,
            start=start,
            source_io=source.to_params() if source is not None else {},
            sink_io={"path": str(seam)},
            name=modem.name,
            quality_tap=True,
            trace_dir=trace_dir,
        )
        _validate_entry(cp, input_stream)
        flagged = _reject_nonfinite(cp, start, source, input_stream)
        if flagged is not None:
            return flagged
        seg = _run_gr_segment(modem, cp, registry, backend, trace_dir)
        if seg.failed is not None:
            return seg.failed
        marks = seg.marks
        entry_path = seam
        if input_stream is not None:
            entry_path = input_stream.path
            if isinstance(input_stream, Symbolstream):
                marks = list(input_stream.marks)
        check_deadline()
        result = _run_coding_tail(cp, seg, seam, entry_path, workdir, marks)
        check_deadline()
        return _attach_report(modem, cp, registry, result, seg, seam, input_stream)


@dataclass(frozen=True)
class GrSegment:
    """What the GR half of a run produced. `failed` short-circuits the whole
    run; `empty` is a stall the coding tail must not overwrite with a rosier
    status of its own."""

    census: list[BlockCensus]
    diagnostics: list[Diagnostic]
    trace: list[TraceStage]
    marks: list[int]
    empty: PipelineResult | None = None
    failed: PipelineResult | None = None


def _run_gr_segment(
    modem: Modem,
    cp: CompiledPipeline,
    registry: Mapping[str, Stage[Any, Any]],
    backend: Backend | None,
    trace_dir: Path | None,
) -> GrSegment:
    if cp.gr is None:
        return GrSegment(census=[], diagnostics=[], trace=[], marks=[])
    if backend is None:
        from marconi.engine.backends.gnuradio.runner import GnuRadioBackend

        backend = GnuRadioBackend()
    check_deadline()
    r = backend.run_pipeline(cp.gr, timeout=remaining())
    census, diagnostics = list(r.census), list(r.diagnostics)
    trace_rows = (
        _harvest_trace(modem, cp, trace_dir, registry) if trace_dir is not None else []
    )
    if r.status in (RunStatus.ERROR, RunStatus.TIMEOUT):
        return GrSegment(
            census=census,
            diagnostics=diagnostics,
            trace=trace_rows,
            marks=[],
            failed=PipelineResult(
                status=r.status,
                error=r.error,
                stalled_at=r.stalled_at,
                census=census,
                diagnostics=diagnostics,
                trace=trace_rows,
            ),
        )
    empty = (
        PipelineResult(status=RunStatus.EMPTY, error=r.error, stalled_at=r.stalled_at)
        if r.status is RunStatus.EMPTY
        else None
    )
    return GrSegment(
        census=census,
        diagnostics=diagnostics,
        trace=trace_rows,
        marks=_harvest_marks(diagnostics),
        empty=empty,
    )


def _run_coding_tail(
    cp: CompiledPipeline,
    seg: GrSegment,
    seam: Path,
    entry_path: Path,
    workdir: Path,
    marks: list[int],
) -> PipelineResult:
    if cp.coding is None:
        path = seam.with_suffix(cp.final.item_type.suffix)
        if path != seam:
            seam.replace(path)
        result = _ok(_gr_only_stream(cp, path, marks), seg, marks)
    else:
        carrier = _entry_carrier(cp.boundary, entry_path, marks)
        out = run_coding(cp.coding, carrier, seg.census)
        result = _ok(
            _coded_stream(cp.final, out, workdir),
            seg,
            marks,
            [w.start for w in out.windows or []],
        )
    result = _flag_empty_stream(result)
    if seg.empty is not None and result.status in (RunStatus.OK, RunStatus.EMPTY):
        # the worker's stall is the upstream truth; the coding tail's
        # recomputation over the same census would only echo it less precisely.
        result = result.as_empty(error=seg.empty.error, stalled_at=seg.empty.stalled_at)
    return result


def _attach_report(
    modem: Modem,
    cp: CompiledPipeline,
    registry: Mapping[str, Stage[Any, Any]],
    result: PipelineResult,
    seg: GrSegment,
    seam: Path,
    input_stream: Bitstream | Symbolstream | None,
) -> PipelineResult:
    """'empty' keeps the full report: the zero-decode run is exactly where the
    agent needs the quality verdict, the census gradient, and the retry hints."""
    soft = _soft_tap(cp, result, seam, input_stream)
    quality = assess_quality(
        registry=registry,
        census=result.census,
        diagnostics=result.diagnostics,
        marks=result.marks,
        soft_stream=soft.path if soft is not None else None,
        soft_decode_grade=soft.level is Level.BITS if soft is not None else True,
    )
    return result.with_report(
        softstream=soft,
        quality=quality,
        hints=_hints(modem, registry, cp.final, quality.verdict),
        trace=seg.trace,
    )
