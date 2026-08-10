from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from marconi.engine.backends.base import (
    Backend,
    BlockCensus,
    Diagnostic,
    find_diagnostic,
)
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.program import run_coding
from marconi.engine.compile.compiler import (
    CompiledPipeline,
    CompileError,
    compile_pipeline,
    trace_sink_path,
)
from marconi.engine.deadline import check_deadline, remaining, set_deadline
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
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem, Softstream, Symbolstream
from marconi.engine.types.params import ParamValue
from marconi.engine.types.step import stage_label


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
    status: Literal["ok", "error", "timeout", "empty"]
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


_ITEM_BYTES: dict[str, int] = {"c": 8, "f": 4, "s": 2, "b": 1}


def _harvest_trace(
    modem: Modem, cp: CompiledPipeline, trace_dir: Path
) -> list[TraceStage]:
    rows: list[TraceStage] = []
    for i in range(cp.gr_steps):
        conv = modem.path[i].conv
        out = cp.boundaries[i + 1]
        it = out.item_type.value
        path = trace_sink_path(trace_dir, i, conv, out.item_type)
        size = path.stat().st_size if path.is_file() else 0
        rows.append(
            TraceStage(
                after=stage_label(i, conv),
                level=out.level.value,
                item_type=it,
                sample_rate=cp.rates[i + 1],
                path=str(path),
                items=size // _ITEM_BYTES[it],
            )
        )
    return rows


def _harvest_marks(diagnostics: Sequence[Diagnostic]) -> list[int]:
    # a burst probe may re-detect the same position (re-lock); marks are
    # strictly-increasing offsets (types/models.py), enforced here where raw
    # GR output enters the typed lane
    for d in reversed(diagnostics):
        if d.key == "bursts" and d.marks is not None:
            return sorted({int(m) for m in d.marks})
    return []


def _entry_carrier(boundary: Descriptor, path: Path, marks: list[int]) -> CodingCarrier:
    if boundary.item_type == "b":
        return CodingCarrier(bits=read_bits(path), marks=tuple(marks))
    item_type = boundary.item_type.require_symbol()
    return CodingCarrier(
        bits=np.zeros(0, np.uint8),
        symbols=read_symbols(path, item_type),
        marks=tuple(marks),
    )


_FINAL_SUFFIX = {"b": ".u8", "s": ".i16", "f": ".f32", "c": ".cf32"}


def _wrap_gr_only(
    cp: CompiledPipeline,
    seam: Path,
    marks: list[int],
    census: list[BlockCensus],
    diagnostics: list[Diagnostic],
) -> PipelineResult:
    path = seam.with_suffix(_FINAL_SUFFIX[cp.final.item_type])
    if path != seam:
        seam.replace(path)
    if cp.final.item_type == "b" and cp.final.level is Level.SYMBOLS:
        # hard symbol indices on the u8 wire (a qam-class demod final): a
        # Bitstream label would invite bitwise parsing of symbol indices
        symbols = read_bits(path).astype(np.int16)
        sym_path = path.with_suffix(".i16")
        write_symbols(sym_path, symbols)
        path.unlink()
        return PipelineResult(
            status="ok",
            symbolstream=Symbolstream(
                path=sym_path,
                num_symbols=int(symbols.size),
                item_type="s",
                marks=marks,
            ),
            marks=marks,
            census=census,
            diagnostics=diagnostics,
        )
    if cp.final.item_type == "b":
        bitstream = Bitstream(path=path, num_bits=int(read_bits(path).size))
        return PipelineResult(
            status="ok",
            bitstream=bitstream,
            marks=marks,
            census=census,
            diagnostics=diagnostics,
        )
    if cp.final.item_type == "c":
        num = int(path.stat().st_size // _ITEM_BYTES["c"])
        return PipelineResult(
            status="ok",
            symbolstream=Symbolstream(
                path=path, num_symbols=num, item_type="c", marks=marks
            ),
            marks=marks,
            census=census,
            diagnostics=diagnostics,
        )
    item_type = cp.final.item_type.require_symbol()
    stream: npt.NDArray[np.int16] | npt.NDArray[np.float32] = read_symbols(
        path, item_type
    )
    symbolstream = Symbolstream(
        path=path, num_symbols=int(stream.size), item_type=item_type, marks=marks
    )
    return PipelineResult(
        status="ok",
        symbolstream=symbolstream,
        marks=marks,
        census=census,
        diagnostics=diagnostics,
    )


def _wrap_result(
    final: Descriptor,
    carrier: CodingCarrier,
    workdir: Path,
    marks: list[int],
    census: list[BlockCensus],
    diagnostics: list[Diagnostic],
) -> PipelineResult:
    windows = [w.start for w in carrier.windows or []]
    if final.level is Level.BITS:
        path = workdir / "out.u8"
        write_bits(path, carrier.bits)
        bitstream = Bitstream(path=path, num_bits=int(carrier.bits.size))
        return PipelineResult(
            status="ok",
            bitstream=bitstream,
            windows=windows,
            marks=marks,
            census=census,
            diagnostics=diagnostics,
        )
    symbols = carrier.symbols if carrier.symbols is not None else np.zeros(0, np.int16)
    item_type = final.item_type.require_symbol()
    if item_type == "f":
        path = workdir / "out.f32"
        write_llrs(path, symbols)
    else:
        path = workdir / "out.i16"
        write_symbols(path, symbols)
    symbolstream = Symbolstream(
        path=path,
        num_symbols=int(np.asarray(symbols).size),
        item_type=item_type,
        marks=list(carrier.marks),
    )
    return PipelineResult(
        status="ok",
        symbolstream=symbolstream,
        windows=windows,
        marks=marks,
        census=census,
        diagnostics=diagnostics,
    )


_SCAN_DTYPES: dict[str, type] = {"c": np.complex64, "f": np.float32}
_SCAN_CHUNK = 1 << 22


def _first_nonfinite(
    path: Path, item_type: str, offset: int = 0, length: int = 0
) -> int | None:
    """Bounded-memory scan of a float-format input file; integer formats
    cannot hold non-finite values and are never scanned. offset/length (items,
    0 length = to EOF) bound the scan to the slice the run will read: a glitch
    outside the slice must not reject it, and a sliced run over a huge capture
    must not pay a whole-file scan."""
    dtype = _SCAN_DTYPES.get(item_type)
    if dtype is None or not path.is_file():
        return None
    itemsize = int(np.dtype(dtype).itemsize)
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
    path: Path, item_type: str, offset: int = 0, length: int = 0
) -> PipelineResult | None:
    bad = _first_nonfinite(path, item_type, offset, length)
    if bad is None:
        return None
    return PipelineResult(
        status="error",
        error=f"input contains non-finite samples (first at item {bad}); the "
        "capture is corrupt — repair or re-record it before decoding",
    )


def _flag_empty_coding(result: PipelineResult) -> PipelineResult:
    """The coding tail's mirror of the GR worker's empty-sink flag: a run whose
    final stream holds zero items decoded nothing, and 'ok' would be a lie.
    Name the first census row whose items or windows fell to zero — the
    gradient a parameter search needs."""
    if result.status != "ok":
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
    return result.model_copy(
        update={
            "status": "empty",
            "stalled_at": stall.block if stall else None,
            "error": f"coding tail produced 0 items{where}",
            "bitstream": None,
            "symbolstream": None,
        }
    )


_POLARITY_HINT = (
    "a coherent demod in this path (e.g. msk) recovers carrier phase only up to a "
    "rotational ambiguity, so the bit polarity may be inverted; if a downstream "
    'sync or CRC fails, retry with the bits flipped — append descramble {"sequence": '
    '"ff"}, or prepend invert if the capture is spectrally mirrored.'
)


_FSK_OPEN_LOOP_HINT = (
    "fsk ran closed-loop Gardner symbol timing (loop_bw>0), which rails on short "
    "or bursty packets and recovers few clean symbols. If this capture is bursty "
    "or the packets are short (under ~1000 symbols), retry with fsk "
    '{"loop_bw": 0} — open-loop, fixed-rate sampling at the nominal sps.'
)

_OOK_OPEN_LOOP_HINT = (
    "ook_envelope ran closed-loop Gardner symbol timing (loop_bw>0), which "
    "rails on bursty/pulsed environments and decodes nothing. Retry with "
    'ook_envelope {"loop_bw": 0} - open-loop per-burst sampling, which needs '
    "NO agc: remove any agc stage from the path. A sliding-window agc steps "
    "its gain mid-burst on pulsed signals and defeats fixed-threshold "
    "slicing; open-loop normalizes each burst internally instead."
)

_OOK_AGC_REMOVE_HINT = (
    "ook_envelope is running open-loop (amplitude-agnostic, per-burst "
    "normalized) but the path still contains an agc stage - remove it; a "
    "sliding-window agc steps its gain mid-burst on pulsed signals and "
    "corrupts the chips downstream."
)


def composition_warnings(
    modem: Modem, registry: Mapping[str, Stage[Any, Any]]
) -> list[str]:
    """Spec-shape smells that compile fine but silently do the wrong thing.
    Today exactly one rule: a window-seeding stage after another seeder
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
    verdict: Verdict = "decoded",
) -> list[str]:
    hints: list[str] = list(composition_warnings(modem, registry))
    if final.level in (Level.BITS, Level.SYMBOLS) and any(
        getattr(registry.get(s.conv), "polarity_ambiguous", False) for s in modem.path
    ):
        hints.append(_POLARITY_HINT)
    if verdict != "decoded" and any(
        s.conv == "fsk" and getattr(s, "loop_bw", 0.0) > 0.0 for s in modem.path
    ):
        hints.append(_FSK_OPEN_LOOP_HINT)
    if verdict != "decoded" and any(
        s.conv == "ook_envelope" and getattr(s, "loop_bw", 0.0) > 0.0
        for s in modem.path
    ):
        hints.append(_OOK_OPEN_LOOP_HINT)
    if (
        verdict != "decoded"
        and any(
            s.conv == "ook_envelope" and getattr(s, "loop_bw", 0.0) == 0.0
            for s in modem.path
        )
        and any(s.conv == "agc" for s in modem.path)
    ):
        hints.append(_OOK_AGC_REMOVE_HINT)
    return hints


def _soft_file(path: Path, level: Level | None) -> Softstream:
    return Softstream(
        path=path,
        num_items=path.stat().st_size // 4 if path.is_file() else 0,
        level="bits" if level is Level.BITS else "symbols",
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
    if result.symbolstream is not None and result.symbolstream.item_type == "f":
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
            "this modem's path has a GR segment fed by source_io; "
            "input_stream only enters a path that starts with a coding "
            "stage"
        )
    if cp.gr is None and input_stream is None:
        raise CompileError(
            "this modem's path starts with a coding stage, so no GR "
            "segment writes the seam file; supply input_stream"
        )
    if isinstance(input_stream, Bitstream) and cp.boundary.item_type != "b":
        raise CompileError(
            f"input_stream is a Bitstream (item_type 'b') but the entry "
            f"boundary is item_type {cp.boundary.item_type.value!r}"
        )
    if isinstance(input_stream, Symbolstream):
        if cp.boundary.item_type not in ("s", "f"):
            raise CompileError(
                f"input_stream is a Symbolstream (item_type "
                f"{input_stream.item_type!r}) but the entry boundary is "
                f"item_type {cp.boundary.item_type.value!r}"
            )
        if input_stream.item_type != cp.boundary.item_type:
            raise CompileError(
                f"input_stream item_type {input_stream.item_type!r} does "
                "not match the entry boundary item_type "
                f"{cp.boundary.item_type.value!r}"
            )


def _reject_nonfinite(
    cp: CompiledPipeline,
    start: Descriptor,
    source_io: Mapping[str, ParamValue] | None,
    input_stream: Bitstream | Symbolstream | None,
) -> PipelineResult | None:
    if isinstance(input_stream, Symbolstream) and input_stream.item_type == "f":
        return _nonfinite_input(input_stream.path, "f")
    if cp.gr is not None and source_io:
        src_slice = SourceSlice.from_params(source_io)
        return _nonfinite_input(
            src_slice.path, start.item_type, src_slice.offset, src_slice.length
        )
    return None


def run_rx(
    modem: Modem,
    registry: Mapping[str, Stage[Any, Any]],
    *,
    sample_rate: float,
    start: Descriptor,
    workdir: Path,
    source_io: Mapping[str, ParamValue] | None = None,
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
            source_io=source_io or {},
            sink_io={"path": str(seam)},
            name=modem.name,
            quality_tap=True,
            trace_dir=trace_dir,
        )
        _validate_entry(cp, input_stream)
        flagged = _reject_nonfinite(cp, start, source_io, input_stream)
        if flagged is not None:
            return flagged
        census: list[BlockCensus] = []
        diagnostics: list[Diagnostic] = []
        trace_rows: list[TraceStage] = []
        marks: list[int] = []
        entry_path = seam
        if cp.gr is not None:
            if backend is None:
                from marconi.engine.backends.gnuradio.runner import GnuRadioBackend

                backend = GnuRadioBackend()
            check_deadline()
            r = backend.run_pipeline(cp.gr, timeout=remaining())
            census, diagnostics = list(r.census), list(r.diagnostics)
            if trace_dir is not None:
                trace_rows = _harvest_trace(modem, cp, trace_dir)
            if r.status != "ok":
                return PipelineResult(
                    status=r.status,
                    error=r.error,
                    stalled_at=r.stalled_at,
                    census=census,
                    diagnostics=diagnostics,
                    trace=trace_rows,
                )
            marks = _harvest_marks(diagnostics)
        if input_stream is not None:
            entry_path = input_stream.path
            if isinstance(input_stream, Symbolstream):
                marks = list(input_stream.marks)
        check_deadline()
        if cp.coding is None:
            result = _wrap_gr_only(cp, seam, marks, census, diagnostics)
        else:
            carrier = _entry_carrier(cp.boundary, entry_path, marks)
            out = run_coding(cp.coding, carrier, census)
            result = _flag_empty_coding(
                _wrap_result(cp.final, out, workdir, marks, census, diagnostics)
            )
        if result.status != "ok":
            return result
        check_deadline()
        soft = _soft_tap(cp, result, seam, input_stream)
        quality = assess_quality(
            registry=registry,
            census=result.census,
            diagnostics=result.diagnostics,
            marks=result.marks,
            soft_stream=soft.path if soft is not None else None,
            soft_decode_grade=soft.level == "bits" if soft is not None else True,
        )
        return result.model_copy(
            update={
                "softstream": soft,
                "quality": quality,
                "hints": _hints(modem, registry, cp.final, quality.verdict),
                "trace": trace_rows,
            }
        )
