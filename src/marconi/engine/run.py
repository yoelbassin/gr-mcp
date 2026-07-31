from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
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
)
from marconi.engine.io.bitfile import (
    read_bits,
    read_symbols,
    write_bits,
    write_llrs,
    write_symbols,
)
from marconi.engine.stages.base import Stage
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem, Symbolstream
from marconi.engine.types.params import ParamValue


class PipelineResult(BaseModel):
    status: Literal["ok", "error", "timeout", "empty"]
    bitstream: Bitstream | None = None
    symbolstream: Symbolstream | None = None
    windows: list[int] = []
    marks: list[int] = []
    census: list[BlockCensus] = []
    diagnostics: list[Diagnostic] = []
    stalled_at: str | None = None
    error: str | None = None

    def diagnostic(self, block: str, key: str) -> Diagnostic | None:
        return find_diagnostic(self.diagnostics, block, key)


def _harvest_marks(diagnostics: Sequence[Diagnostic]) -> list[int]:
    marks: list[int] = []
    for d in diagnostics:
        if d.key == "bursts" and d.marks is not None:
            marks = [int(m) for m in d.marks]
    return marks


def _entry_carrier(boundary: Descriptor, path: Path, marks: list[int]) -> CodingCarrier:
    if boundary.item_type == "b":
        return CodingCarrier(bits=read_bits(path), marks=tuple(marks))
    item_type = cast(Literal["s", "f"], boundary.item_type)
    return CodingCarrier(
        bits=np.zeros(0, np.uint8),
        symbols=read_symbols(path, item_type),
        marks=tuple(marks),
    )


def _wrap_gr_only(
    cp: CompiledPipeline,
    seam: Path,
    marks: list[int],
    census: list[BlockCensus],
    diagnostics: list[Diagnostic],
) -> PipelineResult:
    if cp.final.item_type == "b":
        bitstream = Bitstream(path=seam, num_bits=int(read_bits(seam).size))
        return PipelineResult(
            status="ok",
            bitstream=bitstream,
            marks=marks,
            census=census,
            diagnostics=diagnostics,
        )
    item_type = cast(Literal["s", "f"], cp.final.item_type)
    symbols = read_symbols(seam, item_type)
    symbolstream = Symbolstream(
        path=seam, num_symbols=int(symbols.size), item_type=item_type, marks=marks
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
    item_type = cast(Literal["s", "f"], final.item_type)
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


def _first_nonfinite(path: Path, item_type: str) -> int | None:
    """Bounded-memory scan of a float-format input file; integer formats
    cannot hold non-finite values and are never scanned."""
    dtype = _SCAN_DTYPES.get(item_type)
    if dtype is None or not path.is_file():
        return None
    offset = 0
    with path.open("rb") as f:
        while True:
            block: np.ndarray = np.fromfile(f, dtype=dtype, count=_SCAN_CHUNK)
            if block.size == 0:
                return None
            bad = ~np.isfinite(block)
            if bad.any():
                return offset + int(bad.argmax())
            offset += int(block.size)


def _nonfinite_input(path: Path, item_type: str) -> PipelineResult | None:
    bad = _first_nonfinite(path, item_type)
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
    timeout: float = 180.0,
) -> PipelineResult:
    seam = workdir / "seam.dat"
    cp = compile_pipeline(
        modem,
        registry,
        direction="rx",
        sample_rate=sample_rate,
        start=start,
        source_io=source_io or {},
        sink_io={"path": str(seam)},
        name=modem.name,
    )
    if cp.final.level is Level.IQ:
        raise CompileError(
            "rx pipeline ends at IQ; run_rx extracts symbol/bit streams — end "
            "the path with a demodulation stage (for conditioned-IQ output use "
            "compile_modem and run the backend directly)"
        )
    if cp.final.level is Level.AUDIO:
        raise CompileError(
            "rx pipeline ends at AUDIO; run_rx extracts symbol/bit streams — "
            "continue the path with a demodulation stage (for audio output "
            "use compile_modem and run the backend directly)"
        )
    if cp.final.level is Level.SYMBOLS and cp.final.item_type == "c":
        raise CompileError(
            "rx pipeline ends at complex soft symbols, an interior seam with "
            "no stream format; end the path with a demap stage (for raw "
            "symbol output use compile_modem and run the backend directly)"
        )
    if cp.gr is not None and input_stream is not None:
        raise ValueError(
            "this modem's path has a GR segment fed by source_io; input_stream "
            "only enters a path that starts with a coding stage"
        )
    if cp.gr is None and input_stream is None:
        raise ValueError(
            "this modem's path starts with a coding stage, so no GR segment "
            "writes the seam file; supply input_stream"
        )
    if isinstance(input_stream, Bitstream) and cp.boundary.item_type != "b":
        raise ValueError(
            f"input_stream is a Bitstream (item_type 'b') but the entry "
            f"boundary is item_type {cp.boundary.item_type.value!r}"
        )
    if isinstance(input_stream, Symbolstream):
        if cp.boundary.item_type not in ("s", "f"):
            raise ValueError(
                f"input_stream is a Symbolstream (item_type "
                f"{input_stream.item_type!r}) but the entry boundary is item_type "
                f"{cp.boundary.item_type.value!r}"
            )
        if input_stream.item_type != cp.boundary.item_type:
            raise ValueError(
                f"input_stream item_type {input_stream.item_type!r} does not "
                f"match the entry boundary item_type {cp.boundary.item_type.value!r}"
            )
    if isinstance(input_stream, Symbolstream) and input_stream.item_type == "f":
        flagged = _nonfinite_input(input_stream.path, "f")
        if flagged is not None:
            return flagged
    if cp.gr is not None:
        src = (source_io or {}).get("path")
        if isinstance(src, str):
            flagged = _nonfinite_input(Path(src), start.item_type)
            if flagged is not None:
                return flagged
    census: list[BlockCensus] = []
    diagnostics: list[Diagnostic] = []
    marks: list[int] = []
    entry_path = seam
    if cp.gr is not None:
        if backend is None:
            from marconi.engine.backends.gnuradio.runner import GnuRadioBackend

            backend = GnuRadioBackend()
        r = backend.run_pipeline(cp.gr, timeout=timeout)
        census, diagnostics = list(r.census), list(r.diagnostics)
        if r.status != "ok":
            return PipelineResult(
                status=r.status,
                error=r.error,
                stalled_at=r.stalled_at,
                census=census,
                diagnostics=diagnostics,
            )
        marks = _harvest_marks(diagnostics)
    if input_stream is not None:
        entry_path = input_stream.path
        if isinstance(input_stream, Symbolstream):
            marks = list(input_stream.marks)
    if cp.coding is None:
        return _wrap_gr_only(cp, seam, marks, census, diagnostics)
    carrier = _entry_carrier(cp.boundary, entry_path, marks)
    out = run_coding(cp.coding, carrier, census)
    return _flag_empty_coding(
        _wrap_result(cp.final, out, workdir, marks, census, diagnostics)
    )
