from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel

from marconi.core.bitfile import read_bits, read_symbols, write_bits, write_symbols
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream, Symbolstream
from marconi.core.params import ParamValue
from marconi.core.stages import Stage
from marconi.phy.backends.base import Backend, BlockCensus
from marconi.phy.coding.carrier import CodingCarrier
from marconi.phy.coding.program import run_coding
from marconi.phy.compiler import CompiledPipeline, compile_pipeline
from marconi.phy.models import ModemSpec


class PipelineResult(BaseModel):
    status: Literal["ok", "error", "timeout"]
    bitstream: Bitstream | None = None
    symbolstream: Symbolstream | None = None
    windows: list[int] = []
    marks: list[int] = []
    census: list[BlockCensus] = []
    diagnostics: dict[str, dict[str, int | list[int]]] = {}
    error: str | None = None


def _harvest_marks(diagnostics: Mapping[str, Mapping[str, Any]]) -> list[int]:
    marks: list[int] = []
    for d in diagnostics.values():
        b = d.get("bursts")
        if isinstance(b, list):
            marks = [int(m) for m in b]
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
    diagnostics: dict[str, dict[str, int | list[int]]],
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
    diagnostics: dict[str, dict[str, int | list[int]]],
) -> PipelineResult:
    windows = [w.start for w in carrier.windows]
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
    path = workdir / "out.i16"
    write_symbols(path, symbols)
    symbolstream = Symbolstream(
        path=path,
        num_symbols=int(np.asarray(symbols).size),
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


def run_rx(
    modem: ModemSpec,
    registry: Mapping[str, Stage[Any]],
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
    census: list[BlockCensus] = []
    diagnostics: dict[str, dict[str, int | list[int]]] = {}
    marks: list[int] = []
    entry_path = seam
    if cp.gr is not None:
        if backend is None:
            from marconi.phy.backends.gnuradio.runner import GnuRadioBackend

            backend = GnuRadioBackend()
        r = backend.run_pipeline(cp.gr, timeout=timeout)
        census, diagnostics = list(r.census), dict(r.diagnostics)
        if r.status != "ok":
            status: Literal["error", "timeout"] = (
                "timeout" if r.status == "timeout" else "error"
            )
            return PipelineResult(
                status=status, error=r.error, census=census, diagnostics=diagnostics
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
    return _wrap_result(cp.final, out, workdir, marks, census, diagnostics)
