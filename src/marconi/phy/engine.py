from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel

from marconi.core.bitfile import (
    read_bits,
    read_symbols,
    write_bits,
    write_llrs,
    write_symbols,
)
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
    status: Literal["ok", "error", "timeout", "empty"]
    bitstream: Bitstream | None = None
    symbolstream: Symbolstream | None = None
    windows: list[int] = []
    marks: list[int] = []
    census: list[BlockCensus] = []
    diagnostics: dict[str, dict[str, int | list[int]]] = {}
    stalled_at: str | None = None
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
            f"boundary is item_type {cp.boundary.item_type!r}"
        )
    if isinstance(input_stream, Symbolstream):
        if cp.boundary.item_type not in ("s", "f"):
            raise ValueError(
                f"input_stream is a Symbolstream (item_type "
                f"{input_stream.item_type!r}) but the entry boundary is item_type "
                f"{cp.boundary.item_type!r}"
            )
        if input_stream.item_type != cp.boundary.item_type:
            raise ValueError(
                f"input_stream item_type {input_stream.item_type!r} does not "
                f"match the entry boundary item_type {cp.boundary.item_type!r}"
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
    return _wrap_result(cp.final, out, workdir, marks, census, diagnostics)
