"""How a run reports the stream it produced: its length, its rung, and the
verdict when it produced nothing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.engine.backends.base import BlockCensus
from marconi.engine.compile.compiler import CompiledPipeline
from marconi.engine.io.bitfile import CaptureTooLarge
from marconi.engine.run import (
    PipelineResult,
    _flag_empty_stream,
    _gr_only_stream,
    _items_on_disk,
)
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType, RunStatus
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Symbolstream


def _cp(final: Descriptor) -> CompiledPipeline:
    return CompiledPipeline(
        gr=None,
        coding=None,
        boundary=final,
        final=final,
        boundaries=[final],
        rates=[1.0],
    )


def test_stream_length_comes_from_the_filesystem_not_a_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Taking .size off a full read was the same answer at whole-file cost, and
    read_bits enforces the bits-layer memory budget — so a GR-only decode that
    wrote more bits than that budget raised CaptureTooLarge at the REPORTING
    step of a run whose output was already on disk and pageable."""
    path = tmp_path / "seam.u8"
    path.write_bytes(bytes(np.zeros(5000, np.uint8)))

    def _refuse(_: Path) -> np.ndarray:
        raise CaptureTooLarge("a read must not happen here")

    monkeypatch.setattr("marconi.engine.run.read_bits", _refuse)
    stream = _gr_only_stream(_cp(Descriptor(Level.BITS, ItemType.B)), path, [])
    assert isinstance(stream, Bitstream)
    assert stream.num_bits == 5000


@pytest.mark.parametrize(
    "item_type, per_item", [(ItemType.C, 8), (ItemType.F, 4), (ItemType.S, 2)]
)
def test_items_on_disk_counts_whole_items(
    tmp_path: Path, item_type: ItemType, per_item: int
) -> None:
    path = tmp_path / f"x{item_type.suffix}"
    path.write_bytes(b"\0" * (per_item * 41))
    assert _items_on_disk(path, item_type) == 41


def test_a_soft_bits_terminal_reports_the_bits_rung(tmp_path: Path) -> None:
    """The same "f" wire is a per-symbol demod value at symbols and an LLR of
    the OPPOSITE sign convention at bits. A stream that cannot say which it is
    sends its reader to another field to find out."""
    path = tmp_path / "seam.f32"
    path.write_bytes(np.zeros(16, np.float32).tobytes())
    final = Descriptor(Level.BITS, ItemType.F, Carrier.SOFT)
    stream = _gr_only_stream(_cp(final), path, [])
    assert isinstance(stream, Symbolstream)
    assert stream.level is Level.BITS
    assert stream.item_type is ItemType.F


def test_an_iq_terminal_reports_the_iq_rung(tmp_path: Path) -> None:
    path = tmp_path / "seam.cf32"
    path.write_bytes(np.zeros(8, np.complex64).tobytes())
    stream = _gr_only_stream(_cp(Descriptor(Level.IQ, ItemType.C)), path, [])
    assert isinstance(stream, Symbolstream)
    assert stream.level is Level.IQ


def _ok_with(num_bits: int, census: list[BlockCensus]) -> PipelineResult:
    return PipelineResult(
        status=RunStatus.OK,
        bitstream=Bitstream(path=Path("out.u8"), num_bits=num_bits),
        census=census,
    )


def test_a_zero_item_stream_is_never_ok_even_with_no_census() -> None:
    """The GR-only tail used to lean entirely on the worker's empty-sink check,
    which returns early when the census is empty — and _harvest_census returns
    [] on any exception. A harvest failure reported 'ok' over zero items."""
    flagged = _flag_empty_stream(_ok_with(0, []))
    assert flagged.status is RunStatus.EMPTY
    assert flagged.bitstream is None
    assert flagged.error is not None


def test_a_zero_item_stream_names_where_the_signal_died() -> None:
    census = [
        BlockCensus(block="b0", kind="fsk", items_in=1000, items_out=500),
        BlockCensus(block="b1", kind="sync_word", items_in=500, items_out=0),
    ]
    flagged = _flag_empty_stream(_ok_with(0, census))
    assert flagged.status is RunStatus.EMPTY
    assert flagged.stalled_at == "b1"
    assert "sync_word" in (flagged.error or "")
    # the census gradient is exactly what a zero-decode run is read for
    assert flagged.census == census


def test_a_nonempty_stream_is_left_alone() -> None:
    result = _ok_with(64, [])
    assert _flag_empty_stream(result) is result
