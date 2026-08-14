"""GrPipeline is a serialized document: every block reference inside it must
name a block that exists, or the failure surfaces downstream wearing the
wrong cause (a typo'd terminal_sink silently downgraded the empty-sink
verdict to ok; a typo'd connection endpoint died in the worker)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline


def _block(bid: str) -> GrBlock:
    return GrBlock(id=bid, kind="null_source_c")


def test_terminal_sink_must_name_a_block() -> None:
    with pytest.raises(ValidationError, match="terminal_sink"):
        GrPipeline(sample_rate=1e6, blocks=[_block("a")], terminal_sink="typo_sink")


def test_connection_endpoints_must_name_blocks() -> None:
    conn = GrConnection(src_block="a", dst_block="ghost")
    with pytest.raises(ValidationError, match="ghost"):
        GrPipeline(sample_rate=1e6, blocks=[_block("a")], connections=[conn])


def test_valid_references_pass() -> None:
    p = GrPipeline(
        sample_rate=1e6,
        blocks=[_block("a"), _block("b")],
        connections=[GrConnection(src_block="a", dst_block="b")],
        terminal_sink="b",
    )
    assert p.terminal_sink == "b"
