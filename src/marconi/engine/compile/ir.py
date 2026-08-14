from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marconi.engine.types.enums import ItemType
from marconi.engine.types.params import ParamValue

# item_type -> (source_kind, sink_kind). The GR wire type alone selects IO;
# carrier is decision-hardness (a seam invariant), never a routing knob.
# Descriptor data, never stage names. It lives with the IR rather than in the
# compiler because it names the blocks the IR is allowed to contain, and both
# the compiler that emits them and the backend that recognises them read it.
_IO_BLOCKS: dict[ItemType, tuple[str | None, str]] = {
    ItemType.C: ("iq_file_source", "iq_file_sink"),
    ItemType.S: (None, "symbols_file_sink"),
    ItemType.B: ("bits_file_source", "bits_file_sink"),
    ItemType.F: ("soft_bits_file_source", "soft_bits_file_sink"),
}

# Derived, never retyped: the backend needs the same two sets to find the one
# file source (for the EOF probe) and to tell a sink from any other block.
# Held as three hand-written lists they could only diverge, and the failure is
# silent — an unrecognised source drops every chunked block's expected_items,
# so eof_final never fires and each withholds its tail forever, which reads as
# a short output stream rather than an error.
FILE_SOURCE_KINDS = frozenset(src for src, _ in _IO_BLOCKS.values() if src)
FILE_SINK_KINDS = frozenset(sink for _, sink in _IO_BLOCKS.values())


class GrBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    kind: str
    params: dict[str, ParamValue] = {}
    # Rate of the stream entering this block, from the compiler's rate model
    # (stage-input granularity). Consumed by the EOF-probe wiring to size a
    # downstream buffering block's finality; None when the compiler did not
    # annotate it. Advisory only — a wrong value can never force a premature
    # flush, only fall back to the withhold-at-EOF default.
    sample_rate: float | None = None


class GrConnection(BaseModel):
    model_config = ConfigDict(frozen=True)
    src_block: str
    src_port: int = 0
    dst_block: str
    dst_port: int = 0


class GrPipeline(BaseModel):
    """Compiler-internal IR: a GNU Radio block graph. NOT an authoring surface —
    constructed only by the compiler / dev-test code, consumed only by a Backend."""

    name: str = "pipeline"
    sample_rate: float = Field(gt=0)
    blocks: list[GrBlock] = []
    connections: list[GrConnection] = []
    # The block whose output IS the pipeline's product. Trace/soft taps share
    # the sink block kinds, so the empty-terminal verdict must key on this id,
    # never on "all sinks". None only for hand-built dev/test IR.
    terminal_sink: str | None = None

    @model_validator(mode="after")
    def _unique_block_ids(self) -> GrPipeline:
        counts = Counter(b.id for b in self.blocks)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate block ids: {dupes}")
        return self

    @model_validator(mode="after")
    def _references_resolve(self) -> GrPipeline:
        """Every block reference must name a block that exists, or the
        failure surfaces downstream wearing the wrong cause: a typo'd
        terminal_sink made the empty-sink verdict silently unreachable
        (worker.py's `if not sinks` amnesty), and a typo'd connection
        endpoint died in the worker instead of at validation."""
        ids = {b.id for b in self.blocks}
        if self.terminal_sink is not None and self.terminal_sink not in ids:
            raise ValueError(
                f"terminal_sink {self.terminal_sink!r} names no block; "
                f"blocks: {sorted(ids)}"
            )
        ghosts = sorted(
            {
                end
                for c in self.connections
                for end in (c.src_block, c.dst_block)
                if end not in ids
            }
        )
        if ghosts:
            raise ValueError(f"connection endpoints name no block: {ghosts}")
        return self

    @property
    def block_ids(self) -> list[str]:
        return [b.id for b in self.blocks]

    def block(self, block_id: str) -> GrBlock:
        for b in self.blocks:
            if b.id == block_id:
                return b
        raise KeyError(block_id)
