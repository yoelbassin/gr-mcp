from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marconi.engine.types.params import ParamValue


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

    @model_validator(mode="after")
    def _unique_block_ids(self) -> GrPipeline:
        counts = Counter(b.id for b in self.blocks)
        dupes = sorted(i for i, n in counts.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate block ids: {dupes}")
        return self

    @property
    def block_ids(self) -> list[str]:
        return [b.id for b in self.blocks]

    def block(self, block_id: str) -> GrBlock:
        for b in self.blocks:
            if b.id == block_id:
                return b
        raise KeyError(block_id)
