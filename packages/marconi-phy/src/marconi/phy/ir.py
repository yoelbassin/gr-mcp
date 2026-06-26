from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marconi.core.params import ParamValue


class GrBlock(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    kind: str
    params: dict[str, ParamValue] = {}


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
        ids = [b.id for b in self.blocks]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
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
