from __future__ import annotations

from typing import Any, Literal, get_args

from gnuradio.grc.core.blocks.block import Block
from gnuradio.grc.core.params.param import Param
from gnuradio.grc.core.ports.port import Port
from pydantic import BaseModel


class BlockModel(BaseModel):
    label: str
    key: str

    @classmethod
    def from_block(cls, block: Block) -> BlockModel:
        return cls(label=block.label, key=block.key)


class ParamModel(BaseModel):
    parent: str
    key: str
    name: str
    dtype: str
    value: Any

    @classmethod
    def from_param(cls, param: Param) -> ParamModel:
        return cls(
            parent=param.parent.name,
            key=param.key,
            name=param.name,
            dtype=param.dtype,
            value=param.get_value(),
        )


DirectionType = Literal["sink", "source"]
SINK, SOURCE = get_args(DirectionType)


class PortModel(BaseModel):
    parent: str
    key: str
    name: str
    dtype: str
    direction: DirectionType
    optional: bool = False

    @classmethod
    def from_port(
        cls,
        port: Port,
        direction: DirectionType | None = None,
    ) -> PortModel:
        direction = direction or port._dir
        return cls(
            parent=port.parent.name,
            key=port.key,
            name=port.name,
            dtype=port.dtype,
            direction=direction,
            optional=port.optional,
        )
