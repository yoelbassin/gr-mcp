from __future__ import annotations

from gnuradio.grc.core.blocks.block import Block

from gnuradio_mcp.models import SINK, SOURCE, ParamModel, PortModel


class BlockMiddleware:
    def __init__(self, block: Block):
        self._block = block

    @property
    def params(self) -> list[ParamModel]:
        return [ParamModel.from_param(param) for param in self._block.params.values()]

    @property
    def sinks(self) -> list[PortModel]:
        return [PortModel.from_port(port, SINK) for port in self._block.sinks]

    @property
    def sources(self) -> list[PortModel]:
        return [PortModel.from_port(port, SOURCE) for port in self._block.sources]
