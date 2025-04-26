from typing import List
from gnuradio.grc.core.blocks.block import Block

from gnuradio_mcp.models import SINK, SOURCE, ParamModel, PortModel


class BlockMiddleware:
    def __init__(self, block: Block):
        self._block = block

    @property
    def params(self) -> List[ParamModel]:
        return [ParamModel.from_param(param) for param in self._block.params.values()]

    @property
    def sinks(self) -> List[PortModel]:
        return [PortModel.from_port(port, SINK) for port in self._block.sinks]

    @property
    def sources(self) -> List[PortModel]:
        return [PortModel.from_port(port, SOURCE) for port in self._block.sources]
