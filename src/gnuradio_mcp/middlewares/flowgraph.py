
from typing import List, Optional
from gnuradio.grc.core.FlowGraph import FlowGraph
from gnuradio_mcp.models import BlockModel


class FlowGraphMiddleware:
    def __init__(self, flowgraph: FlowGraph):
        self._flowgraph = flowgraph

    @property
    def blocks(self) -> List[BlockModel]:
        return [
            BlockModel(key=block.key, label=block.label)
            for block in self._flowgraph.blocks
        ]
    
    def add_block(self, key: str, name) -> None:
        block = self._flowgraph.new_block(key)
        block.params["id"].set_value(name)
