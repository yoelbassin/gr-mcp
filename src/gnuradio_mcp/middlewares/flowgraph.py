from typing import List, Optional
from gnuradio.grc.core.FlowGraph import FlowGraph
from gnuradio_mcp.middlewares.block import BlockMiddleware
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

    def add_block(self, block_type: str, block_name: str) -> None:
        block = self._flowgraph.new_block(block_type)
        block.params["id"].set_value(block_name)

    def remove_block(self, block_name: str) -> None:
        block = self._flowgraph.get_block(block_name)
        self._flowgraph.remove_element(block)

    def get_block(self, block_name: str) -> BlockMiddleware:
        block = self._flowgraph.get_block(block_name)
        return BlockMiddleware(block)
