from __future__ import annotations

from typing import Dict, Optional

from gnuradio.grc.core.blocks.block import Block
from gnuradio.grc.core.FlowGraph import FlowGraph

from gnuradio_mcp.middlewares.block import BlockMiddleware
from gnuradio_mcp.models import BlockModel, ConnectionModel, PortModel
from gnuradio_mcp.utils import get_unique_id


class FlowGraphMiddleware:
    def __init__(self, flowgraph: FlowGraph):
        self._flowgraph = flowgraph
        self._blocks: Dict[str, BlockMiddleware] = {}

    @property
    def blocks(self) -> list[BlockModel]:
        return [
            BlockModel(key=block.key, label=block.label)
            for block in self._flowgraph.blocks
        ]

    def add_block(
        self, block_type: str, block_name: Optional[str] = None
    ) -> BlockMiddleware:
        block_name = block_name or get_unique_id(self._flowgraph.blocks)
        block = self._flowgraph.new_block(block_type)
        block.params["id"].set_value(block_name)
        self._blocks[block_name] = BlockMiddleware(block)
        return self._blocks[block_name]

    def remove_block(self, block_name: str) -> None:
        block = self._flowgraph.get_block(block_name)
        self._flowgraph.remove_element(block)
        del self._blocks[block_name]

    def get_block(self, block_name: str) -> BlockMiddleware:
        # TODO: Check if calling two times you get different results
        return self._blocks[block_name]

    def connect_blocks(
        self, src_port_model: PortModel, dst_port_model: PortModel
    ) -> None:
        def get_block_by_port_model(port_model: PortModel) -> Block:
            return self._flowgraph.get_block(port_model.parent)

        src_port = get_block_by_port_model(src_port_model).sources[src_port_model.key]
        dst_port = get_block_by_port_model(dst_port_model).sinks[dst_port_model.key]
        self._flowgraph.connect(src_port, dst_port)

    def disconnect_blocks(
        self, src_port_model: PortModel, dst_port_model: PortModel
    ) -> None:
        def get_block_by_port_model(port_model: PortModel) -> Block:
            return self._flowgraph.get_block(port_model.parent)

        src_port = get_block_by_port_model(src_port_model).sources[src_port_model.key]
        dst_port = get_block_by_port_model(dst_port_model).sinks[dst_port_model.key]
        self._flowgraph.disconnect(src_port, dst_port)

    def get_connections(self) -> list[ConnectionModel]:
        return [
            ConnectionModel.from_connection(connection)
            for connection in self._flowgraph.connections
        ]
