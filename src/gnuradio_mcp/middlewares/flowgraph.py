from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from gnuradio.grc.core.blocks.block import Block
from gnuradio.grc.core.FlowGraph import FlowGraph

from gnuradio_mcp.middlewares.base import ElementMiddleware
from gnuradio_mcp.middlewares.block import BlockMiddleware
from gnuradio_mcp.models import (
    BlockModel,
    ConnectionModel,
    PortModel,
)
from gnuradio_mcp.utils import get_port_from_port_model, get_unique_id

if TYPE_CHECKING:
    from gnuradio_mcp.middlewares.platform import PlatformMiddleware


def set_block_name(block: Block, name: str):
    block.params["id"].set_value(name)


class FlowGraphMiddleware(ElementMiddleware):
    def __init__(self, flowgraph: FlowGraph):
        super().__init__(flowgraph)
        self._flowgraph = self._element

    @property
    def blocks(self) -> list[BlockModel]:
        return [BlockModel.from_block(block) for block in self._flowgraph.blocks]

    def add_block(
        self, block_type: str, block_name: Optional[str] = None
    ) -> BlockModel:
        block_name = block_name or get_unique_id(self._flowgraph.blocks, block_type)
        block = self._flowgraph.new_block(block_type)
        assert block is not None, f"Failed to create block: {block_type}"
        set_block_name(block, block_name)
        return BlockModel.from_block(block)

    def remove_block(self, block_name: str) -> None:
        block_middleware = self.get_block(block_name)
        self._flowgraph.remove_element(block_middleware._block)

    @lru_cache(maxsize=None)
    def get_block(self, block_name: str) -> BlockMiddleware:
        return BlockMiddleware(
            next(block for block in self._flowgraph.blocks if block.name == block_name)
        )

    def connect_blocks(
        self, src_port_model: PortModel, dst_port_model: PortModel
    ) -> None:
        src_port = get_port_from_port_model(self._flowgraph, src_port_model)
        dst_port = get_port_from_port_model(self._flowgraph, dst_port_model)
        self._flowgraph.connect(src_port, dst_port)

    def disconnect_blocks(
        self, src_port_model: PortModel, dst_port_model: PortModel
    ) -> None:
        src_port = get_port_from_port_model(self._flowgraph, src_port_model)
        dst_port = get_port_from_port_model(self._flowgraph, dst_port_model)
        self._flowgraph.disconnect(src_port, dst_port)

    def get_connections(self) -> list[ConnectionModel]:
        return [
            ConnectionModel.from_connection(connection)
            for connection in self._flowgraph.connections
        ]

    @classmethod
    def from_file(
        cls, platform: "PlatformMiddleware", filepath: str = ""
    ) -> FlowGraphMiddleware:
        initial_state = platform._platform.parse_flow_graph(filepath)
        flowgraph = FlowGraph(platform._platform)
        flowgraph.import_data(initial_state)
        return cls(flowgraph)
