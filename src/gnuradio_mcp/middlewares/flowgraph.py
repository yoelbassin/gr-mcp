from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set

from gnuradio.grc.core.blocks.block import Block
from gnuradio.grc.core.FlowGraph import FlowGraph
from gnuradio.grc.core.ports.port import Port

from gnuradio_mcp.middlewares.base import ElementMiddleware
from gnuradio_mcp.middlewares.block import BlockMiddleware
from gnuradio_mcp.models import SINK, SOURCE, BlockModel, ConnectionModel, PortModel
from gnuradio_mcp.utils import get_unique_id

if TYPE_CHECKING:
    from gnuradio_mcp.middlewares.platform import PlatformMiddleware


def get_port_from_port_model_in_port_list(
    port_list: list[Port], port_model: PortModel
) -> Block:
    for port in port_list:
        if port.key == port_model.key:
            return port
    raise ValueError(f"Port not found: {port_model.key}")


def get_port_from_port_model(flowgraph, port_model: PortModel) -> Block:
    block_from_port_model = flowgraph.get_block(port_model.parent)
    if port_model.direction == SOURCE:
        return get_port_from_port_model_in_port_list(
            block_from_port_model.sources, port_model
        )
    elif port_model.direction == SINK:
        return get_port_from_port_model_in_port_list(
            block_from_port_model.sinks, port_model
        )
    else:
        raise ValueError(f"Invalid port direction: {port_model.direction}")


class FlowGraphMiddleware(ElementMiddleware):
    def __init__(self, flowgraph: FlowGraph):
        super().__init__(flowgraph)
        self._flowgraph = self._element
        self._blocks: Set[BlockMiddleware] = set()

    @property
    def blocks(self) -> list[BlockModel]:
        return [BlockModel.from_block(block) for block in self._flowgraph.blocks]

    def add_block(
        self, block_type: str, block_name: Optional[str] = None
    ) -> BlockMiddleware:
        block_name = block_name or get_unique_id(self._flowgraph.blocks, block_type)
        block = self._flowgraph.new_block(block_type)
        assert block is not None, f"Failed to create block: {block_type}"
        block_middleware = BlockMiddleware(block)
        block_middleware.name = block_name
        self._blocks.add(block_middleware)
        return block_middleware

    def remove_block(self, block_name: str) -> None:
        block_middleware = self.get_block(block_name)
        self._flowgraph.remove_element(block_middleware._block)
        self._blocks.remove(block_middleware)

    def get_block(self, block_name: str) -> BlockMiddleware:
        return next(block for block in self._blocks if block.name == block_name)

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
