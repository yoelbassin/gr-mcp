from typing import Any, Dict, List

from gnuradio_mcp.middlewares.platform import PlatformMiddleware
from gnuradio_mcp.models import (
    SINK,
    SOURCE,
    BlockModel,
    BlockTypeModel,
    ConnectionModel,
    ErrorModel,
    ParamModel,
    PortModel,
)
from gnuradio_mcp.utils import get_port_by_key


class PlatformProvider:
    def __init__(self, platform_mw: PlatformMiddleware, flowgraph_path: str = ""):
        self._platform_mw = platform_mw
        self._flowgraph_mw = platform_mw.make_flowgraph(flowgraph_path)

    ##############################################
    # Flowgraph Management
    ##############################################

    def get_blocks(self) -> list[BlockModel]:
        return self._flowgraph_mw.blocks

    def make_block(self, block_name: str) -> str:
        block_mw = self._flowgraph_mw.add_block(block_name)
        return block_mw.name

    def remove_block(self, block_name: str) -> bool:
        self._flowgraph_mw.remove_block(block_name)
        return True

    ##############################################
    # Block Management
    ##############################################

    def get_block_params(self, block_name: str) -> List[ParamModel]:
        return self._flowgraph_mw.get_block(block_name).params

    def set_block_params(self, block_name: str, params: Dict[str, Any]) -> bool:
        self._flowgraph_mw.get_block(block_name).set_params(params)
        return True

    def get_block_sources(self, block_name: str) -> list[PortModel]:
        return self._flowgraph_mw.get_block(block_name).sources

    def get_block_sinks(self, block_name: str) -> list[PortModel]:
        return self._flowgraph_mw.get_block(block_name).sinks

    ##############################################
    # Connection Management
    ##############################################

    def get_connections(self) -> list[ConnectionModel]:
        return self._flowgraph_mw.get_connections()

    def connect_blocks(
        self,
        source_block_name: str,
        sink_block_name: str,
        source_port_name: str,
        sink_port_name: str,
    ) -> bool:
        source_port = get_port_by_key(
            self._flowgraph_mw, source_block_name, source_port_name, SOURCE
        )
        sink_port = get_port_by_key(
            self._flowgraph_mw, sink_block_name, sink_port_name, SINK
        )
        self._flowgraph_mw.connect_blocks(source_port, sink_port)
        return True

    def disconnect_blocks(self, source_port: PortModel, sink_port: PortModel) -> bool:
        self._flowgraph_mw.disconnect_blocks(source_port, sink_port)
        return True

    ##############################################
    # Flowgraph Validation
    ##############################################

    def validate_block(self, block_name: str) -> bool:
        return self._flowgraph_mw.get_block(block_name).validate()

    def validate_flowgraph(self) -> bool:
        return self._flowgraph_mw.validate()

    def get_all_errors(self) -> list[ErrorModel]:
        return self._flowgraph_mw.get_all_errors()

    ##############################################
    # Platform Management
    ##############################################

    def get_all_available_blocks(self) -> list[BlockTypeModel]:
        return self._platform_mw.blocks

    def save_flowgraph(self, filepath: str) -> bool:
        self._platform_mw.save_flowgraph(filepath, self._flowgraph_mw)
        return True
