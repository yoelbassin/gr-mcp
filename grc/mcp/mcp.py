from itertools import count
import re
from typing import Any, Dict, List, Optional, Set

from grc.core.FlowGraph import FlowGraph
from grc.core.blocks.block import Block
from grc.core.params.param import Param
from grc.core.Connection import Connection
from grc.core.platform import Platform
from grc.core.ports.port import Port


class CoreMiddleware:
    def __init__(self, platform: Platform, filepath: Optional[str] = ""):

        self._platform = platform

        initial_state = platform.parse_flow_graph(filepath)
        self._flowgraph = FlowGraph(platform)
        self._flowgraph.import_data(initial_state)

    ##############################################
    # Static Data
    ##############################################

    def list_all_blocks(self) -> List[Dict[str, Any]]:
        blocks_data = []
        for block in self._platform.blocks.values():
            blocks_data.append(
                {
                    "key": block.key,
                    "label": block.label,
                }
            )
        return blocks_data

    def list_block_parameters_data(self, block_key: str) -> List[str]:
        block = self._platform.blocks[block_key]
        return block.parameters_data

    def list_block_output_data(self, block_key: str) -> List[Dict[str, Any]]:
        block = self._platform.blocks[block_key]
        return block.outputs_data

    def list_block_input_data(self, block_key: str) -> List[Dict[str, Any]]:
        block = self._platform.blocks[block_key]
        return block.inputs_data

    ##############################################
    # Flowgraph Data
    ##############################################

    def get_placed_blocks(self) -> List[Dict[str, Any]]:
        blocks_data = []
        for block in self._flowgraph.blocks:
            blocks_data.append(
                {
                    "key": block.key,
                    "name": block.params["id"].get_value(),
                }
            )
        return blocks_data

    def get_placed_block_params(self, block_name: str) -> Dict[str, Any]:
        block = self._flowgraph.get_block(block_name)
        return {param.key: param.get_value() for param in block.params.values()}

    def get_placed_connections(self) -> Set[Dict[str, Any]]:
        connections_data = []
        for connection in self._flowgraph.connections:
            connection: Connection = connection
            source_port: Port = connection.source_port
            sink_port: Port = connection.sink_port
            connections_data.append(
                {
                    "src_block": source_port.parent.key,
                    "src_port": source_port.key,
                    "dst_block": sink_port.parent.key,
                    "dst_port": sink_port.key,
                }
            )
        return connections_data

    ##############################################
    # Flowgraph Operations
    ##############################################

    def add_block(self, block_key: str, name: Optional[str] = None) -> Block:
        name = name or self._get_unique_id(block_key)
        block = self._flowgraph.new_block(block_key)
        block.params["id"].set_value(name)
        return "success"

    def remove_block(self, block_name: str) -> None:
        block = self._flowgraph.get_block(block_name)
        self._flowgraph.remove_element(block)
        return "success"

    def connect_blocks(
        self,
        src_block_name: str,
        src_port_index: int,
        dst_block_name: str,
        dst_port_index: int,
    ) -> Connection:
        src_block = self._flowgraph.get_block(src_block_name)
        dst_block = self._flowgraph.get_block(dst_block_name)
        self._flowgraph.connect(
            src_block.sources[src_port_index], dst_block.sinks[dst_port_index]
        )
        return "success"

    def disconnect_blocks(
        self,
        src_block_name: str,
        src_port_index: int,
        dst_block_name: str,
        dst_port_index: int,
    ) -> None:
        src_block = self._flowgraph.get_block(src_block_name)
        dst_block = self._flowgraph.get_block(dst_block_name)
        self._flowgraph.disconnect(
            src_block.sources[src_port_index], dst_block.sinks[dst_port_index]
        )
        return "success"

    def update_block_params(self, block_name: str, params: Dict[str, Any]) -> None:
        block = self._flowgraph.get_block(block_name)
        for param_name, param_value in params.items():
            block.params[param_name].set_value(param_value)
        return "success"

    ##############################################
    # Flowgraph Validation
    ##############################################

    def validate_block(self, block_name: str) -> bool:
        self._flowgraph.rewrite()
        block = self._flowgraph.get_block(block_name)
        block.validate()
        return block.is_valid()

    def validate_connection(
        self,
        src_block_name: str,
        src_port_index: int,
        dst_block_name: str,
        dst_port_index: int,
    ) -> bool:
        self._flowgraph.rewrite()
        connections: Set[Connection] = self._flowgraph.connections
        for connection in connections:
            if (
                connection.source_port.parent.key == src_block_name
                and connection.sink_port.parent.key == dst_block_name
                and connection.source_port.key == src_port_index
                and connection.sink_port.key == dst_port_index
            ):
                connection.validate()
                return connection.is_valid()
        raise ValueError("Connection not found")

    def validate_flowgraph(self) -> bool:
        self._flowgraph.rewrite()
        self._flowgraph.validate()
        return self._flowgraph.is_valid()

    def get_all_errors(self) -> List[Dict[str, Any]]:
        self._flowgraph.rewrite()
        self._flowgraph.validate()
        errors = []
        for elem, msg in self._flowgraph.iter_error_messages():
            msg = re.sub("[^A-Za-z0-9]+", " ", msg).strip()
            if isinstance(elem, Connection):
                connection: Connection = elem
                source_port: Port = elem.source_port
                sink_port: Port = elem.sink_port
                errors.append(
                    {
                        "type": "connection",
                        "key": {
                            "src_block": source_port.parent.key,
                            "src_port": source_port.key,
                            "dst_block": sink_port.parent.key,
                            "dst_port": sink_port.key,
                        },
                        "message": msg,
                    }
                )

            if isinstance(elem, Param):
                errors.append(
                    {
                        "type": "param",
                        "key": f"{elem.parent.params["id"].get_value()}:{elem.key}",
                        "message": msg,
                    }
                )

            if isinstance(elem, Port):
                errors.append(
                    {
                        "type": "port",
                        "key": f"{elem.parent.params["id"].get_value()}:{elem.key}",
                        "message": msg,
                    }
                )

            if isinstance(elem, Block):
                errors.append(
                    {
                        "type": "block",
                        "key": elem.params["id"].get_value(),
                        "message": msg,
                    }
                )

        return errors

    ##############################################
    # Misc
    ##############################################

    def save_flowgraph(self, filepath: str) -> None:
        self._platform.save_flow_graph(filepath, self._flowgraph)
        return "success"

    ##############################################
    # Helper Functions
    ##############################################

    def _get_unique_id(self, base_id=""):
        """
        Get a unique id starting with the base id.

        Args:
            base_id: the id starts with this and appends a count

        Returns:
            a unique id
        """
        block_ids = set(b.name for b in self._flowgraph.blocks)
        for index in count():
            block_id = "{}_{}".format(base_id, index)
            if block_id not in block_ids:
                break
        return block_id
