import re
from itertools import count

from gnuradio.grc.core.blocks.block import Block
from gnuradio.grc.core.Connection import Connection
from gnuradio.grc.core.params.param import Param
from gnuradio.grc.core.ports.port import Port
from pydantic import BaseModel

from gnuradio_mcp.models import (
    SINK,
    SOURCE,
    BlockModel,
    ConnectionModel,
    DirectionType,
    ErrorModel,
    ParamModel,
    PortModel,
)


def get_unique_id(flowgraph_blocks, base_id=""):
    block_ids = set(b.name for b in flowgraph_blocks)
    for index in count():
        block_id = "{}_{}".format(base_id, index)
        if block_id not in block_ids:
            break
    return block_id


def format_error_message(elem, msg) -> ErrorModel:
    msg = re.sub("[^A-Za-z0-9]+", " ", msg).strip()
    model: BaseModel
    match (elem):
        case Connection():
            model = ConnectionModel.from_connection(elem)

        case Param():
            model = ParamModel.from_param(elem)

        case Port():
            model = PortModel.from_port(elem)

        case Block():
            model = BlockModel.from_block(elem)

        case _:
            raise ValueError(f"Unsupported element type: {type(elem)}")
    return ErrorModel(
        type=type(model).__name__,
        key=model,  # type: ignore
        message=msg,
    )


def get_port_by_key_in_port_list(port_list: list[Port], key: str) -> Block:
    for port in port_list:
        if port.key == key:
            return port
    raise ValueError(f"Port not found: {key}")


def get_port_by_key(
    flowgraph, block_name: str, port_name: str, direction: DirectionType
) -> Port:
    block = flowgraph.get_block(block_name)
    if direction == SOURCE:
        return get_port_by_key_in_port_list(block.sources, port_name)
    elif direction == SINK:
        return get_port_by_key_in_port_list(block.sinks, port_name)
    else:
        raise ValueError(f"Invalid port direction: {direction}")


def get_port_from_port_model(flowgraph, port_model: PortModel) -> Port:
    block_from_port_model = flowgraph.get_block(port_model.parent)
    return get_port_by_key(
        flowgraph, block_from_port_model.name, port_model.key, port_model.direction
    )
