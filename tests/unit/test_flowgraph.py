from __future__ import annotations

from typing import Generator

import pytest

from gnuradio_mcp.middlewares.block import BlockMiddleware
from gnuradio_mcp.middlewares.flowgraph import FlowGraphMiddleware
from gnuradio_mcp.middlewares.platform import PlatformMiddleware
from gnuradio_mcp.models import BlockModel, ConnectionModel, ErrorModel


@pytest.fixture
def flowgraph_middleware(platform_middleware: PlatformMiddleware):
    return platform_middleware.make_flowgraph()


@pytest.fixture
def initial_blocks(flowgraph_middleware: FlowGraphMiddleware):
    return [
        BlockModel.from_block(block) for block in flowgraph_middleware._flowgraph.blocks
    ]


def test_flowgraph_block_addition_and_removal(
    flowgraph_middleware: FlowGraphMiddleware,
    initial_blocks: list[BlockModel],
    block_key: str,
):
    explicit_name = "my_custom_block_name"

    flowgraph_middleware.add_block(block_key, explicit_name)

    blocks = flowgraph_middleware.blocks
    assert all(b in blocks for b in initial_blocks)
    assert any(b.name == explicit_name for b in blocks)

    flowgraph_middleware.remove_block(explicit_name)

    blocks = flowgraph_middleware.blocks
    assert all(b in initial_blocks for b in blocks)


def test_flowgraph_initial_state(
    flowgraph_middleware: FlowGraphMiddleware,
    initial_blocks: list[BlockModel],
):
    assert flowgraph_middleware.blocks == initial_blocks


def test_block_naming(flowgraph_middleware: FlowGraphMiddleware, block_key: str):
    explicit_name = "my_custom_block_name"

    block = flowgraph_middleware.add_block(block_key, explicit_name)

    assert block.name == explicit_name


def test_block_unique_names_for_same_type(
    flowgraph_middleware: FlowGraphMiddleware, block_key: str
):
    first_block = flowgraph_middleware.add_block(block_key)
    second_block = flowgraph_middleware.add_block(block_key)

    assert first_block.name != second_block.name


def test_block_default_name(flowgraph_middleware: FlowGraphMiddleware, block_key: str):
    block_model = flowgraph_middleware.add_block(block_key)
    assert block_key in block_model.name


def test_remove_existing_block(flowgraph_middleware: FlowGraphMiddleware):
    DEFAULT_VARIABLE_BLOCK_NAME = "samp_rate"
    flowgraph_middleware.remove_block(DEFAULT_VARIABLE_BLOCK_NAME)
    assert not any(
        block.name == DEFAULT_VARIABLE_BLOCK_NAME
        for block in flowgraph_middleware.blocks
    )


@pytest.mark.parametrize(
    "block_key, sinks_number, sources_number",
    [("blocks_add_xx", 2, 1), ("blocks_copy", 1, 1), ("blocks_selector", 2, 2)],
)
def test_block_connections(
    flowgraph_middleware: FlowGraphMiddleware,
    block_key: str,
    sources_number: int,
    sinks_number: int,
):
    source_block, dest_block = create_and_connect_blocks(
        flowgraph_middleware, block_key, block_key
    )

    for connection in util_iter_possible_connections(source_block, dest_block):
        connections = flowgraph_middleware.get_connections()
        assert any(
            c.source.key == connection.source.key and c.sink.key == connection.sink.key
            for c in connections
        )

    assert len(flowgraph_middleware.get_connections()) == sources_number * sinks_number


def test_block_disconnection(flowgraph_middleware: FlowGraphMiddleware, block_key: str):
    source_block, dest_block = create_and_connect_blocks(
        flowgraph_middleware, block_key, block_key
    )

    for connection in util_iter_possible_connections(source_block, dest_block):
        flowgraph_middleware.disconnect_blocks(connection.source, connection.sink)
        connections = flowgraph_middleware.get_connections()
        assert not any(
            c.source.key == connection.source.key and c.sink.key == connection.sink.key
            for c in connections
        )

    assert len(flowgraph_middleware.get_connections()) == 0


def test_default_flowgraph_errors(flowgraph_middleware: FlowGraphMiddleware):
    for error in flowgraph_middleware.get_all_errors():
        assert isinstance(error, ErrorModel)
    assert len(flowgraph_middleware.get_all_errors()) == 0


def create_and_connect_blocks(
    flowgraph_middleware: FlowGraphMiddleware,
    source_block_key: str,
    dest_block_key: str,
):
    source_block_model = flowgraph_middleware.add_block(source_block_key)
    dest_block_model = flowgraph_middleware.add_block(dest_block_key)

    source_block = flowgraph_middleware.get_block(source_block_model.name)
    dest_block = flowgraph_middleware.get_block(dest_block_model.name)

    for connection in util_iter_possible_connections(source_block, dest_block):
        flowgraph_middleware.connect_blocks(connection.source, connection.sink)

    return source_block, dest_block


def util_iter_possible_connections(
    source_block: BlockMiddleware,
    dest_block: BlockMiddleware,
) -> Generator[ConnectionModel]:
    for sink in source_block.sinks:
        for source in dest_block.sources:
            yield ConnectionModel(source=source, sink=sink)
