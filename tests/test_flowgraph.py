from __future__ import annotations

import pytest
from gnuradio.grc.core.platform import Platform

from gnuradio_mcp.middlewares.flowgraph import FlowGraphMiddleware
from gnuradio_mcp.models import BlockModel


@pytest.fixture
def flowgraph_middleware(platform: Platform):
    flowgraph = platform.make_flow_graph("")
    return FlowGraphMiddleware(flowgraph)


@pytest.fixture
def initial_blocks(flowgraph_middleware: FlowGraphMiddleware):
    return [
        BlockModel(key=block.key, label=block.label)
        for block in flowgraph_middleware._flowgraph.blocks
    ]


def test_flowgraph_block_addition_and_removal(
    flowgraph_middleware: FlowGraphMiddleware,
    platform: Platform,
    initial_blocks: list[BlockModel],
    block_key: str,
):
    block_keys = platform.blocks.keys()
    assert block_keys, "No blocks available in platform library."
    block_name = f"test_block_{block_key}"
    flowgraph_middleware.add_block(block_key, block_name)
    blocks = flowgraph_middleware.blocks
    assert all(b in blocks for b in initial_blocks)
    assert any(b.key == block_key for b in blocks)

    flowgraph_middleware._flowgraph.remove_element(
        flowgraph_middleware._flowgraph.get_block(block_name),
    )
    current_blocks = [
        BlockModel(key=block.key, label=block.label)
        for block in flowgraph_middleware._flowgraph.blocks
    ]
    assert current_blocks == initial_blocks


def test_flowgraph_initial_state(
    flowgraph_middleware: FlowGraphMiddleware,
    initial_blocks: list[BlockModel],
):
    assert flowgraph_middleware.blocks == initial_blocks


def test_block_naming(flowgraph_middleware: FlowGraphMiddleware, block_key: str):
    # Explicit name
    explicit_name = "my_custom_block_name"
    flowgraph_middleware.add_block(block_key, explicit_name)
    block = flowgraph_middleware._flowgraph.get_block(explicit_name)
    assert block.params["id"].get_value() == explicit_name
    # Remove for clean state
    flowgraph_middleware._flowgraph.remove_element(block)

    # Implicit name (should use get_unique_id logic)
    flowgraph_middleware.add_block(block_key)
    # The last block added should be the last in the blocks list
    last_block = flowgraph_middleware._flowgraph.blocks[-1]
    # The id param should match the block's name
    assert last_block.params["id"].get_value() == last_block.name
    # Remove for clean state
    flowgraph_middleware._flowgraph.remove_element(last_block)


def test_block_unique_names_for_same_type(
    flowgraph_middleware: FlowGraphMiddleware, block_key: str
):
    # Add two blocks of the same type without explicit names
    flowgraph_middleware.add_block(block_key)
    first_block = flowgraph_middleware._flowgraph.blocks[-1]
    flowgraph_middleware.add_block(block_key)
    second_block = flowgraph_middleware._flowgraph.blocks[-1]
    # They should have different names
    assert first_block.name != second_block.name
    assert first_block.params["id"].get_value() != second_block.params["id"].get_value()
    # Clean up
    flowgraph_middleware._flowgraph.remove_element(first_block)
    flowgraph_middleware._flowgraph.remove_element(second_block)
