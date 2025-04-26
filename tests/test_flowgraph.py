from typing import List
import pytest
from gnuradio.grc.core.platform import Platform
from gnuradio_mcp.middlewares.flowgraph import FlowGraphMiddleware
from gnuradio import gr
from gnuradio_mcp.models import BlockModel
from .utils import get_block_keys, add_block, remove_block, get_current_blocks


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
    initial_blocks: List[BlockModel],
    block_key: str,
):
    block_keys = get_block_keys(platform)
    assert block_keys, "No blocks available in platform library."
    block_name = f"test_block_{block_key}"
    add_block(flowgraph_middleware, block_key, block_name)
    blocks = flowgraph_middleware.blocks
    assert all(b in blocks for b in initial_blocks)
    assert any(b.key == block_key for b in blocks)

    remove_block(flowgraph_middleware, block_name)
    current_blocks = get_current_blocks(flowgraph_middleware)
    assert current_blocks == initial_blocks


def test_flowgraph_initial_state(
    flowgraph_middleware: FlowGraphMiddleware, initial_blocks: List[BlockModel]
):
    assert flowgraph_middleware.blocks == initial_blocks
