import pytest
from gnuradio.grc.core.platform import Platform
from gnuradio_mcp.middlewares.flowgraph import FlowGraphMiddleware
from gnuradio import gr

from gnuradio_mcp.models import BlockModel

BLOCK_KEYS_TO_TEST = [1, 10]


@pytest.fixture(scope="module")
def platform() -> Platform:
    platform = Platform(
        version=gr.version(),
        version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()),
        prefs=gr.prefs(),
    )
    platform.build_library()
    return platform


@pytest.fixture
def flowgraph_middleware(platform):
    flowgraph = platform.make_flow_graph("")
    return FlowGraphMiddleware(flowgraph)


@pytest.fixture
def initial_blocks(flowgraph_middleware):
    return [
        BlockModel(key=block.key, label=block.label)
        for block in flowgraph_middleware._flowgraph.blocks
    ]


def get_block_keys(platform):
    return list(platform.blocks.keys())


def add_block(flowgraph_middleware, block_key, block_name):
    flowgraph_middleware.add_block(block_key, block_name)


def remove_block(flowgraph_middleware, block_name):
    flowgraph_middleware._flowgraph.remove_element(
        flowgraph_middleware._flowgraph.get_block(block_name)
    )


def get_current_blocks(flowgraph_middleware):
    return [
        BlockModel(key=block.key, label=block.label)
        for block in flowgraph_middleware._flowgraph.blocks
    ]


def test_blocks_match_initial_state(flowgraph_middleware, initial_blocks):
    assert flowgraph_middleware.blocks == initial_blocks


@pytest.mark.parametrize("block_index", BLOCK_KEYS_TO_TEST)
def test_add_block_preserves_and_adds(
    flowgraph_middleware, platform, initial_blocks, block_index
):
    block_keys = get_block_keys(platform)
    assert block_keys, "No blocks available in platform library."
    block_key = block_keys[block_index]
    block_name = f"test_block_{block_index}"
    add_block(flowgraph_middleware, block_key, block_name)
    blocks = flowgraph_middleware.blocks
    assert all(b in blocks for b in initial_blocks)
    assert any(b.key == block_key for b in blocks)


@pytest.mark.parametrize("block_index", BLOCK_KEYS_TO_TEST)
def test_remove_block_restores_initial(
    flowgraph_middleware, platform, initial_blocks, block_index
):
    block_keys = get_block_keys(platform)
    block_key = block_keys[block_index]
    block_name = f"block_to_remove_{block_index}"
    add_block(flowgraph_middleware, block_key, block_name)
    assert any(b.key == block_key for b in flowgraph_middleware.blocks)
    remove_block(flowgraph_middleware, block_name)
    current_blocks = get_current_blocks(flowgraph_middleware)
    assert current_blocks == initial_blocks
