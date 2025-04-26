from typing import List
import pytest
from gnuradio.grc.core.platform import Platform
from gnuradio.grc.core.blocks.block import Block
from gnuradio.grc.core.FlowGraph import FlowGraph
from gnuradio import gr
from gnuradio_mcp.middlewares.block import BlockMiddleware
from gnuradio_mcp.middlewares.flowgraph import FlowGraphMiddleware
from gnuradio_mcp.models import SINK, SOURCE, ParamModel


@pytest.fixture
def flowgraph_middleware(platform: Platform):
    flowgraph = platform.make_flow_graph("")
    return FlowGraphMiddleware(flowgraph)


@pytest.fixture
def block(flowgraph_middleware: FlowGraphMiddleware, block_key: str):
    flowgraph_middleware.add_block(block_key, block_key)
    return flowgraph_middleware._flowgraph.get_block(block_key)


def test_block_middleware_params(block: Block):
    middleware = BlockMiddleware(block)
    check_param_models(block, middleware.params)


def test_block_middleware_sinks(block: Block):
    middleware = BlockMiddleware(block)
    check_port_models(middleware.sinks, block.sinks, SINK)


def test_block_middleware_sources(block: Block):
    middleware = BlockMiddleware(block)
    check_port_models(middleware.sources, block.sources, SOURCE)


def check_param_models(block: Block, params: List[ParamModel]):
    assert params
    assert len(params) == len(block.params)
    for param in params:
        original_param = block.params[param.key]
        assert param.key == original_param.key
        assert param.name == original_param.name
        assert param.dtype == original_param.dtype
        assert param.value == original_param.value


def check_port_models(port_models, ports, direction):
    assert isinstance(port_models, list)
    assert len(port_models) == len(ports)
    for model, port in zip(port_models, ports):
        assert model.key == port.key
        assert model.name == port.name
        assert model.dtype == port.dtype
        assert model.direction == direction
        assert model.optional == port.optional
