from __future__ import annotations

import pytest
from gnuradio.grc.core.blocks.block import Block
from gnuradio.grc.core.platform import Platform

from gnuradio_mcp.middlewares.block import BlockMiddleware
from gnuradio_mcp.middlewares.flowgraph import FlowGraphMiddleware
from gnuradio_mcp.models import SINK, SOURCE, ParamModel


@pytest.fixture
def flowgraph_middleware(platform: Platform) -> FlowGraphMiddleware:
    flowgraph = platform.make_flow_graph("")
    return FlowGraphMiddleware(flowgraph)


@pytest.fixture
def block_middleware(
    flowgraph_middleware: FlowGraphMiddleware, block_key: str
) -> BlockMiddleware:
    return flowgraph_middleware.add_block(block_key)


def test_block_middleware_params(block_middleware: BlockMiddleware):
    check_param_models(block_middleware._block, block_middleware.params)


def test_block_middleware_sinks(block_middleware: BlockMiddleware):
    check_port_models(block_middleware.sinks, block_middleware._block.sinks, SINK)


def test_block_middleware_sources(block_middleware: BlockMiddleware):
    check_port_models(block_middleware.sources, block_middleware._block.sources, SOURCE)


def test_block_middleware_set_param(block_middleware: BlockMiddleware):
    block_middleware.set_param("id", "my_custom_block_name")
    assert block_middleware._block.params["id"].get_value() == "my_custom_block_name"


def test_block_middleware_set_params(block_middleware: BlockMiddleware):
    block_middleware.set_params({"id": "my_custom_block_name"})
    assert block_middleware._block.params["id"].get_value() == "my_custom_block_name"


def check_param_models(block: Block, params: list[ParamModel]):
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
        assert model.key == int(port.key)
        assert model.name == port.name
        assert model.dtype == port.dtype
        assert model.direction == direction
        assert model.optional == port.optional
