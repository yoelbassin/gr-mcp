from __future__ import annotations

from gnuradio.grc.core.blocks.block import Block
from gnuradio.grc.core.platform import Platform

from gnuradio_mcp.middlewares.platform import BlockTypeModel, PlatformMiddleware


def test_block_model_from_block(platform: Platform):
    block_type = Block
    model = BlockTypeModel.from_block_type(block_type)
    assert model.label == block_type.label
    assert model.key == block_type.key


def test_platform_middleware_blocks(platform: Platform):
    middleware = PlatformMiddleware(platform)
    block_models = middleware.blocks
    assert block_models  # Checks that the list is not empty
    assert all(isinstance(block_model, BlockTypeModel) for block_model in block_models)
