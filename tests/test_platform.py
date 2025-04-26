import pytest
from gnuradio_mcp.middlewares.platform import BlockModel, PlatformMiddleware
from gnuradio.grc.core.blocks.block import Block
from gnuradio.grc.core.platform import Platform
from gnuradio import gr


def test_block_model_from_block(platform: Platform):
    block = Block(platform)
    model = BlockModel.from_block(block)
    assert model.label == block.label
    assert model.key == block.key


def test_platform_middleware_blocks(platform: Platform):
    middleware = PlatformMiddleware(platform)
    block_models = middleware.blocks
    assert block_models  # Checks that the list is not empty
    assert all(isinstance(block_model, BlockModel) for block_model in block_models)
