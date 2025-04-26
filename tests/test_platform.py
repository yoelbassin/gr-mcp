import pytest
from gnuradio_mcp.middlewares.platform import BlockModel, PlatformMiddleware

class MockBlock:
    def __init__(self, label, key):
        self.label = label
        self.key = key

class MockPlatform:
    def __init__(self, blocks):
        self.blocks = {block.key: block for block in blocks}

def test_block_model_from_block():
    block = MockBlock(label="Test Block", key="test_key")
    model = BlockModel.from_block(block)
    assert model.label == "Test Block"
    assert model.key == "test_key"

def test_platform_middleware_blocks():
    blocks = [
        MockBlock(label="Block A", key="a"),
        MockBlock(label="Block B", key="b"),
    ]
    platform = MockPlatform(blocks)
    middleware = PlatformMiddleware(platform)
    block_models = middleware.blocks
    assert len(block_models) == 2
    assert block_models[0].label == "Block A"
    assert block_models[0].key == "a"
    assert block_models[1].label == "Block B"
    assert block_models[1].key == "b"
