import pytest
from gnuradio.grc.core.platform import Platform
from gnuradio import gr

@pytest.fixture(scope="module")
def platform() -> Platform:
    platform = Platform(
        version=gr.version(),
        version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()),
        prefs=gr.prefs(),
    )
    platform.build_library()
    return platform

@pytest.fixture(params=[1, 2, 10]) # Arbitrary number of blocks to test
def block_key(platform, request):
    block_keys = list(platform.blocks.keys())
    assert block_keys, "No blocks available in platform library."
    idx = request.param
    if idx < len(block_keys):
        return block_keys[idx]
    return block_keys[0] 