from typing import List
from gnuradio.grc.core.platform import Platform

from gnuradio_mcp.models import BlockModel

class PlatformMiddleware:
    def __init__(self, platform: Platform):
        self._platform = platform

    @property
    def blocks(self) -> List[BlockModel]:
        return [BlockModel.from_block(block) for block in self._platform.blocks.values()]
