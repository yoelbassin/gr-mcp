from __future__ import annotations

from gnuradio.grc.core.platform import Platform

from gnuradio_mcp.middlewares.base import ElementMiddleware
from gnuradio_mcp.middlewares.flowgraph import FlowGraphMiddleware
from gnuradio_mcp.models import BlockModel


class PlatformMiddleware(ElementMiddleware):
    def __init__(self, platform: Platform):
        super().__init__(platform)
        self._platform = self._element

    @property
    def blocks(self) -> list[BlockModel]:
        return [
            BlockModel.from_block(block) for block in self._platform.blocks.values()
        ]

    def make_flowgraph(self, filepath: str = "") -> FlowGraphMiddleware:
        return FlowGraphMiddleware.from_file(self, filepath)
