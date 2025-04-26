from typing import List, Optional
from gnuradio.grc.core.platform import Platform
from gnuradio.grc.core.FlowGraph import FlowGraph
from gnuradio_mcp.middlewares.flowgraph import FlowGraphMiddleware
from gnuradio_mcp.models import BlockModel

class PlatformMiddleware:
    def __init__(self, platform: Platform):
        self._platform = platform
        flowgraph = self._platform.make_flow_graph("")
        self._flowgraph_mw = FlowGraphMiddleware(flowgraph)

    @property
    def blocks(self) -> List[BlockModel]:
        return [BlockModel.from_block(block) for block in self._platform.blocks.values()]
    
    @property
    def flowgraph(self) -> FlowGraphMiddleware:
        return self._flowgraph_mw
