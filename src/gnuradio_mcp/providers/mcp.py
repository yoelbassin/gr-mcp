from fastmcp import FastMCP

from gnuradio_mcp.middlewares.platform import PlatformMiddleware
from gnuradio_mcp.providers.base import PlatformProvider


class McpPlatformProvider:
    def __init__(self, mcp_instance: FastMCP, platform_provider: PlatformProvider):
        self._mcp_instance = mcp_instance
        self._platform_provider = platform_provider
        self.__init_tools()

    def __init_tools(self):
        self._mcp_instance.add_tool(self._platform_provider.get_blocks)
        self._mcp_instance.add_tool(self._platform_provider.make_block)
        self._mcp_instance.add_tool(self._platform_provider.remove_block)
        self._mcp_instance.add_tool(self._platform_provider.get_block_params)
        self._mcp_instance.add_tool(self._platform_provider.set_block_params)
        self._mcp_instance.add_tool(self._platform_provider.get_block_sources)
        self._mcp_instance.add_tool(self._platform_provider.get_block_sinks)
        self._mcp_instance.add_tool(self._platform_provider.get_connections)
        self._mcp_instance.add_tool(self._platform_provider.connect_blocks)
        self._mcp_instance.add_tool(self._platform_provider.disconnect_blocks)
        self._mcp_instance.add_tool(self._platform_provider.validate_block)
        self._mcp_instance.add_tool(self._platform_provider.validate_flowgraph)
        self._mcp_instance.add_tool(self._platform_provider.get_all_errors)
        self._mcp_instance.add_tool(self._platform_provider.save_flowgraph)
        self._mcp_instance.add_tool(self._platform_provider.get_all_available_blocks)

    @property
    def app(self) -> FastMCP:
        return self._mcp_instance

    @classmethod
    def from_platform_middleware(
        cls,
        mcp_instance: FastMCP,
        platform_middleware: PlatformMiddleware,
        flowgraph_path: str = "",
    ):
        platform_provider = PlatformProvider(platform_middleware, flowgraph_path)
        return cls(mcp_instance, platform_provider)
