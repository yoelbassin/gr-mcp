from __future__ import annotations

from fastmcp import FastMCP

from gnuradio_mcp.middlewares.platform import PlatformMiddleware
from gnuradio_mcp.providers.mcp import McpPlatformProvider

try:
    from gnuradio import gr
    from gnuradio.grc.core.platform import Platform
except ImportError:
    raise Exception("Cannot find GNU Radio!") from None

platform = Platform(
    version=gr.version(),
    version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()),
    prefs=gr.prefs(),
)
platform.build_library()

app: FastMCP = FastMCP(
    "GNU Radio MCP", description="Provide a MCP interface to GNU Radio"
)

McpPlatformProvider.from_platform_middleware(app, PlatformMiddleware(platform))

if __name__ == "__main__":
    app.run()
