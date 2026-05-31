from __future__ import annotations

import os
import subprocess
import sys

# uv's isolated venv can't see system site-packages where GNU Radio lives.
# Fall back to the base Python (which can) to locate and inject the gnuradio path.
try:
    import gnuradio  # noqa: F401
except ImportError:
    _base_python = os.path.join(sys.base_prefix, "bin", "python3")
    result = subprocess.run(
        [
            _base_python,
            "-c",
            "import gnuradio,os;p=os.path.dirname;" "print(p(p(gnuradio.__file__)))",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and (site := result.stdout.strip()):
        sys.path.insert(0, site)

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

app: FastMCP = FastMCP("GNU Radio MCP", instructions="Create GNU Radio flowgraphs")

McpPlatformProvider.from_platform_middleware(app, PlatformMiddleware(platform))

if __name__ == "__main__":
    app.run()
