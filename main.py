from __future__ import annotations

import sys

from gnuradio_mcp.middlewares.platform import PlatformMiddleware

# Load GNU Radio
try:
    from gnuradio import gr
except ImportError:
    # Throw a new exception with more information
    print(
        "Cannot find GNU Radio! (Have you sourced the environment file?)",
        file=sys.stderr,
    )
    # Throw the new exception
    raise Exception("Cannot find GNU Radio!") from None

from gnuradio.grc.core.platform import Platform

platform = Platform(
    version=gr.version(),
    version_parts=(gr.major_version(), gr.api_version(), gr.minor_version()),
    prefs=gr.prefs(),
)
platform.build_library()


platform_middleware = PlatformMiddleware(platform)
flowgraph_mw = platform_middleware.make_flowgraph()
flowgraph_mw.add_block("blocks_add_xx")
for error in flowgraph_mw.get_all_errors():
    print(error)
