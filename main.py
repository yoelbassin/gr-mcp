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
    # install_prefix=gr.prefix()
)
platform.build_library()


platform_middleware = PlatformMiddleware(platform)

print(platform_middleware.blocks)
