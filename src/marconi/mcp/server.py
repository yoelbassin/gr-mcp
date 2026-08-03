from __future__ import annotations

from typing import TYPE_CHECKING

from marconi.mcp.tools import TOOLS

if TYPE_CHECKING:
    from fastmcp import FastMCP


def build_server() -> "FastMCP":
    from fastmcp import FastMCP

    mcp: FastMCP = FastMCP("marconi")
    for name, fn in TOOLS.items():
        mcp.tool(name=name)(fn)
    return mcp


def main() -> None:
    try:
        mcp = build_server()
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "the 'mcp' extra is required to run marconi-mcp: install it with "
            "`pip install marconi[mcp]` (or `uv sync --extra mcp`)"
        ) from exc

    from marconi.engine.backends.gnuradio.runner import ensure_worker_warm

    ensure_worker_warm()
    mcp.run()
