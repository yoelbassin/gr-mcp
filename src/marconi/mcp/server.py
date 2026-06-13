"""The FastMCP server: registers the tool functions and runs over stdio."""

from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

from marconi.mcp.state import ServerState, set_state
from marconi.mcp.tools import TOOLS
from marconi.workspace import Workspace


def build_server() -> FastMCP:
    mcp: FastMCP = FastMCP("marconi")
    for name, fn in TOOLS.items():
        mcp.tool(name=name)(fn)
    return mcp


def main() -> None:
    root = os.environ.get("MARCONI_WORKSPACE", ".")
    set_state(ServerState(Workspace(Path(root))))
    build_server().run()
