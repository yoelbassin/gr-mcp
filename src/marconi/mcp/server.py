from __future__ import annotations

from typing import TYPE_CHECKING

from marconi.mcp.tools import TOOLS

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastmcp import FastMCP


def build_server() -> "FastMCP":
    from fastmcp import FastMCP
    from fastmcp.tools.base import ToolResult

    mcp: FastMCP = FastMCP("marconi")
    for name, fn in TOOLS.items():
        mcp.tool(name=name, output_schema=None)(_structured_only(ToolResult, fn))
    return mcp


def _structured_only(
    tool_result: type, fn: "Callable[..., object]"
) -> "Callable[..., object]":
    """Wrap a tool so its payload ships ONCE. FastMCP auto-derives an output
    schema from the dict annotation and then sends every result as BOTH a
    TextContent JSON blob and structured_content - measured through a real
    client: describe_stages cost 14,327 wire bytes for a 7,314-byte payload,
    and a full symbol page 106,702 for 53,351. Every context budget in this
    tree prices the payload, so the duplication silently doubled all of them."""
    import functools

    @functools.wraps(fn)
    def wrapped(*args: object, **kwargs: object) -> object:
        return tool_result(structured_content=fn(*args, **kwargs), content=[])

    return wrapped


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
