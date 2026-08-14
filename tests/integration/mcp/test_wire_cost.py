"""What the agent actually receives, priced at the real client seam: FastMCP
sends a tool's dict result as BOTH a TextContent JSON blob and
structured_content, so every internal byte budget silently priced half the
wire (measured: describe_stages 14,327 wire bytes for a 7,314-byte payload,
a full symbol page 106,702 for 53,351)."""

from __future__ import annotations

import pytest
from fastmcp import Client

from marconi.mcp.server import build_server


@pytest.mark.asyncio
async def test_responses_ship_once() -> None:
    async with Client(build_server()) as client:
        res = await client.call_tool("describe_stages", {})
    assert res.structured_content, "the payload must arrive as structured content"
    text_bytes = sum(len(getattr(c, "text", "") or "") for c in res.content)
    assert (
        text_bytes == 0
    ), f"{text_bytes} bytes of TextContent duplicate the structured payload"
