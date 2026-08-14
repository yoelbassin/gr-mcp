from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastmcp import Client
from helpers import _synth as synth

from marconi.mcp.server import build_server


async def test_all_tools_are_registered() -> None:
    async with Client(build_server()) as client:
        names = {t.name for t in await client.list_tools()}
    assert names == {
        "explain",
        "describe_stages",
        "validate_modem",
        "run_rx",
        "read_stream",
        "stream_stats",
        "survey",
        "capture",
    }


async def test_a_poisoned_page_names_its_non_finite_items_on_the_wire(
    tmp_path: Path,
) -> None:
    # the in-process dict and the wire disagreed: render_page put python
    # nan/inf in the page (json.dumps writes the non-RFC tokens NaN/Infinity)
    # and the server's pydantic serializer turned every one of them into
    # `null` - unmarked, and with nan-vs-inf lost. Only a real client can
    # price what the agent actually receives.
    p = tmp_path / "poison.f32"
    np.array([1.5, np.nan, np.inf, -np.inf], np.float32).tofile(p)
    async with Client(build_server()) as client:
        res = await client.call_tool("read_stream", {"path": str(p)})
    page = res.structured_content
    assert page is not None
    assert page["values"] == [1.5, "nan", "inf", "-inf"]
    assert page["non_finite_items"] == 3
    text_bytes = sum(len(getattr(c, "text", "") or "") for c in res.content)
    assert text_bytes == 0


async def test_roundtrip_through_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    spec = {
        "symbol_rate": 1.0,
        "path": [{"conv": "fsk", "deviation": 1.0}, {"conv": "slice"}],
    }
    bits = np.random.default_rng(5).integers(0, 2, 512).astype(np.uint8)
    capture = synth.write(
        tmp_path / "fsk.cf32",
        synth.cpfsk(bits, sps=4, deviation=1.0, sample_rate=4.0),
    )
    async with Client(build_server()) as client:
        rx = (
            await client.call_tool(
                "run_rx",
                {
                    "spec": spec,
                    "sample_rate": 4.0,
                    "capture_path": str(capture),
                },
            )
        ).data
        assert rx["status"] == "ok"
        assert rx["quality"] is not None
        page = (
            await client.call_tool(
                "read_stream",
                {"path": rx["stream"]["path"], "count": 1024},
            )
        ).data
        # against the TRANSMITTED bits, not "is it a bit string" (true by
        # construction): the mutation "every page reads all-zeros" - a
        # previously shipped failure mode - passed the old assertion
        want = "".join(str(b) for b in bits)
        got = page["bits"]
        assert want[: len(got)] == got or want in got, (want[:40], got[:40])
