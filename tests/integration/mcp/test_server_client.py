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
        assert set(page["bits"]) <= {"0", "1"}
