from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastmcp import Client
from helpers import framing
from helpers.assets import asset_fixture

from marconi.mcp.server import build_server

RATE = 128000.0
SC = 0x7CD215D8
SC_HEX = SC.to_bytes(4, "big").hex()
BCH_GEN = 0x769
CODE_BITS, DATA_BITS = 31, 21
NPAR = CODE_BITS - DATA_BITS
BATCH_CODEWORDS = 16
PERM = [slot * 32 + b for slot in range(BATCH_CODEWORDS) for b in range(CODE_BITS)]
IDLE = 0x7A89C197
IDLE_DATA = IDLE >> (NPAR + 1)

ORACLE = {240071: 3, 154320: 3, 151233: 3}


def _bch_parity_masks() -> list[int]:
    masks = [0] * NPAR
    for b in range(DATA_BITS):
        reg = (1 << (DATA_BITS - 1 - b)) << NPAR
        for i in range(CODE_BITS - 1, NPAR - 1, -1):
            if reg & (1 << i):
                reg ^= BCH_GEN << (i - NPAR)
        for p in range(NPAR):
            if (reg >> (NPAR - 1 - p)) & 1:
                masks[p] |= 1 << b
    return masks


PARITY_MASKS = _bch_parity_masks()


def _word(bits: np.ndarray, lo: int, hi: int) -> int:
    return int(bits[lo:hi].dot(1 << np.arange(hi - lo - 1, -1, -1, dtype=np.int64)))


def _pocsag_rics(bits: np.ndarray, windows: list[int]) -> dict[int, int]:
    found: dict[int, int] = {}
    for batch in framing.carve_fixed(bits, windows, BATCH_CODEWORDS * DATA_BITS):
        for d in batch.reshape(-1, DATA_BITS):
            if int(d[0]) != 0:
                continue
            if _word(d, 0, DATA_BITS) == IDLE_DATA:
                continue
            found[_word(d, 1, 19)] = _word(d, 19, 21)
    return found


pocsag_slice = asset_fixture("POCSAG/pocsag.cf32")


def _pocsag_spec() -> dict[str, object]:
    return {
        "symbol_rate": 1200.0,
        "path": [
            {
                "conv": "channelize",
                "decim": 4,
                "bandwidth_hz": 14000.0,
                "center_hz": -10250.0,
            },
            {"conv": "agc", "window_symbols": 64.0},
            {"conv": "fsk", "deviation": 4500.0},
            {"conv": "slice"},
            {"conv": "sync_word", "sync": SC_HEX, "max_errors": 0},
            {"conv": "permute", "perm": list(PERM)},
            {
                "conv": "block_code",
                "code_bits": CODE_BITS,
                "data_bits": DATA_BITS,
                "parity_masks": list(PARITY_MASKS),
                "correct_single": True,
                "emit": "data",
            },
        ],
    }


async def test_pocsag_offair_through_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pocsag_slice: Path
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    async with Client(build_server()) as client:
        validated = (
            await client.call_tool(
                "validate_modem",
                {
                    "spec": _pocsag_spec(),
                    "sample_rate": RATE,
                },
            )
        ).data
        assert validated["valid"] is True
        rx = (
            await client.call_tool(
                "run_rx",
                {
                    "spec": _pocsag_spec(),
                    "sample_rate": RATE,
                    "capture_path": str(pocsag_slice),
                },
            )
        ).data
        assert rx["status"] == "ok", rx
        assert rx["quality"]["verdict"] == "decoded"
        assert any(
            e["metric"] == "sync_matches" and e["assessment"] == "positive"
            for e in rx["quality"]["evidence"]
        )
        stream = rx["stream"]
        page = (
            await client.call_tool(
                "read_stream",
                {
                    "path": stream["path"],
                    "count": stream["items"],
                },
            )
        ).data
    assert page["count"] == page["total_items"]
    bits = np.frombuffer(page["bits"].encode(), np.uint8) - ord("0")
    found = _pocsag_rics(bits, rx["windows"])
    assert found == ORACLE, f"decoded {found}, oracle {ORACLE}"
