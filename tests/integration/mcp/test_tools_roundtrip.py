from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marconi.mcp.tools import read_stream, run_rx_tool, run_tx_tool

_TXRX = {
    "symbol_rate": 1.0,
    "path": [{"conv": "fsk", "deviation": 1.0}, {"conv": "slice"}],
}


def test_tx_rx_roundtrip_ber0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    rng = np.random.default_rng(11)
    bits = "".join("01"[b] for b in rng.integers(0, 2, 2048))
    tx = run_tx_tool(_TXRX, sample_rate=4.0, bits=bits)
    assert tx["status"] == "ok", tx
    assert cast(int, tx["num_samples"]) > 0
    rx = run_rx_tool(_TXRX, sample_rate=4.0, capture_path=cast(str, tx["iq_path"]))
    assert rx["status"] == "ok", rx
    stream = cast(dict[str, object], rx["stream"])
    assert stream["item_type"] == "b"
    page = read_stream(cast(str, stream["path"]), offset=0, count=4096)
    decoded = cast(str, page["bits"])
    assert bits in decoded or decoded in bits or _aligned_equal(bits, decoded)


def _aligned_equal(tx_bits: str, rx_bits: str, max_shift: int = 64) -> bool:
    tx_arr = np.frombuffer(tx_bits.encode(), np.uint8)
    rx_arr = np.frombuffer(rx_bits.encode(), np.uint8)
    n = min(tx_arr.size, rx_arr.size) - max_shift
    if n <= 0:
        return False
    return any(
        np.array_equal(rx_arr[s : s + n], tx_arr[:n])
        or np.array_equal(rx_arr[:n], tx_arr[s : s + n])
        for s in range(max_shift)
    )


def test_run_rx_rejects_both_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    from fastmcp.exceptions import ToolError

    from marconi.mcp.tools import TOOLS

    with pytest.raises(ToolError, match=r"\[invalid_argument\]"):
        TOOLS["run_rx"](
            _TXRX, sample_rate=4.0, capture_path="a.cf32", input_path="b.u8"
        )
