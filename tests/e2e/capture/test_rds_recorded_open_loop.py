from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from marconi.mcp.tools import run_rx_tool

_ASSET = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "assets"
    / "RDS"
    / "fm_rds_250k_1Msamples.iq"
)


@pytest.mark.skipif(not _ASSET.exists(), reason="RDS asset absent")
def test_rds_dbpsk_via_open_loop_timing() -> None:
    # Real off-air RDS: a DBPSK biphase subcarrier at 57 kHz. Condition to the
    # subcarrier, then recover it with OPEN-LOOP Oerder-Meyr timing +
    # differential (which absorbs the residual subcarrier offset). A clean BPSK
    # constellation on real data proves the open-loop timing works off-air, not
    # only in sim. R2 = |mean(exp(j2*angle(z)))| collapses BPSK to one point.
    cond: list[dict[str, object]] = [
        {"conv": "fm_demod", "deviation": 75000.0},
        {"conv": "analytic"},
        {
            "conv": "channelize",
            "decim": 5,
            "bandwidth_hz": 4800.0,
            "center_hz": 57000.0,
        },
        {"conv": "resample", "interpolation": 19, "decimation": 50},
        {"conv": "agc", "mode": "power"},
    ]
    demod: list[dict[str, object]] = [
        {"conv": "symbol_sync", "sps": 8, "loop_bw": 0.0},
        {"conv": "sample_symbols"},
        {"conv": "differential_demod"},
    ]
    r: dict[str, Any] = run_rx_tool(
        {"symbol_rate": 2375.0, "path": cond + demod},
        250000.0,
        capture_path=str(_ASSET),
        timeout=180.0,
    )
    assert r["status"] == "ok", r
    st: Any = r["stream"]
    assert st and st["item_type"] == "c" and st["items"] > 3000, st
    z: np.ndarray[Any, np.dtype[np.complex64]] = np.fromfile(
        st["path"], dtype=np.complex64
    )
    z = z[np.abs(z) > np.median(np.abs(z))]
    r2: np.floating[Any] = abs(np.mean(np.exp(1j * 2 * np.angle(z))))
    assert r2 > 0.9, r2  # measured 0.977 open-loop vs 0.976 closed-loop reference
