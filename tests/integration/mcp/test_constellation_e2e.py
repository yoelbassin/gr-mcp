from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from marconi.mcp.tools import run_rx_tool, run_tx_tool, stream_stats

_FS, _RATE = 8000.0, 1000.0  # sps = 8, ample margin over the 2.0 min_input_sps floor
_N_BITS = 120_000  # long enough that the symbol_sync/costas acquisition transient
# (a few thousand symbols) is a small fraction of the aggregate EVM

# full QPSK modulator (bits -> constellation points -> pulse-shaped IQ), mirrors
# tests/integration/engine/modulation/test_psk_roundtrip.py's `_full`
_QPSK_TX = {
    "symbol_rate": _RATE,
    "path": [{"conv": "psk_demod", "order": 4}, {"conv": "psk_demap", "order": 4}],
}
# bare demod-only path: lands on complex soft symbols (item_type "c", Task 4);
# psk_demod requires normalized amplitude, hence the leading agc
_QPSK_RX = {
    "symbol_rate": _RATE,
    "path": [{"conv": "agc"}, {"conv": "psk_demod", "order": 4}],
}
# full FSK modulator, mirrors test_fsk_roundtrip.py's `_modem`; deviation =
# 0.75x symbol_rate is well past the h=0.5 MSK/near-PSK regime, so this reads
# as genuinely frequency- rather than phase-modulated
_FSK_TX = {
    "symbol_rate": _RATE,
    "path": [{"conv": "fsk", "deviation": 750.0}, {"conv": "slice"}],
}


def _bits(seed: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join("01"[b] for b in rng.integers(0, 2, _N_BITS))


def _tx(spec: dict[str, Any], tmp_path: Path, name: str, seed: int) -> str:
    tx = run_tx_tool(spec, _FS, bits=_bits(seed), out_path=str(tmp_path / name))
    assert tx["status"] == "ok", tx
    return cast(str, tx["iq_path"])


def test_real_qpsk_is_tight_but_fm_through_psk_is_high_evm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))

    # (a) a real QPSK capture, demodulated by the matching path
    qpsk_iq = _tx(_QPSK_TX, tmp_path, "qpsk.cf32", seed=0)
    res_ok = run_rx_tool(_QPSK_RX, _FS, capture_path=qpsk_iq, timeout=60.0)
    assert res_ok["status"] == "ok", res_ok
    stream_ok = cast(dict[str, object], res_ok["stream"])
    assert stream_ok["item_type"] == "c"
    s_ok = stream_stats(cast(str, stream_ok["path"]), item_type="c", clusters=4)
    clusters_ok, evm_ok = s_ok["clusters"], s_ok["evm"]
    assert isinstance(clusters_ok, list) and len(clusters_ok) == 4
    assert isinstance(evm_ok, float) and evm_ok < 0.3

    # (b) a frequency-modulated capture forced through the QPSK demod: false lock
    fm_iq = _tx(_FSK_TX, tmp_path, "fm.cf32", seed=1)
    res_bad = run_rx_tool(_QPSK_RX, _FS, capture_path=fm_iq, timeout=60.0)
    assert res_bad["status"] == "ok", res_bad
    stream_bad = cast(dict[str, object], res_bad["stream"])
    assert stream_bad["item_type"] == "c"
    s_bad = stream_stats(cast(str, stream_bad["path"]), item_type="c", clusters=4)
    evm_bad = s_bad["evm"]
    assert isinstance(evm_bad, float)

    assert evm_bad > evm_ok  # the discriminator dogfood #3 lacked
    assert evm_bad > 0.4  # a clear false lock, not a near-tie
