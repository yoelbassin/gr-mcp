from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from marconi.mcp.tools import run_rx_tool, run_tx_tool, stream_stats

_FS, _RATE = 8000.0, 1000.0  # sps = 8, ample margin over the 2.0 min_input_sps floor
_N_BITS = 120_000

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

# seeds 0/1 are unremarkable QPSK; 23 is a MEASURED mid-capture Costas cycle
# slip on this long unwhitened random run (K=4 EVM balloons to ~0.58 there,
# since a fixed-K cluster fit assumes one rotation for the whole stream) --
# included deliberately because it is exactly the case where EVM stops being
# a safe discriminator while constant_modulus_ratio (a magnitude-only
# statistic, blind to phase/rotation) does not move.
_QPSK_SEEDS = (0, 1, 23)
_CMR_OK_MAX = 0.15

# wideband FSK deviations, all well past the h=0.5 MSK/near-PSK regime (a
# deviation=250 attempt -- true MSK -- nearly locks onto the QPSK costas
# loop, since MSK is OQPSK-adjacent; not a fair "obviously not PSK" fixture).
# EVM at these deviations is NOT a robust separator -- MEASURED evm_bad as
# low as 0.08 at deviation=1500, well under a genuine QPSK capture's own EVM
# on an unlucky seed (e.g. 0.58 at seed 23 above) -- but
# constant_modulus_ratio is: a false lock on a frequency-modulated signal
# never settles onto one ring, at any of these deviations.
_FM_DEVIATIONS = (750.0, 1500.0, 2500.0)
_FM_SEEDS = (1, 2)
_CMR_BAD_MIN = 0.3


def _bits(seed: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join("01"[b] for b in rng.integers(0, 2, _N_BITS))


def _fsk_tx(deviation: float) -> dict[str, Any]:
    return {
        "symbol_rate": _RATE,
        "path": [{"conv": "fsk", "deviation": deviation}, {"conv": "slice"}],
    }


def _tx(spec: dict[str, Any], tmp_path: Path, name: str, seed: int) -> str:
    tx = run_tx_tool(spec, _FS, bits=_bits(seed), out_path=str(tmp_path / name))
    assert tx["status"] == "ok", tx
    return cast(str, tx["iq_path"])


def _constant_modulus_ratio(
    spec: dict[str, Any], tmp_path: Path, name: str, seed: int
) -> float:
    iq = _tx(spec, tmp_path, name, seed)
    res = run_rx_tool(_QPSK_RX, _FS, capture_path=iq, timeout=60.0)
    assert res["status"] == "ok", res
    stream = cast(dict[str, object], res["stream"])
    assert stream["item_type"] == "c"
    s = stream_stats(cast(str, stream["path"]), item_type="c", clusters=4)
    cmr = s["constant_modulus_ratio"]
    assert isinstance(cmr, float)
    return cmr


def test_real_qpsk_is_tight_but_fm_through_psk_is_high_cmr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))

    cmr_ok = [
        _constant_modulus_ratio(_QPSK_TX, tmp_path, f"qpsk_{seed}.cf32", seed)
        for seed in _QPSK_SEEDS
    ]
    for cmr in cmr_ok:
        assert cmr < _CMR_OK_MAX

    cmr_bad = [
        _constant_modulus_ratio(_fsk_tx(dev), tmp_path, f"fm_{dev}_{seed}.cf32", seed)
        for dev in _FM_DEVIATIONS
        for seed in _FM_SEEDS
    ]
    for cmr in cmr_bad:
        assert cmr > _CMR_BAD_MIN

    # the discriminator dogfood #3 lacked: a real PSK lock keeps constant
    # modulus; a false lock on a frequency-modulated signal never settles
    # onto one ring -- true with a clear margin at every swept point, not
    # just on one hand-picked (seed, deviation) pair
    assert min(cmr_bad) > max(cmr_ok)
