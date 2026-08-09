from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry, step_models
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

FS, RS, SPS = 8000.0, 1000.0, 8


def _rrc(sps: int, span: int = 11, beta: float = 0.35) -> np.ndarray:
    n = span * sps
    t = (np.arange(n + 1) - n / 2) / sps
    num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(np.pi * t * (1 + beta))
    den = np.pi * t * (1 - (4 * beta * t) ** 2)
    h = np.where(np.abs(t) < 1e-8, 1 - beta + 4 * beta / np.pi, num / den)
    return (h / np.sqrt(np.sum(h**2))).astype(float)


def _bursty_dqpsk(
    nb: int, L: int, G: int, sto: float, sfo_ppm: float, snr: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = _rrc(SPS)
    seq: list = []
    for _ in range(nb):
        k = rng.integers(0, 4, L)
        seq.extend(np.exp(1j * np.cumsum(k) * (np.pi / 2)).tolist())
        seq.extend([0j] * G)
    up = np.zeros(len(seq) * SPS, complex)
    up[::SPS] = np.asarray(seq)
    x = np.convolve(up, H, "same")
    # SFO: resample by (1+ppm); STO: fractional sub-sample shift
    n = np.arange(x.size)
    x = np.interp(n * (1 + sfo_ppm * 1e-6) + sto, n, x.real) + 1j * np.interp(
        n * (1 + sfo_ppm * 1e-6) + sto, n, x.imag
    )
    pw = np.mean(np.abs(x[np.abs(x) > 1e-6]) ** 2)
    x = x + np.sqrt(pw / 10 ** (snr / 10) / 2) * (
        rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)
    )
    return x.astype(np.complex64)


def _r4(z: np.ndarray) -> float:
    z = z[np.abs(z) > np.median(np.abs(z))]
    return 0.0 if z.size < 8 else float(abs(np.mean(np.exp(1j * 4 * np.angle(z)))))


def _decode(iq: np.ndarray, tmp: Path) -> np.ndarray:
    iqn = (iq / np.sqrt(np.mean(np.abs(iq) ** 2))).astype(np.complex64)
    (tmp / "iq.cf32").write_bytes(iqn.tobytes())
    path = [
        {"conv": "symbol_sync", "sps": SPS, "loop_bw": 0.0},
        {"conv": "sample_symbols"},
        {"conv": "differential_demod"},
    ]
    modem = Modem.from_spec({"symbol_rate": RS, "path": path}, step_models())
    start = Descriptor(Level.IQ, ItemType.C, Carrier.HARD, Amplitude.RMS_UNITY)
    res = run_rx(
        modem,
        stage_registry(),
        sample_rate=FS,
        start=start,
        workdir=tmp,
        source_io={"path": str(tmp / "iq.cf32")},
        timeout=120.0,
    )
    assert res.symbolstream is not None, res
    return np.fromfile(res.symbolstream.path, dtype=np.complex64)


def test_open_loop_recovers_short_bursts(tmp_path: Path) -> None:
    iq = _bursty_dqpsk(60, 80, 120, sto=0.37, sfo_ppm=25.0, snr=25, seed=11)
    z = _decode(iq, tmp_path)
    assert _r4(z) > 0.9, _r4(z)  # spike: Gardner ~0.68 here; oracle ~0.998


def test_open_loop_recovers_continuous(tmp_path: Path) -> None:
    # a long continuous signal (no gaps) must decode via the block's max_region
    # cap-flush even though the engine's ratio-1 chain gives no finality probe.
    # z is already the differential_demod output; R4 measures its constellation
    # cleanliness directly (do NOT differentiate it again).
    iq = _bursty_dqpsk(1, 2000, 0, sto=0.29, sfo_ppm=10.0, snr=25, seed=12)
    z = _decode(iq, tmp_path)
    assert z.size > 1000, z.size  # not withheld-forever (regressed to 0 once)
    assert _r4(z) > 0.95, _r4(z)  # matches the perfect-timing oracle (~0.996)
