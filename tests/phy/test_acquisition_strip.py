from pathlib import Path

import numpy as np

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

SYM_C = Descriptor(Level.SYMBOLS, "c", carrier=Carrier.SOFT)


def _qpsk_points() -> np.ndarray:
    from gnuradio import digital

    return np.asarray(digital.constellation_qpsk().points())


def _ramp(n: int) -> np.ndarray:
    r = np.ones(n, dtype=np.complex64)
    r[1::2] = -1.0
    return r


def _nearest(z: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.argmin(np.abs(z[:, None] - pts[None, :]), axis=1)


def _run_rx(pre: np.ndarray, stream: np.ndarray, tmp_path: Path) -> np.ndarray:
    ensure_worker_warm()
    src, snk = tmp_path / "in.cf32", tmp_path / "out.cf32"
    stream.astype(np.complex64).tofile(src)
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(
                conv="preamble_sync",
                params={
                    "preamble_i": pre.real.tolist(),
                    "preamble_q": pre.imag.tolist(),
                    "pad_symbols": 192,
                },
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=SYM_C,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    return np.fromfile(snk, np.complex64)


def test_strip_derotates_payload_to_the_constellation(tmp_path: Path) -> None:
    pts = _qpsk_points()
    rng = np.random.default_rng(0)
    pre = pts[rng.integers(0, 4, 64)]
    payload = pts[rng.integers(0, 4, 400)]
    stream = np.concatenate([_ramp(192), pre, payload]) * np.exp(1j * 0.7)
    out = _run_rx(pre, stream, tmp_path)
    assert len(payload) - 128 < len(out) <= len(payload)
    k = len(out)
    assert np.array_equal(_nearest(out, pts), _nearest(payload[:k], pts))


def test_strip_locks_first_preamble_not_stronger_replica(tmp_path: Path) -> None:
    # issue 02: a 2x-amplitude preamble replica inside the payload would steal a
    # global-argmax lock. corr_est_cc tags in stream order and sym_strip latches
    # the FIRST, so the true preamble wins and the payload is not truncated.
    pts = _qpsk_points()
    rng = np.random.default_rng(1)
    pre = pts[rng.integers(0, 4, 64)]
    payload = pts[rng.integers(0, 4, 400)].astype(np.complex128)
    payload[150:214] = 2.0 * pre
    stream = np.concatenate([_ramp(192), pre, payload]) * np.exp(-1j * 0.5)
    out = _run_rx(pre, stream, tmp_path)
    assert len(out) > len(payload) - 128
    assert np.array_equal(_nearest(out[:20], pts), _nearest(payload[:20], pts))
