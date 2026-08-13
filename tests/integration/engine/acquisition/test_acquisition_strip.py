from pathlib import Path

import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.stages.acquisition import PreambleSyncStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

SYM_C = Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT)


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
    modem = Modem(
        symbol_rate=1.0,
        path=[
            PreambleSyncStep(
                preamble_i=pre.real.tolist(),
                preamble_q=pre.imag.tolist(),
                pad_symbols=192,
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
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


def test_strip_first_lock_intact_replica_rearms(tmp_path: Path) -> None:
    # corr_est_cc tags in stream order, so a 2x-amplitude preamble replica
    # inside the payload cannot retroactively steal the first lock (the old
    # global-argmax failure): the payload head decodes under burst-1's phase.
    # Every detection re-arms — the replica reads as a new
    # burst: its own span is stripped and the continuation is re-derotated
    # (an exact 64-symbol preamble inside a payload IS a burst boundary).
    pts = _qpsk_points()
    rng = np.random.default_rng(1)
    pre = pts[rng.integers(0, 4, 64)]
    payload = pts[rng.integers(0, 4, 400)].astype(np.complex128)
    payload[150:214] = 2.0 * pre
    stream = np.concatenate([_ramp(192), pre, payload]) * np.exp(-1j * 0.5)
    out = _run_rx(pre, stream, tmp_path)
    assert np.array_equal(_nearest(out[:145], pts), _nearest(payload[:145], pts))
    # the post-replica payload re-emerges (re-armed, same carrier phase)
    tail = _nearest(out[150:230], pts)
    ref = _nearest(payload, pts)
    hits = [
        s
        for s in range(205, 226)
        if np.array_equal(tail[: len(ref) - s][:60], ref[s : s + 60])
    ]
    assert hits, "post-replica payload not re-acquired"
