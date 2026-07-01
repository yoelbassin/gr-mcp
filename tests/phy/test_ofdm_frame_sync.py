import numpy as np

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

FFT, CP, SYM, NULL, DS = 16, 4, 20, 24, 3
FRAME = NULL + 4 * SYM


def test_frame_sync_strips_cp(tmp_path):
    ensure_worker_warm()
    rng = np.random.default_rng(0)
    usefuls = [
        rng.standard_normal(FFT) + 1j * rng.standard_normal(FFT) for _ in range(4)
    ]
    parts = [np.zeros(NULL, complex)]
    for u in usefuls:
        parts.append(np.concatenate([u[-CP:], u]))
    sig = np.concatenate(parts).astype(np.complex64)
    src = tmp_path / "i.cf32"
    sig.tofile(src)
    snk = tmp_path / "o.cf32"
    modem = ModemSpec(
        name="fs",
        symbol_rate=float(2_048_000 / SYM),
        path=[
            ModemStep(
                conv="ofdm_frame_sync_probe",
                params={
                    "fft_len": FFT,
                    "cp_len": CP,
                    "sym_len": SYM,
                    "null_len": NULL,
                    "frame_len": FRAME,
                    "data_syms": DS,
                },
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=2_048_000.0,
        start=Descriptor(Level.IQ, "c"),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    out = np.fromfile(snk, np.complex64)
    assert out.size == (DS + 1) * FFT
    # ofdm_frame_sync normalizes each frame's usefuls by their std (so the stock
    # soft decoder downstream sees unit-ish magnitude); compare against that.
    expected = np.concatenate(usefuls)
    expected = (expected / np.std(expected)).astype(np.complex64)
    assert np.allclose(out, expected, atol=1e-4)


def test_frame_sync_resyncs_under_drift(tmp_path):
    # Actual on-air period is FRAME + DRIFT (SFO): a constant stride drifts off
    # the useful part after the first frame; per-frame null resync must track it.
    ensure_worker_warm()
    rng = np.random.default_rng(2)
    drift, m_frames = 3, 6
    usefuls_per_frame = []
    parts = []
    for _ in range(m_frames):
        parts.append(np.zeros(NULL + drift, complex))
        usefuls = [
            rng.standard_normal(FFT) + 1j * rng.standard_normal(FFT)
            for _ in range(DS + 1)
        ]
        usefuls_per_frame.append(usefuls)
        for u in usefuls:
            parts.append(np.concatenate([u[-CP:], u]))
    sig = np.concatenate(parts).astype(np.complex64)
    src = tmp_path / "i.cf32"
    sig.tofile(src)
    snk = tmp_path / "o.cf32"
    modem = ModemSpec(
        name="fs",
        symbol_rate=float(2_048_000 / SYM),
        path=[
            ModemStep(
                conv="ofdm_frame_sync_probe",
                params={
                    "fft_len": FFT,
                    "cp_len": CP,
                    "sym_len": SYM,
                    "null_len": NULL,
                    "frame_len": FRAME,
                    "data_syms": DS,
                },
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=2_048_000.0,
        start=Descriptor(Level.IQ, "c"),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    out = np.fromfile(snk, np.complex64)
    expected = np.concatenate(
        [
            (np.concatenate(u) / np.std(np.concatenate(u))).astype(np.complex64)
            for u in usefuls_per_frame
        ]
    )
    assert out.size == expected.size
    assert np.allclose(out, expected, atol=1e-4)


def test_resync_base_snaps_corrects_and_rejects():
    from marconi.phy.backends.gnuradio.embedded.ofdm import _resync_base

    null_len, tol, max_corr = 24, 2, 20
    rng = np.random.default_rng(3)
    z = rng.standard_normal(400) + 1j * rng.standard_normal(400)
    buf = z.astype(np.complex64)
    buf[107:134] = 0.0  # a null gap; true next base = 134
    r = dict(null_len=null_len, tol=tol, max_corr=max_corr)
    assert _resync_base(buf, 131, **r) == 134  # genuine drift -> follow the null
    assert _resync_base(buf, 133, **r) == 133  # within tol -> snap to prediction
    assert _resync_base(buf, 200, **r) == 200  # no null in window -> keep prediction
    assert _resync_base(buf, 395, **r) == 395  # insufficient buffer -> keep prediction
