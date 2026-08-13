from pathlib import Path

import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.ofdm.stages import OfdmFrameSyncProbeStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

FFT, CP, SYM, NULL, DS = 16, 4, 20, 24, 3
FRAME = NULL + 4 * SYM


def test_frame_sync_strips_cp(tmp_path: Path) -> None:
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
    modem = Modem(
        name="fs",
        symbol_rate=float(2_048_000 / SYM),
        path=[
            OfdmFrameSyncProbeStep(
                fft_len=FFT,
                cp_len=CP,
                sym_len=SYM,
                null_len=NULL,
                frame_len=FRAME,
                data_syms=DS,
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        sample_rate=2_048_000.0,
        start=Descriptor(Level.IQ, ItemType.C),
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


def test_frame_sync_resyncs_under_drift(tmp_path: Path) -> None:
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
    modem = Modem(
        name="fs",
        symbol_rate=float(2_048_000 / SYM),
        path=[
            OfdmFrameSyncProbeStep(
                fft_len=FFT,
                cp_len=CP,
                sym_len=SYM,
                null_len=NULL,
                frame_len=FRAME,
                data_syms=DS,
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        sample_rate=2_048_000.0,
        start=Descriptor(Level.IQ, ItemType.C),
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


def test_frame_sync_buffer_stays_bounded() -> None:
    ensure_worker_warm()
    from gnuradio import blocks as gb
    from gnuradio import gr

    from marconi.engine.backends.gnuradio.embedded.ofdm import make_ofdm_frame_sync

    fft, cp, ds = 256, 64, 9
    sym = fft + cp
    null = 400
    frame = null + (ds + 1) * sym
    # seed 4 trips find_null's known one-frame-short quirk
    rng = np.random.default_rng(5)
    parts, usefuls_all = [], []
    for _ in range(40):
        parts.append(np.zeros(null, complex))
        us = [
            rng.standard_normal(fft) + 1j * rng.standard_normal(fft)
            for _ in range(ds + 1)
        ]
        usefuls_all.append(np.concatenate(us))
        for u in us:
            parts.append(np.concatenate([u[-cp:], u]))
    sig = np.concatenate(parts).astype(np.complex64)
    blk = make_ofdm_frame_sync(
        gr,
        fft_len=fft,
        cp_len=cp,
        sym_len=sym,
        null_len=null,
        frame_len=frame,
        data_syms=ds,
    )
    tb = gr.top_block()
    src = gb.vector_source_c(sig.tolist(), False)
    snk = gb.vector_sink_c()
    tb.connect(src, blk, snk)
    tb.run()
    out = np.array(snk.data(), np.complex64)
    expected = np.concatenate(
        [(u / np.std(u)).astype(np.complex64) for u in usefuls_all]
    )
    assert out.size == expected.size
    assert np.allclose(out, expected, atol=1e-4)
    assert blk._core._buf.size <= 2 * frame + 8192, f"buffer held {blk._core._buf.size}"
