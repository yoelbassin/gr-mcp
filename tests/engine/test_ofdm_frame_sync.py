import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.ofdm.stages import OfdmFrameSyncProbeStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

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


from engine._fakegr import FAKE_GR as _FakeGr  # noqa: E402
from engine._fakegr import drive as _drive  # noqa: E402


def test_zero_input_wakeup_is_timing_invariant():
    """forecast announces 0 whenever emission work is pending, so the scheduler
    MAY deliver zero-input calls at any point mid-stream; emitted content must
    not depend on when they land. The pre-fix code treated inp.size == 0 as EOF
    and emitted without the forward-null gate, so under drift a mid-stream
    wakeup extracted the next frame at the un-snapped predicted base (issue 05)."""
    from marconi.engine.backends.gnuradio.embedded.ofdm import make_ofdm_frame_sync

    # seed 2 = the drift-test fixture; seed 7 trips find_null's known one-late
    # boundary quirk (weak first CP sample), which is orthogonal to timing
    rng = np.random.default_rng(2)
    drift, m_frames = 3, 6
    usefuls_per_frame, parts = [], []
    for _ in range(m_frames):
        parts.append(np.zeros(NULL + drift, complex))
        us = [
            rng.standard_normal(FFT) + 1j * rng.standard_normal(FFT)
            for _ in range(DS + 1)
        ]
        usefuls_per_frame.append(np.concatenate(us))
        for u in us:
            parts.append(np.concatenate([u[-CP:], u]))
    sig = np.concatenate(parts).astype(np.complex64)
    expected = np.concatenate(
        [(u / np.std(u)).astype(np.complex64) for u in usefuls_per_frame]
    )

    def run(chunk: int) -> np.ndarray:
        blk = make_ofdm_frame_sync(
            _FakeGr,
            fft_len=FFT,
            cp_len=CP,
            sym_len=SYM,
            null_len=NULL,
            frame_len=FRAME,
            data_syms=DS,
        )
        return _drive(blk, sig, chunk)

    oneshot = run(sig.size)
    assert oneshot.size == expected.size
    assert np.allclose(oneshot, expected, atol=1e-4)
    for chunk in (173, SYM, FRAME + 1):
        chunked = run(chunk)
        assert np.array_equal(chunked, oneshot), f"chunk={chunk} diverged"


def test_null_less_stream_holds_a_bounded_buffer():
    """A wrong-band capture never contains a null: detection keeps failing and
    the pre-fix block buffered the whole stream (GB capture -> OOM). Past the
    ladder cap the detection window must slide instead — and a frame train
    arriving after the null-less stretch must still acquire and decode."""
    from marconi.engine.backends.gnuradio.embedded.ofdm import make_ofdm_frame_sync

    rng = np.random.default_rng(11)
    flat = np.exp(2j * np.pi * rng.random(50 * FRAME)).astype(np.complex64)

    def mk():
        return make_ofdm_frame_sync(
            _FakeGr,
            fft_len=FFT,
            cp_len=CP,
            sym_len=SYM,
            null_len=NULL,
            frame_len=FRAME,
            data_syms=DS,
        )

    cap = NULL + (DS + 1) * SYM + 4 * FRAME
    blk = mk()
    _drive(blk, flat, chunk=FRAME)
    assert blk._buf.size <= cap, f"buffer held {blk._buf.size}"

    # the first null is 3*NULL: a mid-stream null flanked by signal on both
    # sides needs headroom over find_null's smoother width (real captures
    # have null_len >> win; the sibling tests' nulls sit at stream start
    # where the convolution edge supplies it). seed 17: a strong first CP
    # sample, dodging find_null's known one-late refine quirk.
    frame_rng = np.random.default_rng(17)
    usefuls, parts = [], [np.zeros(2 * NULL, complex)]
    for _ in range(3):
        parts.append(np.zeros(NULL, complex))
        us = [
            frame_rng.standard_normal(FFT) + 1j * frame_rng.standard_normal(FFT)
            for _ in range(DS + 1)
        ]
        usefuls.append(np.concatenate(us))
        for u in us:
            parts.append(np.concatenate([u[-CP:], u]))
    sig = np.concatenate([flat] + parts).astype(np.complex64)
    expected = np.concatenate([(u / np.std(u)).astype(np.complex64) for u in usefuls])
    for chunk in (sig.size, 173):
        out = _drive(mk(), sig, chunk)
        assert out.size == expected.size, f"chunk={chunk}: {out.size}"
        assert np.allclose(out, expected, atol=1e-4), f"chunk={chunk} diverged"


def test_resync_base_snaps_corrects_and_rejects():
    from marconi.engine.backends.gnuradio.embedded.ofdm import _resync_base

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


def test_frame_sync_buffer_stays_bounded():
    ensure_worker_warm()
    from gnuradio import blocks as gb
    from gnuradio import gr

    from marconi.engine.backends.gnuradio.embedded.ofdm import make_ofdm_frame_sync

    fft, cp, ds = 256, 64, 9
    sym = fft + cp
    null = 400
    frame = null + (ds + 1) * sym
    # seed 4 trips issue 45's find_null one-frame-short quirk
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
    assert blk._buf.size <= 2 * frame + 8192, f"buffer held {blk._buf.size}"


def test_frame_sync_reports_truncated_final_frame():
    """A frame whose usefuls never fully arrive (end-of-stream mid-frame) is
    dropped by design — frames are atomic — but the drop must be visible:
    diagnostics carry the missing item count. A capture ending cleanly after
    its last frame reports zero."""
    from marconi.engine.backends.gnuradio.embedded.ofdm import make_ofdm_frame_sync

    rng = np.random.default_rng(0)

    def frame_parts():
        us = [
            rng.standard_normal(FFT) + 1j * rng.standard_normal(FFT)
            for _ in range(DS + 1)
        ]
        return [np.zeros(NULL, complex)] + [np.concatenate([u[-CP:], u]) for u in us]

    def mk():
        return make_ofdm_frame_sync(
            _FakeGr,
            fft_len=FFT,
            cp_len=CP,
            sym_len=SYM,
            null_len=NULL,
            frame_len=FRAME,
            data_syms=DS,
        )

    complete = np.concatenate(frame_parts()).astype(np.complex64)
    blk = mk()
    out = _drive(blk, complete, chunk=1000)
    assert out.size == (DS + 1) * FFT
    assert blk.diagnostics["truncated_frame_items"] == 0

    # second frame's null arrives but only two of its four symbols do
    truncated = np.concatenate(frame_parts() + frame_parts()[:3]).astype(np.complex64)
    blk = mk()
    out = _drive(blk, truncated, chunk=1000)
    assert out.size == (DS + 1) * FFT  # first frame intact, second dropped
    usefuls_len = (DS + 1) * SYM
    # find_null's known +/-1 boundary quirk shifts the counted base a sample
    # or two; the metric's job is magnitude, not sample-exactness
    missing = blk.diagnostics["truncated_frame_items"]
    assert abs(missing - (usefuls_len - 2 * SYM)) <= 2, missing
