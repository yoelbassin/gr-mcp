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
