from pathlib import Path

import numpy as np
import numpy.typing as npt

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.ofdm.stages import OfdmDemodStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

FFT, CP, SYM, NULL, NC = 16, 4, 20, 24, 4
ACTIVE = [1, 2, 14, 15]
BIN_PERM = ACTIVE + [b for b in range(FFT) if b not in ACTIVE]
FRAME = NULL + 4 * SYM


def _synth() -> tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex128]]:
    rng = np.random.default_rng(7)
    out = [np.zeros(NULL, np.complex64)]
    cells = []
    for _ in range(4):
        c = (rng.integers(0, 2, NC) * 2 - 1) + 1j * (rng.integers(0, 2, NC) * 2 - 1)
        cells.append(c.astype(complex))
        spec = np.zeros(FFT, complex)
        spec[ACTIVE] = c
        u = np.fft.ifft(spec)
        out.append(np.concatenate([u[-CP:], u]).astype(np.complex64))
    return np.concatenate(out), np.stack(cells)  # cells: (symbol, carrier)


def test_ofdm_demod_symbol_major(tmp_path: Path) -> None:
    ensure_worker_warm()
    sig, cells = _synth()
    src = tmp_path / "in.cf32"
    sig.astype(np.complex64).tofile(src)
    snk = tmp_path / "car.cf32"
    modem = Modem(
        name="od",
        symbol_rate=float(2_048_000 / SYM),
        path=[
            OfdmDemodStep(
                fft_len=FFT,
                cp_len=CP,
                sym_len=SYM,
                null_len=NULL,
                frame_len=FRAME,
                n_frame_syms=4,
                data_syms=3,
                n_carriers=NC,
                bin_perm=BIN_PERM,
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=2_048_000.0,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=60.0)
    assert r.status == "ok", r
    out = np.fromfile(snk, np.complex64)
    assert out.size == 4 * NC
    # ofdm_frame_sync normalizes each frame by std, so the recovered carriers equal
    # the planted cells up to one global (real) scale; compare up to that scale.
    carriers = out.reshape(4, NC)  # symbol-major (symbol, carrier)
    scale = np.vdot(cells, carriers) / np.vdot(cells, cells)
    assert np.allclose(carriers, cells * scale, rtol=1e-3, atol=1e-4)
