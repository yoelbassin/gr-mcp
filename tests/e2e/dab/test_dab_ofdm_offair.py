# tests/e2e/dab/test_dab_ofdm_offair.py
from pathlib import Path

import numpy as np
import pytest

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.ofdm.primitives import qpsk_lock
from marconi.engine.modulation.ofdm.stages import OfdmDemodStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

_SLICE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "DAB"
    / "bbc_slice.cf32"
)
_NC, _DS, _FFT = 1536, 3, 2048


def _carrier_bins():
    tmp = [0]
    for _ in range(1, _FFT):
        tmp.append((13 * tmp[-1] + 511) % _FFT)
    off = [x - 1024 for x in tmp if x != 1024 and 256 <= x <= 1792]
    return [d if d > 0 else _FFT + d for d in off]


def _bin_perm():
    bins = _carrier_bins()
    return bins + [b for b in range(_FFT) if b not in set(bins)]


@pytest.mark.skipif(
    not _SLICE.exists(), reason="DAB slice absent — run tests/e2e/dab/make_dab_slice.py"
)
def test_dab_carriers_lock(tmp_path):
    ensure_worker_warm()
    snk = tmp_path / "car.cf32"
    modem = Modem(
        name="dab",
        symbol_rate=float(2_048_000 / 2552),
        path=[
            OfdmDemodStep(
                fft_len=_FFT,
                cp_len=504,
                sym_len=2552,
                null_len=2656,
                frame_len=196608,
                n_frame_syms=76,
                data_syms=_DS,
                n_carriers=_NC,
                bin_perm=_bin_perm(),
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=2_048_000.0,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": str(_SLICE)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=120.0)
    assert r.status == "ok", r
    car = np.fromfile(snk, np.complex64)
    assert car.size >= (_DS + 1) * _NC
    frame0 = car[: (_DS + 1) * _NC].reshape(
        _DS + 1, _NC
    )  # symbol-major (symbol, carrier)
    diff = (frame0[1:] * np.conj(frame0[:-1])).reshape(-1)  # per-carrier DQPSK
    assert qpsk_lock(diff) > 0.8
