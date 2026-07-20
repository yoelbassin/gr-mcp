from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bits import _drm

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages.registry import stage_registry

_SLICE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "DRM"
    / "dw_modeb.cf32"
)


@pytest.mark.skipif(
    not _SLICE.exists(), reason="DRM slice absent — run tests/bits/make_drm_slice.py"
)
def test_ofdm_coherent_equalizes_to_clean_qpsk(tmp_path: Path) -> None:
    ensure_worker_warm()
    snk = tmp_path / "carr.cf32"
    modem = ModemSpec(
        name="drm_sync",
        symbol_rate=_drm.RATE / _drm.SYM_LEN,
        path=[ModemStep(conv="ofdm_coherent_sync", params=_drm.sync_params())],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=_drm.RATE,
        start=Descriptor(Level.IQ, "c"),
        source_io={"path": str(_SLICE)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=180.0).status == "ok"
    carr = np.fromfile(snk, np.complex64).reshape(-1, _drm.N_CARRIERS)
    fac = _drm.gather_fac_cells(carr)
    fac = fac / np.sqrt(np.mean(np.abs(fac) ** 2))
    corners = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    evm = np.mean(np.min(np.abs(fac.ravel()[:, None] - corners[None, :]), axis=1) ** 2)
    assert evm < 0.10, f"equalized FAC EVM {evm:.3f} (scratch reference 0.048)"
