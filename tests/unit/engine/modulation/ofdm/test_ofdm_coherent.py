from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from e2e.drm import _drm
from helpers import _lattice
from helpers._paths import ARTIFACTS

from marconi.engine.backends.base import find_diagnostic
from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.ofdm.stages import OfdmCoherentSyncStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

_SLICE = ARTIFACTS / "assets" / "DRM" / "dw_modeb.cf32"


@pytest.mark.skipif(
    not _SLICE.exists(), reason="DRM slice absent — run tests/e2e/make_drm_slice.py"
)
def test_ofdm_coherent_equalizes_to_clean_qpsk(tmp_path: Path) -> None:
    ensure_worker_warm()
    snk = tmp_path / "carr.cf32"
    modem = Modem(
        name="drm_sync",
        symbol_rate=_drm.RATE / _drm.SYM_LEN,
        path=[OfdmCoherentSyncStep(**_drm.sync_params())],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=_drm.RATE,
        start=Descriptor(Level.IQ, ItemType.C),
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


def test_synthetic_lattice_full_chain_equalizes(tmp_path: Path) -> None:
    ensure_worker_warm()
    iq = _lattice.make_iq(
        200, cfo_hz=9.0, snr_db=28.0, lead_noise=1200, sto_frac=0.4, seed=11
    )
    src = tmp_path / "lat.cf32"
    iq.tofile(src)
    snk = tmp_path / "eq.cf32"
    modem = Modem(
        name="lattice_sync",
        symbol_rate=_lattice.RATE / _lattice.SYM_LEN,
        path=[OfdmCoherentSyncStep(**_lattice.sync_params())],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=_lattice.RATE,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    result = GnuRadioBackend().run_pipeline(pipe, timeout=120.0)
    assert result.status == "ok"
    ncell = len(_lattice.EMIT)
    eq = np.fromfile(snk, np.complex64).reshape(-1, ncell)
    assert len(eq) >= 40 * _lattice.NS
    mask = _lattice.data_mask()
    cells = np.concatenate([eq[i][mask[i % _lattice.NS]] for i in range(len(eq))])
    cells = cells / np.sqrt(np.mean(np.abs(cells) ** 2))
    corners = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    evm = np.mean(np.min(np.abs(cells[:, None] - corners[None, :]), axis=1) ** 2)
    assert evm < 0.10, f"synthetic full-chain EVM {evm:.3f}"
    frames_row = next(
        (d for d in result.diagnostics if d.key == "frames_emitted"), None
    )
    assert frames_row is not None
    locks_row = find_diagnostic(result.diagnostics, frames_row.block, "locks")
    assert (
        locks_row is not None and locks_row.count is not None and locks_row.count >= 1
    )


def test_synthetic_dropout_relocks_through_real_chain(tmp_path: Path) -> None:
    ensure_worker_warm()
    iq = np.concatenate(
        [
            _lattice.make_iq(120, snr_db=28.0, seed=21),
            np.zeros(4000, np.complex64),
            _lattice.make_iq(120, snr_db=28.0, seed=22),
        ]
    )
    src = tmp_path / "lat_dropout.cf32"
    iq.tofile(src)
    snk = tmp_path / "eq_dropout.cf32"
    modem = Modem(
        name="lattice_sync",
        symbol_rate=_lattice.RATE / _lattice.SYM_LEN,
        path=[OfdmCoherentSyncStep(**_lattice.sync_params())],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=_lattice.RATE,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    result = GnuRadioBackend().run_pipeline(pipe, timeout=120.0)
    assert result.status == "ok"
    ncell = len(_lattice.EMIT)
    eq = np.fromfile(snk, np.complex64).reshape(-1, ncell)
    finite = np.isfinite(eq).all(axis=1)
    # grace-window dead-air frames are non-finite by construction here
    # (exact-zero gap -> 0/0); off-air they are finite garbage — CRC-gated
    assert int((~finite).sum()) <= 12 * _lattice.NS
    assert int(finite.sum()) >= 30 * _lattice.NS
    mask = _lattice.data_mask()
    cells = np.concatenate(
        [eq[i][mask[i % _lattice.NS]] for i in np.flatnonzero(finite)]
    )
    cells = cells / np.sqrt(np.mean(np.abs(cells) ** 2))
    corners = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    evm = np.mean(np.min(np.abs(cells[:, None] - corners[None, :]), axis=1) ** 2)
    assert evm < 0.10, f"synthetic dropout-relock EVM {evm:.3f}"
    eq_block = next(
        (d.block for d in result.diagnostics if d.key == "frames_emitted"), None
    )
    assert eq_block is not None
    relocks_row = find_diagnostic(result.diagnostics, eq_block, "relocks")
    assert (
        relocks_row is not None
        and relocks_row.count is not None
        and relocks_row.count >= 1
    )
    frames_emitted_blocks = {
        d.block for d in result.diagnostics if d.key == "frames_emitted"
    }
    trk_row = next(
        (
            d
            for d in result.diagnostics
            if d.key == "locks" and d.block not in frames_emitted_blocks
        ),
        None,
    )
    assert trk_row is not None and trk_row.count is not None and trk_row.count >= 2
