"""Real off-air DRM (Deutsche Welle, Mode B, spectrum occupancy 3), one
Modem spanning phy through the coding tail, CRC-8 as the FAC oracle plus
the FAC channel-parameter invariants. Known-good on this slice: 109/109 CRC-8
-valid FAC blocks, every one occupancy 0011, identities cycling 01/10/11
across the super-frame. The gate is 109 minus margin — enough for
cross-machine float drift, far above any partial-decode regression.

The PHY (ofdm_coherent_sync/cell_select/soft_demap/deinterleave/depuncture/fec)
and the energy-dispersal descramble + frame segmentation compose in a single
Modem (_drm.fac_phy_steps/_drm.sdc_phy_steps) — descramble/segment are
product coding stages, the same GR-chain-then-coding-tail composition as the
DAB gate. CRC checking and field parsing are test-side helpers
(_drm.fac_check/_drm.sdc_check/_drm.parse_fac/_drm.parse_sdc_label) over
run_rx's per-window carve (helpers.framing.carve_fixed).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from e2e import _drm
from helpers import framing

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.bitfile import read_bits
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, "c")
_SLICE = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "assets"
    / "DRM"
    / "dw_modeb.cf32"
)


@pytest.mark.skipif(
    not _SLICE.exists(), reason="DRM slice absent — run tests/e2e/make_drm_slice.py"
)
def test_drm_fac(tmp_path: Path) -> None:
    ensure_worker_warm()
    modem = Modem(
        name="drm_fac",
        symbol_rate=_drm.RATE / _drm.SYM_LEN,
        path=_drm.fac_phy_steps(),
    )
    res = run_rx(
        modem,
        stage_registry(),
        sample_rate=_drm.RATE,
        start=IQ,
        workdir=tmp_path,
        source_io={"path": str(_SLICE)},
    )
    assert res.status == "ok", res
    assert res.windows, "no FAC blocks segmented"
    assert res.bitstream is not None
    bits = read_bits(res.bitstream.path)

    bodies = [
        body
        for window in framing.carve_fixed(bits, res.windows, _drm.FAC_FRAME_BITS)
        if (body := _drm.fac_check(window)) is not None
    ]
    num_ok = len(bodies)
    assert (
        num_ok >= 90
    ), f"expected >=90 CRC-8-valid FAC blocks (scratch 109/109), got {num_ok}"
    fields = [_drm.parse_fac(body) for body in bodies]
    assert all(f["occupancy"] == "0011" for f in fields)  # occ 3 / 10 kHz
    ids = {str(f["identity"]) for f in fields}
    assert {"01", "10", "11"} <= ids  # super-frame identity cycle


@pytest.mark.skipif(
    not _SLICE.exists(), reason="DRM slice absent — run tests/e2e/make_drm_slice.py"
)
def test_drm_sdc(tmp_path: Path) -> None:
    # SDC lives in symbols 0-1 of the SUPER-frame; the coherent sync emits only
    # FRAME-aligned symbols, so which of the 3 frame positions is super-frame-frame-0
    # is unknown. Sweep all 3 phases and keep the one that yields CRC-16-valid SDC —
    # the honest, generic form of the scratch's brute-force (winner: phase 2).
    ensure_worker_warm()
    best_phase = 0
    best_bodies: list[bytes] = []
    for phase in range(3):
        workdir = tmp_path / f"phase{phase}"
        workdir.mkdir()
        modem = Modem(
            name="drm_sdc",
            symbol_rate=_drm.RATE / _drm.SYM_LEN,
            path=_drm.sdc_phy_steps(phase),
        )
        res = run_rx(
            modem,
            stage_registry(),
            sample_rate=_drm.RATE,
            start=IQ,
            workdir=workdir,
            source_io={"path": str(_SLICE)},
        )
        assert res.status == "ok", res
        assert res.bitstream is not None
        bits = read_bits(res.bitstream.path)
        bodies = [
            body
            for window in framing.carve_fixed(bits, res.windows, _drm.SDC_FRAME_BITS)
            if (body := _drm.sdc_check(window)) is not None
        ]
        if len(bodies) > len(best_bodies):
            best_phase, best_bodies = phase, bodies

    labels = {_drm.parse_sdc_label(body) for body in best_bodies}
    assert len(best_bodies) >= 30, (
        f"expected >=30 CRC-16-valid SDC super-frames (scratch 36/36), got "
        f"{len(best_bodies)} at winning phase {best_phase}"
    )
    assert "DW DRM" in labels, (best_phase, labels)
