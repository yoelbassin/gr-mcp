from __future__ import annotations

import os
from pathlib import Path

import pytest
from helpers import rds
from helpers.hardware import sdr_present

from marconi.capture import capture_iq
from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.io.bitfile import read_bits
from marconi.engine.io.source import SourceSlice
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream

IQ = Descriptor(Level.IQ, ItemType.C)
BITS = Descriptor(Level.BITS, ItemType.B)
_FM_HZ = os.environ.get("MARCONI_TEST_FM_HZ")
MIN_LIVE_BLOCKS = 16

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.xdist_group("sdr"),
    pytest.mark.skipif(not sdr_present(), reason="no SDR attached"),
    pytest.mark.skipif(
        not _FM_HZ,
        reason="MARCONI_TEST_FM_HZ unset — export a local RDS-carrying FM "
        "station frequency to run the live gate",
    ),
]


def test_fm_rds_live(tmp_path: Path) -> None:
    ensure_worker_warm()
    cap = capture_iq(
        tmp_path / "fm.cf32",
        center_hz=float(str(_FM_HZ)),
        sample_rate=250_000.0,
        duration_s=10.0,
    )
    assert cap.status == "ok", cap
    res = run_rx(
        rds.phy_modem(),
        stage_registry(),
        sample_rate=cap.sample_rate,
        start=IQ,
        workdir=tmp_path,
        source=SourceSlice(path=Path(cap.path)),
    )
    assert res.status == "ok", res
    assert res.bitstream is not None
    n_symbols = res.bitstream.num_bits
    assert n_symbols > 4000, f"only {n_symbols} biphase symbols demodulated"
    results = []
    for off in (0, 1):
        res2 = run_rx(
            rds.codec_modem(off),
            stage_registry(),
            sample_rate=1.0,
            start=BITS,
            workdir=tmp_path,
            input_stream=Bitstream(path=res.bitstream.path, num_bits=n_symbols),
        )
        assert res2.status == "ok", res2
        assert res2.bitstream is not None
        results.append(rds.decode_groups(read_bits(res2.bitstream.path)))
    n_valid, ps = max(results)
    assert n_valid >= MIN_LIVE_BLOCKS, f"{n_valid} checkword-valid blocks: {results}"
    assert ps.strip("_"), f"no PS characters recovered: {results}"
