from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry


def test_mslice_maps_four_levels_to_dibit_bits(tmp_path: Path) -> None:
    ensure_worker_warm()
    # four levels (low->high): -1.0, -0.33, +0.33, +1.0 ; DMR-style code [3,2,0,1]
    levels = np.array([-1.0, -0.33, 0.33, 1.0], np.float32)
    seq = np.tile(levels, 500)  # 2000 symbols
    src = tmp_path / "sym.f32"
    seq.tofile(src)
    snk = tmp_path / "bits.u8"
    modem = ModemSpec(
        symbol_rate=4800.0,
        path=[
            ModemStep(
                conv="mslice",
                params={
                    "thresholds": [-0.667, 0.0, 0.667],
                    "code": [3, 2, 0, 1],
                    "bits": 2,
                },
            ),
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=39062.0,
        start=Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=60.0).status == "ok"
    out = np.fromfile(snk, np.uint8)
    # code MSB-first: 3->[1,1] 2->[1,0] 0->[0,0] 1->[0,1]
    expect = np.tile(np.array([1, 1, 1, 0, 0, 0, 0, 1], np.uint8), 500)
    assert out.size == expect.size
    assert np.array_equal(out, expect)
