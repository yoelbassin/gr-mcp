from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry


def test_dc_block_attenuates_dc_offset(tmp_path: Path) -> None:
    ensure_worker_warm()
    n = 200_000
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    sig += np.complex64(3.0 + 2.0j)  # strong DC offset
    src = tmp_path / "dc_in.cf32"
    sig.tofile(src)
    snk = tmp_path / "dc_out.cf32"
    modem = ModemSpec(
        symbol_rate=4800.0,
        path=[
            ModemStep(conv="dc_block", params={"dc_block_len": 32}),
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=39062.0,
        start=Descriptor(Level.IQ, "c"),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=60.0).status == "ok"
    out = np.fromfile(snk, np.complex64)
    assert abs(out[1000:].mean()) < 0.2  # DC removed (input mean magnitude ~3.6)
