from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_SR, _SYM, _DEV = 4.0, 1.0, 1.0


def make_clean_capture(tmp_path: Path, n_bits: int = 4096) -> Path:
    rng = np.random.default_rng(7)
    bits_path = tmp_path / "tx.u8"
    rng.integers(0, 2, n_bits).astype(np.uint8).tofile(bits_path)
    iq_path = tmp_path / "clean.cf32"
    tx = Modem(symbol_rate=_SYM, path=[FskStep(deviation=_DEV), SliceStep()])
    gr = compile_modem(
        tx,
        stage_registry(),
        direction="tx",
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(bits_path)},
        sink_io={"path": str(iq_path)},
    )
    ensure_worker_warm()
    r = GnuRadioBackend().run_pipeline(gr, timeout=120.0)
    assert r.status == "ok", r
    return iq_path
