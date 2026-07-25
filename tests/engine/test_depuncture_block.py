import numpy as np
import pytest

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.bitfile import read_llrs
from marconi.engine.compiler import CompileError, compile_modem
from marconi.engine.descriptor import Carrier, Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import ModemSpec, ModemStep
from marconi.engine.stages import stage_registry


def test_depuncture_scatters_per_mask(tmp_path):
    ensure_worker_warm()
    mask = [1, 1, 0, 1, 0, 0, 1, 1]
    data = np.array(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], np.float32
    )  # 2 full groups of 5 kept + a 2-item partial tail
    src = tmp_path / "in.f32"
    data.tofile(src)
    snk = tmp_path / "out.f32"
    modem = ModemSpec(
        name="dp",
        symbol_rate=1.0,
        path=[ModemStep(conv="depuncture", params={"keep_mask": mask})],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=Descriptor(Level.BITS, "f", "stream", Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    out = read_llrs(snk)
    assert out[:8].tolist() == [10, 20, 0, 30, 0, 0, 40, 50]
    assert out[8:16].tolist() == [60, 70, 0, 80, 0, 0, 90, 100]
    assert out.size == 16, "partial tail group must be dropped, not padded"


def test_all_erasure_mask_rejected_at_compile() -> None:
    """A mask keeping nothing consumes no input while emitting zeros forever —
    the flowgraph never terminates. Reject at spec validation."""
    modem = ModemSpec(
        name="dp0",
        symbol_rate=1.0,
        path=[ModemStep(conv="depuncture", params={"keep_mask": [0, 0, 0, 0]})],
    )
    with pytest.raises(CompileError, match="keep_mask"):
        compile_modem(
            modem,
            stage_registry(),
            direction="rx",
            sample_rate=1.0,
            start=Descriptor(Level.BITS, "f", carrier=Carrier.SOFT),
            source_io={"path": "a"},
            sink_io={"path": "b"},
        )
