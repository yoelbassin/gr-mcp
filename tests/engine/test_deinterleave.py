import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_llrs
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep


def test_deinterleave_applies_perm(tmp_path):
    ensure_worker_warm()
    perm = [3, 0, 1, 2]
    data = np.array([10, 20, 30, 40, 50, 60, 70, 80], np.float32)
    src = tmp_path / "i.f32"
    data.tofile(src)
    snk = tmp_path / "o.f32"
    modem = ModemSpec(
        name="di",
        symbol_rate=1.0,
        path=[ModemStep(conv="deinterleave", params={"perm": perm})],
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
    assert out[:4].tolist() == [40, 10, 20, 30]  # out[t] = in[perm[t]]
    assert out[4:8].tolist() == [80, 50, 60, 70]
