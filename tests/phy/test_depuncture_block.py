import numpy as np

from marconi.core.bitfile import read_llrs
from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry


def test_depuncture_scatters_per_mask(tmp_path):
    ensure_worker_warm()
    mask = [1, 1, 0, 1, 0, 0, 1, 1]
    data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], np.float32)
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
