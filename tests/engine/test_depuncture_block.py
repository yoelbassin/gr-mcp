import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_llrs
from marconi.engine.modulation.coding.stages import DepunctureStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem


def test_depuncture_scatters_per_mask(tmp_path):
    ensure_worker_warm()
    mask = [1, 1, 0, 1, 0, 0, 1, 1]
    data = np.array(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], np.float32
    )  # 2 full groups of 5 kept + a 2-item partial tail
    src = tmp_path / "in.f32"
    data.tofile(src)
    snk = tmp_path / "out.f32"
    modem = Modem(
        name="dp",
        symbol_rate=1.0,
        path=[DepunctureStep(keep_mask=mask)],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=Descriptor(Level.BITS, ItemType.F, Carrier.SOFT),
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
    with pytest.raises(ValidationError, match="keep_mask"):
        DepunctureStep(keep_mask=[0, 0, 0, 0])
