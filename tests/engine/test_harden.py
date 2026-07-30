import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import HardenStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem


def test_harden_matches_bitfile_convention(tmp_path) -> None:
    """harden decides bit 1 = negative LLR, the same rule as bitfile.harden and
    the soft coding lane (trellis Euclidean bit 0 = +1)."""
    ensure_worker_warm()
    llrs = np.array([2.0, -2.0, 0.5, -0.5, 4.0, -4.0, 0.1, -0.1], np.float32)
    src = tmp_path / "in.f32"
    llrs.tofile(src)
    snk = tmp_path / "out.u8"
    modem = Modem(symbol_rate=1.0, path=[HardenStep()])
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=Descriptor(Level.BITS, "f", Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    out = read_bits(snk)
    assert out.tolist() == (llrs < 0).astype(np.uint8).tolist()


def test_harden_rejects_hard_input() -> None:
    """harden consumes soft LLRs; a hard-bit input is a seam error at compile."""
    modem = Modem(
        symbol_rate=1.0,
        path=[HardenStep(), HardenStep()],
    )
    try:
        compile_modem(
            modem,
            stage_registry(),
            direction="rx",
            sample_rate=1.0,
            start=Descriptor(Level.BITS, "f", Carrier.SOFT),
            source_io={"path": "x"},
            sink_io={"path": "y"},
        )
    except CompileError as e:
        assert "harden" in str(e)
    else:
        raise AssertionError("harden after harden (hard input) must fail the seam")
