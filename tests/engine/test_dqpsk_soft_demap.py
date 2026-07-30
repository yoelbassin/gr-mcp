import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_llrs
from marconi.engine.modulation.ofdm.stages import DqpskSoftDemapStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

NC, DS = 5, 3  # 5 carriers, PRS + 3 FIC symbols


def test_dqpsk_soft_demap_recovers_bits(tmp_path):
    ensure_worker_warm()
    rng = np.random.default_rng(2)
    qpsk = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    carriers = np.empty((DS + 1, NC), complex)
    carriers[0] = 1.0  # PRS reference
    syms = rng.integers(0, 4, (DS, NC))
    for s in range(DS):
        carriers[s + 1] = carriers[s] * qpsk[syms[s]]  # differential encode
    smaj = carriers.reshape(-1).astype(np.complex64)  # symbol-major
    src = tmp_path / "c.cf32"
    smaj.tofile(src)
    snk = tmp_path / "s.f32"
    modem = Modem(
        name="dm",
        symbol_rate=1.0,
        path=[DqpskSoftDemapStep(data_syms=DS, n_carriers=NC, scheme="psk", order=4)],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=Descriptor(Level.SYMBOLS, "c", carrier=Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    soft = read_llrs(snk)
    assert soft.size == DS * NC * 2  # PRS dropped, 2 soft/carrier
    assert np.all(np.abs(soft) > 0.1)  # clean QPSK -> confident
