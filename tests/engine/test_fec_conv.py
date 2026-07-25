import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import Fec
from marconi.engine.stages.base import validate_params
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep


def _roundtrip(tmp_path, *, rate_inv, polys, frame_bits, seed):
    ensure_worker_warm()
    from gnuradio import blocks as gb
    from gnuradio import gr, trellis

    tail = 6
    rng = np.random.default_rng(seed)
    info = rng.integers(0, 2, frame_bits).astype(np.uint8)
    bits = np.concatenate([info, np.zeros(tail, np.uint8)])
    fsm = trellis.fsm(1, rate_inv, polys)

    class Enc(gr.top_block):
        def __init__(self, data):
            gr.top_block.__init__(self)
            src = gb.vector_source_b(list(map(int, data)), False)
            enc = trellis.encoder_bb(fsm, 0)
            snk = gb.vector_sink_b()
            self.connect(src, enc, snk)
            self.snk = snk

    e = Enc(bits)
    e.run()
    syms = np.array(e.snk.data(), np.int64)
    soft = np.empty(syms.size * rate_inv, np.float32)
    for i, s in enumerate(syms):
        for d in range(rate_inv):
            soft[i * rate_inv + d] = 1.0 - 2 * ((s >> (rate_inv - 1 - d)) & 1)
    src = tmp_path / f"s{seed}.f32"
    soft.tofile(src)
    snk = tmp_path / f"b{seed}.u8"
    modem = ModemSpec(
        name="fec",
        symbol_rate=1.0,
        path=[
            ModemStep(
                conv="fec",
                params={
                    "scheme": "cc",
                    "rate_inv": rate_inv,
                    "polys": list(polys),
                    "frame_bits": frame_bits,
                    "tail": tail,
                },
            )
        ],
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
    return read_bits(snk)[:frame_bits], info


def test_fec_conv_dab_rate_quarter(tmp_path):
    out, info = _roundtrip(
        tmp_path, rate_inv=4, polys=[0o133, 0o171, 0o145, 0o133], frame_bits=768, seed=1
    )
    assert np.array_equal(out, info)


def test_fec_conv_generic_rate_half(tmp_path):
    out, info = _roundtrip(
        tmp_path, rate_inv=2, polys=[0o133, 0o171], frame_bits=200, seed=2
    )
    assert np.array_equal(out, info)


_CC_PARAMS = {
    "scheme": "cc",
    "rate_inv": 2,
    "polys": [0o171, 0o133],
    "frame_bits": 100,
    "tail": 6,
}


def test_fec_params_reject_unknown_scheme():
    issues: list = []
    validate_params(
        "fec[0]", Fec().params_model, {**_CC_PARAMS, "scheme": "turbo"}, issues
    )
    assert issues, "unknown scheme must fail at validation, not KeyError at emit"


def test_fec_params_reject_nonpositive_and_empty():
    for override in ({"rate_inv": 0}, {"frame_bits": 0}, {"polys": []}, {"tail": -1}):
        issues: list = []
        validate_params(
            "fec[0]", Fec().params_model, {**_CC_PARAMS, **override}, issues
        )
        assert issues, f"{override} must be rejected"


def test_fec_params_accept_cc():
    ok: list = []
    validate_params("fec[0]", Fec().params_model, _CC_PARAMS, ok)
    assert not ok, ok
