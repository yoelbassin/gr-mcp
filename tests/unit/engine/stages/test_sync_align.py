import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import FecStep, SyncAlignStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

SYNC = "10110100011100101101001110001011"  # 32-bit access code


def _cc_encode(info: np.ndarray, rate_inv: int, polys: list[int]) -> np.ndarray:
    from gnuradio import blocks as gb
    from gnuradio import gr, trellis

    fsm = trellis.fsm(1, rate_inv, polys)

    class Enc(gr.top_block):
        def __init__(self, data: np.ndarray) -> None:
            gr.top_block.__init__(self)
            src = gb.vector_source_b(list(map(int, data)), False)
            enc = trellis.encoder_bb(fsm, 0)
            snk = gb.vector_sink_b()
            self.connect(src, enc, snk)
            self.snk = snk

    e = Enc(info)
    e.run()
    syms = np.array(e.snk.data(), np.int64)
    soft = np.empty(syms.size * rate_inv, np.float32)
    for i, s in enumerate(syms):
        for d in range(rate_inv):
            soft[i * rate_inv + d] = 1.0 - 2 * ((s >> (rate_inv - 1 - d)) & 1)
    return soft


def _soft_bits(bits: np.ndarray) -> np.ndarray:
    return (1.0 - 2.0 * bits.astype(np.float32)).astype(np.float32)


def test_sync_align_then_fec_decodes_bursts(tmp_path) -> None:
    ensure_worker_warm()
    frame_bits, tail, rate_inv = 96, 6, 2
    polys = [0o7, 0o5]
    frame_len = (frame_bits + tail) * rate_inv
    sync_soft = _soft_bits(np.array([int(c) for c in SYNC], np.uint8))

    rng = np.random.default_rng(3)
    infos = [rng.integers(0, 2, frame_bits).astype(np.uint8) for _ in range(3)]
    parts: list[np.ndarray] = []
    for k, info in enumerate(infos):
        gap = _soft_bits(rng.integers(0, 2, 40 + 13 * k).astype(np.uint8))
        coded = _cc_encode(
            np.concatenate([info, np.zeros(tail, np.uint8)]), rate_inv, polys
        )
        coded[17] = -coded[17]  # inject a channel error inside the frame
        parts += [gap, sync_soft, coded]
    parts.append(_soft_bits(rng.integers(0, 2, 30).astype(np.uint8)))
    stream = np.concatenate(parts).astype(np.float32)

    src = tmp_path / "in.f32"
    stream.tofile(src)
    snk = tmp_path / "out.u8"
    modem = Modem(
        name="sync_fec",
        symbol_rate=1.0,
        path=[
            SyncAlignStep(access_code=SYNC, frame_len=frame_len, threshold=2),
            FecStep(
                scheme="cc",
                rate_inv=rate_inv,
                polys=polys,
                frame_bits=frame_bits,
                tail=tail,
            ),
        ],
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
    out = read_bits(snk)
    assert out.size == frame_bits * len(infos), out.size
    expect = np.concatenate(infos)
    assert np.array_equal(out, expect), np.flatnonzero(out != expect)
    # the tag_gate census point: one sync tag per burst, chance expectation
    # far below one - this is what feeds sync_matches evidence on GR paths
    tags = next(d for d in r.diagnostics if d.key == "sync_tags")
    chance = next(d for d in r.diagnostics if d.key == "sync_chance_micro")
    assert tags.count == len(infos), r.diagnostics
    assert chance.count is not None and chance.count < 1000
