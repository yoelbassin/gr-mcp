from pathlib import Path

import numpy as np

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.io.bitfile import read_bits, write_bits
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, ModemSpec, ModemStep

POLYS = [0o7, 0o5]
POLYS_PARAM: list[float | int] = list(POLYS)


def _encode(info: np.ndarray, rate_inv: int, polys: list[int]) -> np.ndarray:
    # GR fsm convention (probed): reg = (u << m) | state, next = reg >> 1,
    # poly 0 = first output bit
    m = max(int(p).bit_length() for p in polys) - 1
    state = 0
    out: list[int] = []
    for u in info:
        reg = (int(u) << m) | state
        out.extend((int(p) & reg).bit_count() & 1 for p in polys)
        state = reg >> 1
    return np.asarray(out, np.uint8)


def test_conv_code_terminated_frame_with_channel_errors() -> None:
    frame_bits, tail = 96, 6
    rng = np.random.default_rng(1)
    info = rng.integers(0, 2, frame_bits).astype(np.uint8)
    coded = _encode(np.concatenate([info, np.zeros(tail, np.uint8)]), 2, POLYS)
    coded[10] ^= 1
    coded[60] ^= 1
    out = ops_bits.conv_code_rx(
        CodingCarrier(bits=coded),
        rate_inv=2,
        polys=POLYS,
        frame_bits=frame_bits,
        tail=tail,
    )
    assert np.array_equal(out.bits, info)


def test_conv_code_unterminated_frame() -> None:
    frame_bits = 64
    rng = np.random.default_rng(2)
    info = rng.integers(0, 2, frame_bits).astype(np.uint8)
    info[-1] = 1
    coded = _encode(info, 2, POLYS)
    out = ops_bits.conv_code_rx(
        CodingCarrier(bits=coded),
        rate_inv=2,
        polys=POLYS,
        frame_bits=frame_bits,
        tail=0,
    )
    assert np.array_equal(out.bits, info)


def test_conv_code_matches_gr_encoder() -> None:
    from marconi.engine.backends.gnuradio.runner import ensure_worker_warm

    ensure_worker_warm()
    from gnuradio import blocks as gb
    from gnuradio import gr, trellis

    frame_bits, tail = 128, 6
    rng = np.random.default_rng(5)
    info = rng.integers(0, 2, frame_bits).astype(np.uint8)
    fsm = trellis.fsm(1, 2, POLYS)

    class Enc(gr.top_block):
        def __init__(self, data):
            gr.top_block.__init__(self)
            src = gb.vector_source_b(list(map(int, data)), False)
            enc = trellis.encoder_bb(fsm, 0)
            snk = gb.vector_sink_b()
            self.connect(src, enc, snk)
            self.snk = snk

    e = Enc(np.concatenate([info, np.zeros(tail, np.uint8)]))
    e.run()
    syms = np.array(e.snk.data(), np.int64)
    coded = np.stack([(syms >> 1) & 1, syms & 1], axis=1).reshape(-1).astype(np.uint8)
    out = ops_bits.conv_code_rx(
        CodingCarrier(bits=coded),
        rate_inv=2,
        polys=POLYS,
        frame_bits=frame_bits,
        tail=tail,
    )
    assert np.array_equal(out.bits, info)


def test_conv_code_windowed_after_sync_word(tmp_path: Path) -> None:
    frame_bits, tail = 96, 6
    rng = np.random.default_rng(9)
    frames = [rng.integers(0, 2, frame_bits).astype(np.uint8) for _ in range(3)]
    sync = "d391"
    sync_bits = ops_bits.bytes_to_bits(bytes.fromhex(sync))
    stream = [rng.integers(0, 2, 137).astype(np.uint8)]
    for info in frames:
        coded = _encode(np.concatenate([info, np.zeros(tail, np.uint8)]), 2, POLYS)
        coded[17] ^= 1
        stream += [sync_bits, coded, rng.integers(0, 2, 61).astype(np.uint8)]
    bits = np.concatenate(stream)
    path = tmp_path / "in.u8"
    write_bits(path, bits)
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="sync_word", params={"sync": sync}),
            ModemStep(
                conv="conv_code",
                params={
                    "rate_inv": 2,
                    "polys": POLYS_PARAM,
                    "frame_bits": frame_bits,
                    "tail": tail,
                },
            ),
        ],
    )
    r = run_rx(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=Descriptor(Level.BITS, "b", carrier=Carrier.HARD),
        workdir=tmp_path,
        input_stream=Bitstream(path=path, num_bits=int(bits.size)),
    )
    assert r.status == "ok", r
    assert r.bitstream is not None
    got = read_bits(r.bitstream.path)
    assert np.array_equal(got, np.concatenate(frames))
