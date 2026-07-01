# tests/bits/test_dab_codec.py
from pathlib import Path

import numpy as np
import pytest

from marconi.bits.carriers import RxCarrier
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.program import run_program
from marconi.bits.registry import registry

_GOLD = Path(__file__).resolve().parent / "data" / "dab_decoded_golden.npy"


def _prbs_hex() -> str:
    sr = [1] * 9
    prbs = []
    for _ in range(768):
        b = sr[8] ^ sr[4]
        prbs.append(b)
        sr = [b] + sr[:8]
    return np.packbits(np.array(prbs, np.uint8)).tobytes().hex()


def _codec() -> CodecSpec:
    return CodecSpec(
        name="dab_fic",
        path=[
            CodecStep(conv="descramble_bits", params={"sequence": _prbs_hex()}),
            CodecStep(conv="segment", params={"frame_body_len": 256}),
            CodecStep(conv="fixed_frame", params={"payload_bits": 256}),
            CodecStep(
                conv="crc",
                params={
                    "poly": 0x1021,
                    "bits": 16,
                    "init": 0xFFFF,
                    "reflected": False,
                    "xorout": 0xFFFF,
                },
            ),
            CodecStep(
                conv="parse",
                params={
                    "bit_order": "msb",
                    "fields": [{"name": "fib_data", "bits": 240}],
                },
            ),
        ],
    )


@pytest.mark.skipif(not _GOLD.exists(), reason="golden absent — see Step 2")
def test_dab_codec_twelve_crc_valid_fibs():
    bits = np.load(_GOLD).astype(np.uint8)  # 3072 whitened decoded bits
    prog = compile_codec(_codec(), registry(), "rx")
    out = run_program(prog, RxCarrier(bits=bits))
    assert len(out.frames) == 12  # 4 CIF x 3 FIBs
    assert all(f.crc_ok is True for f in out.frames)
    assert all(len(f.payload) == 30 for f in out.frames)
