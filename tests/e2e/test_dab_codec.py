"""DAB's coding tail (descramble -> segment) over a golden decoded bit array,
deterministic — pins the pure-coding half of run_rx: a Bitstream in, 12
segmented windows out (4 CIF x 3 FIBs), every one CRC-16 valid. The golden
bits are the FEC-decoded output of tests/e2e/test_dab_offair.py's phy chain on
one real off-air super-frame, captured once (data/dab_decoded_golden.npy,
uint8 shape (3072,); provenance: tests/bits/test_dab_codec.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import bitops, crc, framing

from marconi.engine.bitfile import read_bits, write_bits
from marconi.engine.descriptor import Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import Bitstream, ModemSpec, ModemStep
from marconi.engine.run import run_rx
from marconi.engine.stages import stage_registry

BITS = Descriptor(Level.BITS, "b")
_FRAME_BODY_LEN = 256
_GOLD = Path(__file__).resolve().parent / "data" / "dab_decoded_golden.npy"


def _prbs_hex() -> str:
    sr = [1] * 9
    prbs = []
    for _ in range(768):
        b = sr[8] ^ sr[4]
        prbs.append(b)
        sr = [b] + sr[:8]
    return np.packbits(np.array(prbs, np.uint8)).tobytes().hex()


def _golden_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="descramble", params={"sequence": _prbs_hex()}),
            ModemStep(conv="segment", params={"frame_body_len": _FRAME_BODY_LEN}),
        ],
    )


def _golden_stream(tmp_path: Path) -> Bitstream:
    bits = np.load(_GOLD).astype(np.uint8)  # 3072 whitened decoded bits
    p = tmp_path / "golden.u8"
    write_bits(p, bits)
    return Bitstream(path=p, num_bits=int(bits.size))


@pytest.mark.skipif(not _GOLD.exists(), reason="golden absent — see Step 2")
def test_dab_codec_twelve_crc_valid_fibs(tmp_path: Path) -> None:
    res = run_rx(
        _golden_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=_golden_stream(tmp_path),
    )
    assert res.status == "ok", res
    assert res.bitstream is not None
    bits = read_bits(res.bitstream.path)

    windows = framing.carve_fixed(bits, res.windows, _FRAME_BODY_LEN)
    assert len(windows) == 12  # 4 CIF x 3 FIBs
    bodies: list[bytes] = []
    for window in windows:
        ok, body = crc.crc_check(
            bitops.bits_to_bytes(window),
            poly=0x1021,
            bits=16,
            init=0xFFFF,
            xorout=0xFFFF,
        )
        assert ok, "expected every golden FIB to be CRC-16 valid"
        bodies.append(body)
    assert all(len(body) == 30 for body in bodies)
