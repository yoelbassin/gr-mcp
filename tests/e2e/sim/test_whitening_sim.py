"""Per-packet seeded whitening through the GENERIC vocabulary: sync_word
seeds a window per packet, descramble's LFSR mode restarts from the seed at
each window — the shape of clock-seeded whitening (BT BR/EDR-like). A fixed
XOR sequence shorter than a frame body wraps mid-frame, diverging from the
non-periodic LFSR it's standing in for — proving the per-window reset is
load-bearing. LFSR polynomial/seed, sync word and CRC are caller data, only
here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers import bitops, crc

from marconi.engine.coding.stages_bits import DescrambleStep, SyncWordStep
from marconi.engine.io.bitfile import read_bits, write_bits
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem
from marconi.engine.types.step import Step

BITS = Descriptor(Level.BITS, ItemType.B)
SYNC = "5c3b"
MASK, SEED, LEN = 0b0000011, 0x55, 7
CRC_PARAMS = {"poly": 0x1021, "bits": 16, "init": 0xFFFF, "xorout": 0x0000}
_PAYLOADS = [bytes.fromhex("c0ffee00112233"), bytes.fromhex("deadbeef445566")]
FRAME_BODY_BYTES = 7 + 2  # payload + CRC-16


def _lfsr(n: int) -> np.ndarray:
    out, s = np.empty(n, np.uint8), SEED
    for i in range(n):
        out[i] = s & 1
        fb = bin(s & MASK).count("1") & 1
        s = (s >> 1) | (fb << (LEN - 1))
    return out


def _frame(payload: bytes) -> np.ndarray:
    body = bitops.bytes_to_bits(crc.crc_append(payload, **CRC_PARAMS))
    whitened = body ^ _lfsr(body.size)
    return np.concatenate([bitops.bytes_to_bits(bytes.fromhex(SYNC)), whitened]).astype(
        np.uint8
    )


def _wire() -> np.ndarray:
    # junk gaps put each frame at an arbitrary bit offset — realistic
    # framing; see the control test for the actual discriminator.
    return np.concatenate(
        [
            np.zeros(9, np.uint8),
            _frame(_PAYLOADS[0]),
            np.ones(13, np.uint8),
            _frame(_PAYLOADS[1]),
            np.zeros(5, np.uint8),
        ]
    )


def _run(tmp_path: Path, steps: list[Step]) -> list[bytes]:
    wire = _wire()
    p = tmp_path / "wire.u8"
    write_bits(p, wire)
    res = run_rx(
        Modem(symbol_rate=1.0, path=steps),
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=Bitstream(path=p, num_bits=int(wire.size)),
    )
    assert res.status == "ok", res
    assert res.bitstream is not None
    bits = read_bits(res.bitstream.path)
    out = []
    for w in res.windows:
        want = FRAME_BODY_BYTES * 8
        if w + want > bits.size:
            continue
        ok, body = crc.crc_check(bitops.bits_to_bytes(bits[w : w + want]), **CRC_PARAMS)
        if ok:
            out.append(body)
    return out


def test_seeded_whitening_decodes_both_frames(tmp_path: Path) -> None:
    steps: list[Step] = [
        SyncWordStep(sync=SYNC),
        DescrambleStep(lfsr_mask=MASK, lfsr_seed=SEED, lfsr_len=LEN),
    ]
    assert _run(tmp_path, steps) == _PAYLOADS


def test_fixed_sequence_cannot_fake_per_window_reset(tmp_path: Path) -> None:
    # descramble's sequence mode also restarts its tile at each window's
    # cursor (same reset point as lfsr mode), so a tile at least as long as
    # a frame body reproduces the true per-window LFSR exactly and both
    # frames would decode regardless of the misaligned junk gap. The
    # discriminator is tile length, not gap length: shorter than
    # FRAME_BODY_BYTES forces a byte-tile wrap inside the CRC-checked span,
    # where the true LFSR (aperiodic well past this span) has moved on.
    seq_hex = np.packbits(_lfsr(4 * 8)).tobytes().hex()
    steps: list[Step] = [
        SyncWordStep(sync=SYNC),
        DescrambleStep(sequence=seq_hex),
    ]
    got = _run(tmp_path, steps)
    assert got != _PAYLOADS
    assert len(got) < 2
