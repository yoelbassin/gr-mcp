"""DSSS despread through the GENERIC vocabulary: 32-chip spreading decodes as
codebook(decode="nearest") — min-distance despreading IS nearest-codeword
decode over chip windows. Chip table (an IEEE 802.15.4 O-QPSK *shaped*
16-entry/32-chip codebook — rotations of a base chip sequence plus an
odd-chip-inverted complement group, the same construction the standard uses;
the transcribed standard literals failed the pairwise-distance self-check
below by one bit, so the table here is regenerated from that construction
rather than verbatim spec values), sync word and CRC live here, never in
src/. Chip-level sim with injected chip errors; the IQ half of this shape
rides the capture-hunt plan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers import bitops, crc

from marconi.engine.coding.stages_bits import CodebookStep, SyncWordStep
from marconi.engine.io.bitfile import read_bits, write_bits
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import DecodeMode
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem

BITS = Descriptor(Level.BITS, "b")

# 32-chip / 4-bit-symbol codebook, MSB-first ints — caller data. Same shape
# as IEEE 802.15.4 O-QPSK: entries 1-7 are entry 0 left-rotated by 4 chips at
# a time, entries 8-15 are entries 0-7 with odd-indexed chips inverted. The
# transcribed 802.15.4 spec literals produced this exact construction but
# missed the >=13 pairwise-distance bound by one bit (verified below), so
# entry 0 here is a regenerated (non-IEEE-verbatim) seed carrying the same
# construction through the self-check with margin.
CHIPS = [
    0b10000011100111111011110001010000,
    0b00111001111110111100010100001000,
    0b10011111101111000101000010000011,
    0b11111011110001010000100000111001,
    0b10111100010100001000001110011111,
    0b11000101000010000011100111111011,
    0b01010000100000111001111110111100,
    0b00001000001110011111101111000101,
    0b11010110110010101110100100000101,
    0b01101100101011101001000001011101,
    0b11001010111010010000010111010110,
    0b10101110100100000101110101101100,
    0b11101001000001011101011011001010,
    0b10010000010111010110110010101110,
    0b00000101110101101100101011101001,
    0b01011101011011001010111010010000,
]
SYNC = "00a7"  # SFD-like sync at the data level
CRC_PARAMS = {"poly": 0x1021, "bits": 16, "init": 0x0000, "xorout": 0x0000}
_PAYLOAD = bytes.fromhex("48656c6c6f20444553530000")


def _spread(bits: np.ndarray) -> np.ndarray:
    nib = bits.reshape(-1, 4)
    weights = 1 << np.arange(3, -1, -1, dtype=np.int64)
    vals = nib.astype(np.int64) @ weights
    chips = np.asarray(CHIPS, np.int64)[vals]
    shifts = np.arange(31, -1, -1, dtype=np.int64)
    return ((chips[:, None] >> shifts) & 1).reshape(-1).astype(np.uint8)


def _wire(chip_errors_per_symbol: int, seed: int) -> np.ndarray:
    framed = crc.crc_append(_PAYLOAD, **CRC_PARAMS)
    payload_bits = np.concatenate(
        [bitops.bytes_to_bits(bytes.fromhex(SYNC)), bitops.bytes_to_bits(framed)]
    ).astype(np.uint8)
    chips = _spread(payload_bits)
    rng = np.random.default_rng(seed)
    if chip_errors_per_symbol:
        for s in range(chips.size // 32):
            pos = rng.choice(32, size=chip_errors_per_symbol, replace=False)
            chips[s * 32 + pos] ^= 1
    lead = _spread(np.zeros(8, np.uint8))
    tail = _spread(np.ones(8, np.uint8))
    return np.concatenate([lead, chips, tail])


def _decode(tmp_path: Path, wire: np.ndarray) -> list[bytes]:
    modem = Modem(
        symbol_rate=1.0,
        path=[
            CodebookStep(
                code_bits=32,
                data_bits=4,
                table=CHIPS,
                decode=DecodeMode.NEAREST,
            ),
            SyncWordStep(sync=SYNC),
        ],
    )
    p = tmp_path / "dsss.u8"
    write_bits(p, wire)
    res = run_rx(
        modem,
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
    want = (len(_PAYLOAD) + 2) * 8
    for w in res.windows:
        if w + want > bits.size:
            continue
        ok, body = crc.crc_check(bitops.bits_to_bytes(bits[w : w + want]), **CRC_PARAMS)
        if ok:
            out.append(body)
    return out


def test_chip_table_supports_six_error_despread() -> None:
    # self-check the caller data: 6-error correction needs pairwise chip
    # distance >= 13 across the whole table
    shifts = np.arange(31, -1, -1, dtype=np.int64)
    rows = ((np.asarray(CHIPS, np.int64)[:, None] >> shifts) & 1).astype(np.uint8)
    for i in range(16):
        for j in range(i + 1, 16):
            assert int((rows[i] != rows[j]).sum()) >= 13, (i, j)


def test_dsss_despreads_clean_chips(tmp_path: Path) -> None:
    assert _decode(tmp_path, _wire(0, seed=1)) == [_PAYLOAD]


def test_dsss_despreads_through_heavy_chip_noise(tmp_path: Path) -> None:
    # 6 of 32 chips flipped in EVERY symbol: far beyond any exact-match
    # decode, well inside the chip-code distance.
    assert _decode(tmp_path, _wire(6, seed=2)) == [_PAYLOAD]
