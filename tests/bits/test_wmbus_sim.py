"""Protocol #4 (issue 07): wireless M-Bus mode T decodes end-to-end through the
GENERIC production bits vocabulary, with every protocol constant supplied
test-side. wM-Bus is outside the LoRa/AIS/DAB training set; nothing in
packages/ names it. The decode exercises three primitives the training set
never needed — the 3-of-6 line code (codebook), sync-word seeding, and
length-field framing — plus the existing CRC as the oracle.

Per CLAUDE.md, the protocol's parameters (3-of-6 table, sync word, EN-13757
CRC, field layout) live here in the test, never in the ecosystem source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.bits import framing
from marconi.bits.carriers import TxCarrier
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.program import run_program
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.bits.validate import validate_codec
from marconi.core.bitfile import write_bits
from marconi.core.models import Bitstream

# --- wM-Bus mode T parameters (EN 13757-4) — caller data, not ecosystem code ---
THREE_OF_SIX = [
    0x16, 0x0D, 0x0E, 0x0B, 0x1C, 0x19, 0x1A, 0x13,
    0x2C, 0x25, 0x26, 0x23, 0x34, 0x31, 0x32, 0x29,
]  # fmt: skip
SYNC = "7a96"  # a data-level frame sync word
CRC: dict[str, float | int | str | bool | list] = {
    "poly": 0x3D65,
    "bits": 16,
    "init": 0x0000,
    "xorout": 0xFFFF,
    "reflected": False,
}
FIELDS = [
    {"name": "length", "bits": 8},
    {"name": "c_field", "bits": 8},
    {"name": "m_field", "bits": 16},
    {"name": "address", "bits": 48},
]


def _wmbus_codec() -> CodecSpec:
    return CodecSpec(
        name="wmbus_t",
        path=[
            CodecStep(
                conv="codebook",
                params={"code_bits": 6, "data_bits": 4, "table": THREE_OF_SIX},
            ),
            CodecStep(conv="sync_word", params={"sync": SYNC}),
            CodecStep(
                conv="length_frame",
                params={"length_bits": 8, "base_bytes": 3, "unit_bytes": 1},
            ),
            CodecStep(conv="crc", params=CRC),
            CodecStep(conv="parse", params={"fields": FIELDS}),
        ],
    )


# body = C + M(2) + A(6) = 9 bytes; length counts the bytes after the L-field
_MSG = {"length": 9, "c_field": 0x44, "m_field": 0x2D2C, "address": 0x112233445566}


def _encode(codec: CodecSpec) -> np.ndarray:
    tx = run_program(compile_codec(codec, registry(), "tx"), TxCarrier(items=[_MSG]))
    wire = np.asarray(tx.items[0], np.uint8)
    # realistic acquisition: a 3-of-6-valid preamble before the sync, junk after
    preamble = framing.codebook_tx(
        TxCarrier([framing.bytes_to_bits(bytes([0x55] * 4))]),
        code_bits=6,
        data_bits=4,
        table=THREE_OF_SIX,
    ).items[0]
    trailer = framing.codebook_tx(
        TxCarrier([framing.bytes_to_bits(bytes([0xAA] * 3))]),
        code_bits=6,
        data_bits=4,
        table=THREE_OF_SIX,
    ).items[0]
    return np.concatenate([preamble, wire, trailer]).astype(np.uint8)


def test_wmbus_codec_validates() -> None:
    assert not validate_codec(_wmbus_codec(), registry())


def test_wmbus_mode_t_decodes_crc_valid(tmp_path: Path) -> None:
    codec = _wmbus_codec()
    bits = _encode(codec)
    p = tmp_path / "wmbus.u8"
    write_bits(p, bits)
    res = parse_bitstream(Bitstream(path=p, num_bits=int(bits.size)), codec, registry())
    assert res.num_crc_ok == 1, res
    assert res.messages == [_MSG]


def test_wmbus_crc_is_a_real_oracle(tmp_path: Path) -> None:
    # decode the same wire with the wrong CRC polynomial: the frame is still
    # framed and parsed, but the checksum must not validate.
    bits = _encode(_wmbus_codec())
    p = tmp_path / "wmbus.u8"
    write_bits(p, bits)
    wrong = _wmbus_codec()
    wrong.path[3].params["poly"] = 0x1021  # not EN-13757
    res = parse_bitstream(Bitstream(path=p, num_bits=int(bits.size)), wrong, registry())
    assert res.num_frames == 1 and res.num_crc_ok == 0
