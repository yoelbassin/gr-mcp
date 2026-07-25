"""Protocol #4 (issue 07): wireless M-Bus mode T decodes end-to-end through the
GENERIC production coding vocabulary, with every protocol constant supplied
test-side. wM-Bus is outside the LoRa/AIS/DAB training set; nothing in
packages/ names it. The decode exercises three primitives the training set
never needed -- the 3-of-6 line code (codebook), sync-word seeding, and
length-field framing -- plus a CRC as the oracle.

A SIM round-trip, not an off-air capture: the modem's coding tail is just
[codebook, sync_word] -- the coded-TX path was retired (tests/phy/coding/
test_partition.py::test_tx_with_coding_stage_is_a_compile_error), so the wire
fixture is built test-side in plain numpy from the same table/CRC/field
constants the RX modem decodes with, reproducing the old codec's TX chain
order (parse -> crc -> sync prepend -> codebook encode) by hand.

Per CLAUDE.md, the protocol's parameters (3-of-6 table, sync word, EN-13757
CRC, field layout) live here in the test, never in the ecosystem source.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from helpers import bitops, crc, framing, parse

from marconi.engine.bitfile import read_bits, write_bits
from marconi.engine.descriptor import Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import Bitstream, ModemSpec, ModemStep
from marconi.engine.run import run_rx
from marconi.engine.stages import stage_registry

BITS = Descriptor(Level.BITS, "b")

# --- wM-Bus mode T parameters (EN 13757-4) -- caller data, not ecosystem code ---
THREE_OF_SIX = [
    0x16, 0x0D, 0x0E, 0x0B, 0x1C, 0x19, 0x1A, 0x13,
    0x2C, 0x25, 0x26, 0x23, 0x34, 0x31, 0x32, 0x29,
]  # fmt: skip
SYNC = "7a96"  # a data-level frame sync word
CRC_PARAMS: dict[str, int] = {
    "poly": 0x3D65,
    "bits": 16,
    "init": 0x0000,
    "xorout": 0xFFFF,
}
FIELDS = [
    {"name": "length", "bits": 8},
    {"name": "c_field", "bits": 8},
    {"name": "m_field", "bits": 16},
    {"name": "address", "bits": 48},
]

# body = C + M(2) + A(6) = 9 bytes; length counts the bytes after the L-field
_MSG = {"length": 9, "c_field": 0x44, "m_field": 0x2D2C, "address": 0x112233445566}


def _wmbus_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(
                conv="codebook",
                params={
                    "code_bits": 6,
                    "data_bits": 4,
                    "table": cast("list[float | int]", THREE_OF_SIX),
                },
            ),
            ModemStep(conv="sync_word", params={"sync": SYNC}),
        ],
    )


def _codebook_encode(bits: np.ndarray, table: list[int]) -> np.ndarray:
    """3-of-6 TX, expressed directly over numpy (the coded-TX engine path is
    retired): each 4-bit nibble looks up its 6-bit codeword in the same table
    the RX-side codebook stage inverts."""
    data_bits, code_bits = 4, 6
    n = bits.size // data_bits
    nibbles = np.asarray(bits[: n * data_bits], np.int64).reshape(n, data_bits)
    in_weights = 1 << np.arange(data_bits - 1, -1, -1, dtype=np.int64)
    values = nibbles @ in_weights
    codewords = np.asarray(table, dtype=np.int64)[values]
    out_shifts = np.arange(code_bits - 1, -1, -1, dtype=np.int64)
    return ((codewords[:, None] >> out_shifts) & 1).reshape(-1).astype(np.uint8)


def _encode(msg: dict[str, int]) -> np.ndarray:
    # old TX chain order (compile_codec walked [codebook, sync_word,
    # length_frame, crc, parse] in reverse for TX): parse -> crc -> sync
    # prepend -> codebook encode. length_frame contributed nothing on TX (the
    # length value is already in the built message).
    body = parse.build_message(msg, FIELDS)
    framed = crc.crc_append(
        body,
        poly=CRC_PARAMS["poly"],
        bits=CRC_PARAMS["bits"],
        init=CRC_PARAMS["init"],
        xorout=CRC_PARAMS["xorout"],
    )
    sync_bits = bitops.bytes_to_bits(bytes.fromhex(SYNC))
    payload_bits = np.concatenate([sync_bits, bitops.bytes_to_bits(framed)]).astype(
        np.uint8
    )
    wire = _codebook_encode(payload_bits, THREE_OF_SIX)
    # realistic acquisition: a 3-of-6-valid preamble before the sync, junk after
    preamble = _codebook_encode(bitops.bytes_to_bits(bytes([0x55] * 4)), THREE_OF_SIX)
    trailer = _codebook_encode(bitops.bytes_to_bits(bytes([0xAA] * 3)), THREE_OF_SIX)
    return np.concatenate([preamble, wire, trailer]).astype(np.uint8)


def _decode(
    bits: np.ndarray, windows: list[int], *, poly: int
) -> tuple[int, int, list[dict[str, int | str]]]:
    num_frames = 0
    num_crc_ok = 0
    messages: list[dict[str, int | str]] = []
    for w in windows:
        payload = bitops.bits_to_bytes(bits[w:])
        carved = framing.carve_length(
            payload, length_bits=8, base_bytes=3, unit_bytes=1
        )
        if carved is None:
            continue
        num_frames += 1
        ok, body = crc.crc_check(
            carved,
            poly=poly,
            bits=CRC_PARAMS["bits"],
            init=CRC_PARAMS["init"],
            xorout=CRC_PARAMS["xorout"],
        )
        if not ok:
            continue
        num_crc_ok += 1
        msg = parse.parse_message(body, FIELDS)
        if msg is not None:
            messages.append(msg)
    return num_frames, num_crc_ok, messages


def _run_decode(tmp_path: Path, bits: np.ndarray) -> tuple[np.ndarray, list[int]]:
    p = tmp_path / "wmbus.u8"
    write_bits(p, bits)
    res = run_rx(
        _wmbus_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=Bitstream(path=p, num_bits=int(bits.size)),
    )
    assert res.status == "ok", res
    assert res.bitstream is not None
    return read_bits(res.bitstream.path), res.windows


def test_wmbus_mode_t_decodes_crc_valid(tmp_path: Path) -> None:
    decoded_bits, windows = _run_decode(tmp_path, _encode(_MSG))
    _num_frames, num_crc_ok, messages = _decode(
        decoded_bits, windows, poly=CRC_PARAMS["poly"]
    )
    assert num_crc_ok == 1, (num_crc_ok, messages)
    assert messages == [_MSG]


def test_wmbus_crc_is_a_real_oracle(tmp_path: Path) -> None:
    # decode the same wire with the wrong CRC polynomial: the frame is still
    # framed, but the checksum must not validate.
    decoded_bits, windows = _run_decode(tmp_path, _encode(_MSG))
    num_frames, num_crc_ok, _messages = _decode(decoded_bits, windows, poly=0x1021)
    assert num_frames == 1 and num_crc_ok == 0
