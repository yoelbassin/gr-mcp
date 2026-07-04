from __future__ import annotations

import numpy as np
import pytest

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.program import run_program
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.bitfile import write_bits
from marconi.core.models import Bitstream

# EN 13757-4 3-of-6 (published symbol table; also anchors the wM-Bus vector:
# data 0x0 -> 010110b, 0xF -> 101001b).
THREE_OF_SIX = [
    0x16, 0x0D, 0x0E, 0x0B, 0x1C, 0x19, 0x1A, 0x13,
    0x2C, 0x25, 0x26, 0x23, 0x34, 0x31, 0x32, 0x29,
]  # fmt: skip


def test_three_of_six_table_is_the_published_one() -> None:
    assert THREE_OF_SIX[0x0] == 0b010110
    assert THREE_OF_SIX[0xF] == 0b101001


def _enc(nibbles: list[int]) -> np.ndarray:
    return np.concatenate(
        [[(THREE_OF_SIX[n] >> (5 - j)) & 1 for j in range(6)] for n in nibbles]
    ).astype(np.uint8)


def test_pre_slice_decodes_from_each_cursor() -> None:
    coded = _enc([0x4, 0x4, 0xA, 0xB])
    stream = np.concatenate([[1, 0, 1], coded]).astype(np.uint8)  # misaligned!
    frames = [_Frame(start=3, cursor=3)]
    out = framing.frame_codebook_rx(
        RxCarrier(bits=stream, frames=frames),
        code_bits=6,
        data_bits=4,
        table=THREE_OF_SIX,
    )
    assert out.frames[0].cursor == 0
    assert framing.bits_to_bytes(out.bits[:16]) == bytes([0x44, 0xAB])


def test_pre_slice_regions_are_per_frame() -> None:
    coded = np.concatenate([_enc([0x1, 0x2]), _enc([0xD, 0xE])]).astype(np.uint8)
    frames = [_Frame(start=0, cursor=0), _Frame(start=12, cursor=12)]
    out = framing.frame_codebook_rx(
        RxCarrier(bits=coded, frames=frames),
        code_bits=6,
        data_bits=4,
        table=THREE_OF_SIX,
    )
    assert [f.cursor for f in out.frames] == [0, 8]
    assert framing.bits_to_bytes(out.bits) == bytes([0x12, 0xDE])


def test_non_increasing_cursors_raise() -> None:
    frames = [_Frame(start=6, cursor=6), _Frame(start=0, cursor=0)]
    with pytest.raises(ValueError, match="strictly increasing"):
        framing.frame_codebook_rx(
            RxCarrier(bits=np.zeros(24, np.uint8), frames=frames),
            code_bits=6,
            data_bits=4,
            table=THREE_OF_SIX,
        )


def test_post_slice_transforms_payload() -> None:
    payload_bits = _enc([0x7, 0x7])
    frames = [_Frame(start=0, cursor=12, payload=framing.bits_to_bytes(payload_bits))]
    out = framing.frame_codebook_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), frames=frames),
        code_bits=6,
        data_bits=4,
        table=THREE_OF_SIX,
    )
    assert out.frames[0].payload == bytes([0x77])


def test_tx_is_forward_map_not_identity() -> None:
    tx = framing.frame_codebook_tx(
        TxCarrier([framing.bytes_to_bits(bytes([0x44]))]),
        code_bits=6,
        data_bits=4,
        table=THREE_OF_SIX,
    )
    assert tx.items[0].tolist() == _enc([0x4, 0x4]).tolist()


def _acceptance_codec(sync_hex: str) -> CodecSpec:
    return CodecSpec(
        path=[
            CodecStep(conv="sync_word", params={"sync": sync_hex}),
            CodecStep(
                conv="frame_codebook",
                params={"code_bits": 6, "data_bits": 4, "table": THREE_OF_SIX},
            ),
            CodecStep(
                conv="length_frame",
                params={"length_bits": 8, "base_bytes": 1, "unit_bytes": 1},
            ),
            CodecStep(
                conv="parse",
                params={
                    "fields": [
                        {"name": "length", "bits": 8},
                        {"name": "value", "bits": 16},
                    ]
                },
            ),
        ],
    )


def test_tx_rx_round_trip_through_compiler(tmp_path) -> None:
    # Drive the realistic compiler path (not a hand-built bit-array direct call):
    # frame_codebook is FRAMES->FRAMES, so in a TX pipeline its input items are
    # bytes (parse_tx/length_frame_tx convention), not 0/1 bit-arrays. TX must
    # bridge bytes->bits before the forward map; a verbatim codebook_tx delegation
    # crashes here. Full duplex: encode a message, then decode it back through the
    # SAME codec and recover it.
    msg = {"length": 2, "value": 0x1234}
    coded_sync = _enc([0xF, 0x0, 0x5, 0xA])
    codec = _acceptance_codec(framing.bits_to_bytes(coded_sync).hex())
    tx = run_program(compile_codec(codec, registry(), "tx"), TxCarrier(items=[msg]))
    wire = np.asarray(tx.items[0], np.uint8)
    p = tmp_path / "roundtrip.u8"
    write_bits(p, wire)
    res = parse_bitstream(Bitstream(path=p, num_bits=int(wire.size)), codec, registry())
    assert res.num_frames == 1
    assert res.messages == [msg]


def test_chip_domain_sync_then_decode_end_to_end(tmp_path) -> None:
    # THE acceptance fixture (spec: sync in the CODED stream, then line-decode
    # from that offset, capture NOT starting on a symbol boundary). wM-Bus-shaped
    # data, all constants test-side. The 4-symbol sync codes to 24 chips = 3
    # whole bytes, so it is expressible as sync_word hex while the 5 junk bits
    # up front keep every symbol boundary off byte alignment.
    body = [0x0, 0x2, 0x1, 0x2, 0x3, 0x4]  # length=2, then bytes 0x12 0x34
    coded_sync = _enc([0xF, 0x0, 0x5, 0xA])  # 24 chips -> "a56666"
    coded_body = _enc(body)
    stream = np.concatenate(
        [[0, 1, 1, 0, 1], coded_sync, coded_body, [1, 0, 1]]
    ).astype(np.uint8)
    sync_hex = framing.bits_to_bytes(coded_sync).hex()
    codec = CodecSpec(
        path=[
            CodecStep(conv="sync_word", params={"sync": sync_hex}),
            CodecStep(
                conv="frame_codebook",
                params={"code_bits": 6, "data_bits": 4, "table": THREE_OF_SIX},
            ),
            CodecStep(
                conv="length_frame",
                params={"length_bits": 8, "base_bytes": 1, "unit_bytes": 1},
            ),
            CodecStep(
                conv="parse",
                params={
                    "fields": [
                        {"name": "length", "bits": 8},
                        {"name": "value", "bits": 16},
                    ]
                },
            ),
        ],
    )
    p = tmp_path / "chips.u8"
    write_bits(p, stream)
    res = parse_bitstream(
        Bitstream(path=p, num_bits=int(stream.size)), codec, registry()
    )
    assert res.num_frames == 1
    assert res.messages == [{"length": 2, "value": 0x1234}]
