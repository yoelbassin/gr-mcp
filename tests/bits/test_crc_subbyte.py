from __future__ import annotations

import numpy as np
import pytest

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, _Frame
from marconi.bits.stages.integrity import Crc
from marconi.core.stages import validate_params


def test_crc10_atm_catalog_vector() -> None:
    # CRC-10/ATM from the canonical CRC catalog (reveng): poly=0x233, init=0,
    # xorout=0, check("123456789") = 0x199 — the external anchor for the
    # bit-domain engine.
    bits = framing.bytes_to_bits(b"123456789")
    assert framing._crc_bits(bits, poly=0x233, width=10, init=0, xorout=0) == 0x199


def _frame_with(payload_bits: np.ndarray) -> RxCarrier:
    return RxCarrier(
        bits=np.zeros(0, np.uint8),
        frames=[
            _Frame(
                start=0,
                cursor=int(payload_bits.size),
                payload=framing.bits_to_bytes(payload_bits),
            )
        ],
    )


def test_rds_style_10bit_checkword_validates() -> None:
    # RDS (EN 50067) block: 16 info bits + 10-bit checkword where checkword =
    # crc10(info) XOR offset word A (0x0FC), generator 0x1B9. The engine is
    # anchored by the catalog vector above; this exercises the sub-byte flow
    # (payload_bits, xorout-as-offset) end to end.
    info = 0x3AB0
    info_bits = np.array([(info >> (15 - i)) & 1 for i in range(16)], np.uint8)
    checkword = framing._crc_bits(info_bits, poly=0x1B9, width=10, init=0, xorout=0x0FC)
    block = np.concatenate(
        [info_bits, [(checkword >> (9 - i)) & 1 for i in range(10)]]
    ).astype(np.uint8)
    out = framing.crc_rx(
        _frame_with(block),
        poly=0x1B9,
        bits=10,
        init=0,
        xorout=0x0FC,
        payload_bits=26,
    )
    assert out.frames[0].crc_ok is True
    assert out.frames[0].payload == framing.bits_to_bytes(info_bits)


def test_corrupt_sub_byte_checkword_fails() -> None:
    info_bits = np.zeros(16, np.uint8)
    block = np.concatenate([info_bits, np.ones(10, np.uint8)]).astype(np.uint8)
    out = framing.crc_rx(
        _frame_with(block), poly=0x1B9, bits=10, init=0, payload_bits=26
    )
    assert out.frames[0].crc_ok is False


def test_sub_byte_width_requires_payload_bits() -> None:
    issues: list = []
    validate_params("crc[0]", Crc._Params, {"poly": 0x1B9, "bits": 10}, issues)
    assert issues and "payload_bits" in issues[0].message


@pytest.mark.parametrize(
    "field, value, needle",
    [
        ("reflected", True, "reflected"),
        ("fold_tail", 1, "fold_tail"),
        ("checksum_le", True, "checksum_le"),
        ("bit_order", "lsb", "msb"),
    ],
)
def test_sub_byte_width_rejects_unsupported_options(
    field: str, value: object, needle: str
) -> None:
    # The bit-domain engine is MSB-first only this slice: a sub-byte width paired
    # with reflected/fold_tail/checksum_le or a non-msb bit_order must be rejected
    # with a message naming the offending constraint.
    issues: list = []
    validate_params(
        "crc[0]",
        Crc._Params,
        {"poly": 0x1B9, "bits": 10, "payload_bits": 26, field: value},
        issues,
    )
    assert issues and needle in issues[0].message


def test_byte_widths_reject_payload_bits() -> None:
    issues: list = []
    validate_params(
        "crc[0]",
        Crc._Params,
        {"poly": 0x1021, "bits": 16, "payload_bits": 26},
        issues,
    )
    assert issues
