from __future__ import annotations

import numpy as np
from helpers import bitops, crc

# V1 SF7 reference: 17-byte dewhitened LoRa payload (the last 2 bytes, f4b5,
# are the fold-XOR input, NOT the checksum).  The actual 2-byte CRC field is
# [0xff, 0x10] stored little-endian (= 0x10ff).
# Verify: CRC16(payload[:-2]) ^ 0xf4b5 == 0x10ff.
_SF7_PAYLOAD = bytes.fromhex("407777aa01804f0003ce876fda97d8f4b5")
_SF7_CRC_LE = bytes([0xFF, 0x10])


def test_crc_check_strips_checksum_and_validates() -> None:
    body = b"123456789"
    framed = crc.crc_append(body, poly=0x1021, bits=16, init=0xFFFF)
    ok, out = crc.crc_check(framed, poly=0x1021, bits=16, init=0xFFFF)
    assert ok and out == body


def test_fold_tail_crc_validates_reference_payload() -> None:
    ok, body = crc.crc_check(
        _SF7_PAYLOAD + _SF7_CRC_LE,
        poly=0x1021,
        bits=16,
        init=0,
        fold_tail=2,
        checksum_le=True,
    )
    assert ok is True
    assert body == _SF7_PAYLOAD


def test_fold_tail_crc_tx_roundtrip() -> None:
    framed = crc.crc_append(
        _SF7_PAYLOAD, poly=0x1021, bits=16, init=0, fold_tail=2, checksum_le=True
    )
    assert framed == _SF7_PAYLOAD + _SF7_CRC_LE


def test_crc10_atm_catalog_vector() -> None:
    # CRC-10/ATM from the canonical CRC catalog (reveng): poly=0x233, init=0,
    # xorout=0, check("123456789") = 0x199 — the external anchor for the
    # bit-domain engine.
    bits = bitops.bytes_to_bits(b"123456789")
    assert crc._crc_bits(bits, poly=0x233, width=10, init=0, xorout=0) == 0x199


def test_rds_style_10bit_checkword_validates() -> None:
    # RDS (EN 50067) block: 16 info bits + 10-bit checkword where checkword =
    # crc10(info) XOR offset word A (0x0FC), generator 0x1B9. The engine is
    # anchored by the catalog vector above; this exercises the sub-byte flow
    # (payload_bits, xorout-as-offset) end to end.
    info = 0x3AB0
    info_bits = np.array([(info >> (15 - i)) & 1 for i in range(16)], np.uint8)
    checkword = crc._crc_bits(info_bits, poly=0x1B9, width=10, init=0, xorout=0x0FC)
    block = np.concatenate(
        [info_bits, [(checkword >> (9 - i)) & 1 for i in range(10)]]
    ).astype(np.uint8)
    ok, body = crc.crc_check_bits(block, poly=0x1B9, width=10, init=0, xorout=0x0FC)
    assert ok is True
    assert np.array_equal(body, info_bits)


def test_corrupt_sub_byte_checkword_fails() -> None:
    info_bits = np.zeros(16, np.uint8)
    block = np.concatenate([info_bits, np.ones(10, np.uint8)]).astype(np.uint8)
    ok, _ = crc.crc_check_bits(block, poly=0x1B9, width=10, init=0)
    assert ok is False


def test_crc_x25_and_ccitt_vectors() -> None:
    x25 = crc.crc_value(
        b"123456789", poly=0x1021, bits=16, init=0xFFFF, reflected=True, xorout=0xFFFF
    )
    assert x25 == 0x906E
    ccitt = crc.crc_value(
        b"123456789", poly=0x1021, bits=16, init=0xFFFF, reflected=False, xorout=0
    )
    assert ccitt == 0x29B1
