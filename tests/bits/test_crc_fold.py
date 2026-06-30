import numpy as np

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, TxCarrier, _Frame

# V1 SF7 reference: 17-byte dewhitened LoRa payload (the last 2 bytes, f4b5,
# are the fold-XOR input, NOT the checksum).  The actual 2-byte CRC field is
# [0xff, 0x10] stored little-endian (= 0x10ff).
# Verify: CRC16(payload[:-2]) ^ 0xf4b5 == 0x10ff.
_SF7_PAYLOAD = bytes.fromhex("407777aa01804f0003ce876fda97d8f4b5")
_SF7_CRC_LE = bytes([0xFF, 0x10])


def test_fold_tail_crc_validates_reference_payload():
    # Full trailing frame = payload + CRC field (19 bytes).
    f = _Frame(start=0, cursor=0)
    f.payload = _SF7_PAYLOAD + _SF7_CRC_LE
    c = framing.crc_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), frames=[f]),
        poly=0x1021,
        bits=16,
        init=0,
        fold_tail=2,
        checksum_le=True,
    )
    assert c.frames[0].crc_ok is True


def test_fold_tail_crc_tx_roundtrip():
    tx = framing.crc_tx(
        TxCarrier([_SF7_PAYLOAD]),
        poly=0x1021,
        bits=16,
        init=0,
        fold_tail=2,
        checksum_le=True,
    )
    assert tx.items[0] == _SF7_PAYLOAD + _SF7_CRC_LE
