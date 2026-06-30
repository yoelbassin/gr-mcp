import numpy as np

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, TxCarrier


def test_nibble_swap_swaps_per_byte_and_roundtrips():
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8)  # 1011 0001
    out = framing.nibble_swap_rx(RxCarrier(bits=bits)).bits
    assert list(out) == [0, 0, 0, 1, 1, 0, 1, 1]  # 0001 1011
    tx = framing.nibble_swap_tx(TxCarrier([out]))
    assert list(tx.items[0]) == list(bits)
