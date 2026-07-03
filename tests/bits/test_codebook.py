import numpy as np
import pytest

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, TxCarrier
from marconi.bits.registry import registry

# 3-of-6 line code (one generic op expresses it; the table is caller data).
_3OF6 = [
    0x16, 0x0D, 0x0E, 0x0B, 0x1C, 0x19, 0x1A, 0x13,
    0x2C, 0x25, 0x26, 0x23, 0x34, 0x31, 0x32, 0x29,
]  # fmt: skip
_MANCHESTER = [0b01, 0b10]  # the same op, different table/widths


def test_manchester_round_trips():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    tx = framing.codebook_tx(
        TxCarrier([bits]), code_bits=2, data_bits=1, table=_MANCHESTER
    )
    chips = tx.items[0]
    # 1->10, 0->01, ...
    assert chips.tolist() == [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
    back = framing.codebook_rx(
        RxCarrier(bits=chips), code_bits=2, data_bits=1, table=_MANCHESTER
    )
    assert back.bits.tolist() == bits.tolist()


def test_three_of_six_round_trips_bytes():
    data = bytes([0x44, 0x2D, 0x2C, 0x11])
    bits = np.unpackbits(np.frombuffer(data, np.uint8))
    tx = framing.codebook_tx(TxCarrier([bits]), code_bits=6, data_bits=4, table=_3OF6)
    chips = tx.items[0]
    assert chips.size == data.__len__() * 2 * 6  # two 6-chip codewords per byte
    # every emitted 6-chip group carries exactly three ones (the code's invariant)
    assert set(int(g.sum()) for g in chips.reshape(-1, 6)) == {3}
    back = framing.codebook_rx(
        RxCarrier(bits=chips), code_bits=6, data_bits=4, table=_3OF6
    )
    assert np.packbits(back.bits).tobytes() == data


def test_unknown_code_symbol_maps_to_zero():
    # 111111 is not a valid 3-of-6 codeword; decoder degrades to 0, no crash
    chips = np.ones(6, dtype=np.uint8)
    back = framing.codebook_rx(
        RxCarrier(bits=chips), code_bits=6, data_bits=4, table=_3OF6
    )
    assert back.bits.tolist() == [0, 0, 0, 0]


def test_codebook_registered_and_validates_table_size():
    stage = registry()["codebook"]
    stage.params_model.model_validate({"code_bits": 6, "data_bits": 4, "table": _3OF6})
    with pytest.raises(Exception):  # 3 entries != 2^4
        stage.params_model.model_validate(
            {"code_bits": 6, "data_bits": 4, "table": [1, 2, 3]}
        )
