import numpy as np
import pytest

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, TxCarrier
from marconi.bits.registry import registry


def _bits(*byte_vals):
    return np.unpackbits(np.array(byte_vals, dtype=np.uint8))


def test_seeds_a_frame_after_each_sync():
    # sync 0x7A96, then a body byte, twice
    stream = np.concatenate([_bits(0x7A, 0x96, 0x11), _bits(0x7A, 0x96, 0x22)])
    c = framing.sync_word_rx(RxCarrier(bits=stream), sync="7a96")
    assert [f.cursor for f in c.frames] == [16, 40]  # right after each 16-bit sync


def test_exact_match_rejects_corrupted_sync():
    stream = _bits(0x7B, 0x96, 0x11)  # one bit flipped in the sync
    c = framing.sync_word_rx(RxCarrier(bits=stream), sync="7a96", max_errors=0)
    assert c.frames == []


def test_correlator_tolerates_errors_within_threshold():
    stream = _bits(0x7B, 0x96, 0x11)  # Hamming-1 from 0x7a96
    c = framing.sync_word_rx(RxCarrier(bits=stream), sync="7a96", max_errors=1)
    assert [f.cursor for f in c.frames] == [16]


def test_tx_prepends_sync_and_round_trips():
    body = framing.bytes_to_bits(bytes([0x11, 0x22]))
    tx = framing.sync_word_tx(TxCarrier([body]), sync="7a96")
    assert framing.bits_to_bytes(tx.items[0][:16]) == bytes([0x7A, 0x96])
    rx = framing.sync_word_rx(RxCarrier(bits=tx.items[0]), sync="7a96")
    assert [f.cursor for f in rx.frames] == [16]


def test_registered_and_declares_seeder():
    stage = registry()["sync_word"]
    assert stage.seeds_frames is True
    with pytest.raises(Exception):
        stage.params_model.model_validate({"sync": "zz"})  # not hex
