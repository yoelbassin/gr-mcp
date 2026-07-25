from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.params import ParamValue

IQ = Descriptor(Level.IQ, "c")
_DECHIRP_PARAMS: dict[str, ParamValue] = {"sf": 7, "oversample": 2, "zero_pad": 4}


def test_psk_demod_pins_order() -> None:
    out = stage_registry()["psk_demod"].out_descriptor(IQ, {"order": 4})
    assert out.order == 4


def test_qam_demod_pins_order() -> None:
    out = stage_registry()["qam_demod"].out_descriptor(IQ, {"order": 16})
    assert out.order == 16


def test_dechirp_pins_order() -> None:
    out = stage_registry()["dechirp"].out_descriptor(IQ, _DECHIRP_PARAMS)
    assert out.order == 128


def test_level_preserving_default_propagates_order() -> None:
    pinned = Descriptor(Level.SYMBOLS, "c", Carrier.SOFT, order=4)
    pre = {"preamble_i": [1.0, -1.0], "preamble_q": [0.0, 0.0]}
    out = stage_registry()["preamble_sync"].out_descriptor(pinned, pre)
    assert out.order == 4


def test_level_change_resets_order() -> None:
    pinned = Descriptor(Level.SYMBOLS, "c", Carrier.SOFT, order=4)
    out = stage_registry()["psk_demap"].out_descriptor(pinned, {"order": 4})
    assert out.order is None
