import pytest

from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level


def test_defaults_are_hard_unknown() -> None:
    d = Descriptor(Level.IQ, ItemType.C)
    assert d.carrier is Carrier.HARD


def test_descriptor_is_frozen_hashable() -> None:
    d = Descriptor(Level.SYMBOLS, ItemType.F, carrier=Carrier.SOFT)
    assert hash(d) == hash(Descriptor(Level.SYMBOLS, ItemType.F, carrier=Carrier.SOFT))


def test_order_defaults_none() -> None:
    assert Descriptor(Level.IQ, ItemType.C).order is None


def test_order_allowed_at_symbols() -> None:
    d = Descriptor(Level.SYMBOLS, ItemType.S, order=128)
    assert d.order == 128


def test_order_rejected_off_symbols() -> None:
    with pytest.raises(ValueError):
        Descriptor(Level.IQ, ItemType.C, order=4)
    with pytest.raises(ValueError):
        Descriptor(Level.BITS, ItemType.B, Carrier.HARD, Amplitude.UNKNOWN, 4)


def test_order_lower_bound() -> None:
    with pytest.raises(ValueError):
        Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT, order=1)


def test_iq_level_is_complex_by_construction() -> None:
    # 0 of 23 IQ stages declared accepts_item_type, so validate_modem blessed
    # input_item_type='f'/'b' on complex-only chains and 23/23 died in the
    # worker on a raw GR itemsize mismatch. IQ means complex baseband; a real
    # (float) stream is AUDIO-level until `analytic` lifts it.
    import pytest

    from marconi.engine.types.descriptor import Descriptor
    from marconi.engine.types.enums import ItemType
    from marconi.engine.types.levels import Level

    with pytest.raises(ValueError, match="analytic"):
        Descriptor(Level.IQ, ItemType.F)
    with pytest.raises(ValueError, match="complex"):
        Descriptor(Level.IQ, ItemType.B)
    assert Descriptor(Level.IQ, ItemType.C).item_type is ItemType.C
    assert Descriptor(Level.AUDIO, ItemType.F).item_type is ItemType.F
