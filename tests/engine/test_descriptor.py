import pytest

from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.levels import Level


def test_defaults_are_hard_unknown() -> None:
    d = Descriptor(Level.IQ, "c")
    assert d.carrier is Carrier.HARD


def test_descriptor_is_frozen_hashable() -> None:
    d = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
    assert hash(d) == hash(Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT))


def test_order_defaults_none() -> None:
    assert Descriptor(Level.IQ, "c").order is None


def test_order_allowed_at_symbols() -> None:
    d = Descriptor(Level.SYMBOLS, "s", order=128)
    assert d.order == 128


def test_order_rejected_off_symbols() -> None:
    with pytest.raises(ValueError):
        Descriptor(Level.IQ, "c", order=4)
    with pytest.raises(ValueError):
        Descriptor(Level.BITS, "b", Carrier.HARD, Amplitude.UNKNOWN, 4)


def test_order_lower_bound() -> None:
    with pytest.raises(ValueError):
        Descriptor(Level.SYMBOLS, "c", carrier=Carrier.SOFT, order=1)
