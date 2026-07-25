from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level


def test_defaults_are_hard_unknown() -> None:
    d = Descriptor(Level.IQ, "c")
    assert d.carrier is Carrier.HARD


def test_descriptor_is_frozen_hashable() -> None:
    d = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
    assert hash(d) == hash(Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT))
