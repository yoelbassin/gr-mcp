from marconi.engine.types.descriptor import Carrier, Descriptor, Layout
from marconi.engine.types.levels import Level


def test_defaults_are_stream_and_hard() -> None:
    d = Descriptor(Level.IQ, "c")
    assert d.layout is Layout.STREAM
    assert d.carrier is Carrier.HARD


def test_descriptor_is_frozen_hashable() -> None:
    d = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
    assert hash(d) == hash(Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT))
