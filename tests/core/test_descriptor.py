from marconi.core.descriptor import Carrier, Descriptor, Layout
from marconi.core.levels import Level


def test_identical_descriptors_are_compatible() -> None:
    a = Descriptor(Level.BITS, "b")
    b = Descriptor(Level.BITS, "b")
    assert a.compatible_with(b)


def test_carrier_mismatch_is_incompatible() -> None:
    hard = Descriptor(Level.BITS, "b", carrier=Carrier.HARD)
    soft = Descriptor(Level.BITS, "b", carrier=Carrier.SOFT)
    assert not hard.compatible_with(soft)


def test_defaults_are_stream_and_hard() -> None:
    d = Descriptor(Level.IQ, "c")
    assert d.layout is Layout.STREAM
    assert d.carrier is Carrier.HARD


def test_descriptor_is_frozen_hashable() -> None:
    d = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
    assert hash(d) == hash(Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT))
