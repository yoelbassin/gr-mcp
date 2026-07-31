from marconi.engine.compile.compiler import _sink_kind, _source_kind
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level


def test_complex_soft_symbols_route_to_iq_blocks() -> None:
    d = Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT)
    assert _source_kind(d) == "iq_file_source"
    assert _sink_kind(d) == "iq_file_sink"
