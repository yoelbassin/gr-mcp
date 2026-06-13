"""Drift guard: the curated vocabulary, the GNU Radio factory table, and the
.grc exporter must enumerate exactly the same block types. Adding a block in one
place without the others should fail here, not surface at runtime — this keeps
the v1.1 build-out (new block types) honest across all three sites."""

import pytest

from marconi.backends.gnuradio_backend import _factories
from marconi.models import BlockSpec
from marconi.ops.export_grc import _map_block
from marconi.vocabulary import VOCABULARY

_DUMMY: dict[type, object] = {float: 1.0, int: 1, str: "x", bool: False}


def _fill_params(block_type: str) -> dict:
    """Every declared param filled with a type-correct dummy value, so the
    exporter's direct p[...] lookups don't trip on a missing key."""
    return {p.name: _DUMMY[p.type] for p in VOCABULARY[block_type].params}


# Only the factory test needs GNU Radio: calling _factories() imports the
# gnuradio bindings (the module-level import above is lazy, so collection is
# safe). The .grc exporter (_map_block) is engine-agnostic, so its test below
# runs everywhere and stays unmarked.
@pytest.mark.gnuradio
def test_factories_cover_exactly_the_vocabulary() -> None:
    assert set(_factories(1e6)) == set(VOCABULARY)


def test_grc_exporter_covers_every_vocabulary_block() -> None:
    for block_type in VOCABULARY:
        b = BlockSpec(id="b", type=block_type, params=_fill_params(block_type))
        grc_id, params = _map_block(b, 1e6)
        assert isinstance(grc_id, str) and grc_id
        assert isinstance(params, dict)
