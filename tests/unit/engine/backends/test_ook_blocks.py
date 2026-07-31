import pytest

from marconi.engine.backends.gnuradio.blocks import GR_BLOCKS, _make_ctx

_KINDS = [
    ("complex_to_mag", {}),
    ("multiply_const_ff", {"value": 2.0}),
    ("add_const_ff", {"value": -1.0}),
    ("float_to_complex", {}),
]


@pytest.mark.parametrize("kind,params", _KINDS)
def test_ook_factory_constructs(kind: str, params: dict) -> None:
    assert GR_BLOCKS[kind](_make_ctx(4.0), params) is not None
