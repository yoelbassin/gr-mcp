import pytest
from engine._fixtures import fixture_registry, stub_factories

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.stub import StubBackend
from marconi.engine.compiler import compile_modem
from marconi.engine.descriptor import Descriptor
from marconi.engine.ir import GrBlock, GrConnection, GrPipeline
from marconi.engine.levels import Level
from marconi.engine.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")


def test_instantiate_resolves_a_compiled_pipeline() -> None:
    modem = ModemSpec(
        symbol_rate=2.0,
        path=[ModemStep(conv="fake_demod"), ModemStep(conv="fake_demap")],
    )
    pipe = compile_modem(
        modem,
        fixture_registry(),
        direction="rx",
        sample_rate=8.0,
        start=IQ,
        source_io={"path": "i"},
        sink_io={"path": "o"},
    )
    graph = StubBackend(stub_factories()).instantiate(pipe)
    assert set(graph.blocks) == set(pipe.block_ids)
    assert len(graph.edges) == len(pipe.connections)


def test_unknown_kind_raises() -> None:
    pipe = GrPipeline(sample_rate=1.0, blocks=[GrBlock(id="a", kind="mystery")])
    with pytest.raises(BackendError):
        StubBackend(stub_factories()).instantiate(pipe)


def test_connection_to_unknown_block_raises() -> None:
    pipe = GrPipeline(
        sample_rate=1.0,
        blocks=[GrBlock(id="a", kind="iq_file_source")],
        connections=[GrConnection(src_block="a", dst_block="ghost")],
    )
    with pytest.raises(BackendError):
        StubBackend(stub_factories()).instantiate(pipe)


def test_out_of_range_port_raises() -> None:
    pipe = GrPipeline(
        sample_rate=1.0,
        blocks=[
            GrBlock(id="a", kind="iq_file_source"),
            GrBlock(id="b", kind="iq_file_sink"),
        ],
        connections=[GrConnection(src_block="a", src_port=2, dst_block="b")],
    )
    with pytest.raises(BackendError):
        StubBackend(stub_factories()).instantiate(pipe)


def test_backend_name() -> None:
    assert StubBackend(stub_factories()).name == "stub"
