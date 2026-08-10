import pytest
from helpers._fixtures import (
    FakeDemapStep,
    FakeDemodStep,
    FakeResamplerStep,
    FakeSoftDemapStep,
    fixture_registry,
)

from marconi.engine.compile.compiler import (
    CompileError,
    _sink_kind,
    _source_kind,
    compile_modem,
)
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)


def _modem(*steps: Step, symbol_rate: float) -> Modem:
    return Modem(symbol_rate=symbol_rate, path=list(steps))


def _compile_rx(modem: Modem, sample_rate: float) -> GrPipeline:
    return compile_modem(
        modem,
        fixture_registry(),
        direction="rx",
        sample_rate=sample_rate,
        start=IQ,
        source_io={"path": "in.iq"},
        sink_io={"path": "out.bits"},
    )


def test_rx_chain_blocks_and_connections() -> None:
    modem = _modem(
        FakeResamplerStep(interp=3, decim=2),
        FakeDemodStep(),
        FakeDemapStep(),
        symbol_rate=2.0,
    )
    pipe = _compile_rx(modem, 12.0)
    assert [b.kind for b in pipe.blocks] == [
        "iq_file_source",
        "fake_resampler",
        "fake_demod",
        "fake_demap",
        "bits_file_sink",
    ]
    ids = pipe.block_ids
    edges = [(c.src_block, c.dst_block) for c in pipe.connections]
    assert edges == list(zip(ids, ids[1:]))


def test_rx_source_and_sink_kinds_from_descriptors() -> None:
    pipe = _compile_rx(
        _modem(FakeDemodStep(), FakeSoftDemapStep(), symbol_rate=1.0), 4.0
    )
    assert pipe.blocks[0].kind == "iq_file_source"
    assert pipe.blocks[-1].kind == "soft_bits_file_sink"


def test_io_params_attach_to_source_and_sink() -> None:
    pipe = _compile_rx(_modem(FakeDemodStep(), FakeDemapStep(), symbol_rate=2.0), 8.0)
    assert pipe.blocks[0].params["path"] == "in.iq"
    assert pipe.blocks[-1].params["path"] == "out.bits"


def test_rate_folds_through_resampler_into_demod_sps() -> None:
    modem = _modem(
        FakeResamplerStep(interp=3, decim=2),
        FakeDemodStep(),
        symbol_rate=2.0,
    )
    pipe = _compile_rx(modem, 12.0)
    demod = next(b for b in pipe.blocks if b.kind == "fake_demod")
    assert demod.params["sps"] == 12.0 * 3 / 2 / 2.0  # 9.0


def test_fractional_sps_is_allowed() -> None:
    pipe = _compile_rx(_modem(FakeDemodStep(), symbol_rate=4.0), 10.0)
    demod = next(b for b in pipe.blocks if b.kind == "fake_demod")
    assert demod.params["sps"] == 2.5


def test_pipeline_sample_rate_metadata() -> None:
    pipe = _compile_rx(_modem(FakeDemodStep(), FakeDemapStep(), symbol_rate=2.0), 8.0)
    assert pipe.sample_rate == 8.0


def test_unknown_stage_raises_compile_error() -> None:
    with pytest.raises(CompileError):
        _compile_rx(_modem(Step(conv="nope"), symbol_rate=1.0), 4.0)


def test_bad_direction_raises_compile_error() -> None:
    with pytest.raises(CompileError):
        compile_modem(
            _modem(FakeDemodStep(), symbol_rate=1.0),
            fixture_registry(),
            # the runtime guard is the subject under test
            direction="sideways",  # type: ignore[arg-type]
            sample_rate=4.0,
            start=IQ,
            source_io={},
            sink_io={},
        )


def test_step_carries_conv_with_no_extra_params() -> None:
    step = FakeDemodStep()
    assert step.conv == "fake_demod" and step.model_dump(exclude={"conv"}) == {}


def test_complex_soft_symbols_route_to_iq_blocks() -> None:
    d = Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT)
    assert _source_kind(d) == "iq_file_source"
    assert _sink_kind(d) == "iq_file_sink"
