import pytest
from engine._fixtures import fixture_registry

from marconi.engine.compiler import CompileError, compile_modem
from marconi.engine.descriptor import Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")


def _modem(*steps: tuple[str, dict], symbol_rate: float) -> ModemSpec:
    return ModemSpec(
        symbol_rate=symbol_rate,
        path=[ModemStep(conv=conv, params=params) for conv, params in steps],
    )


def _compile_rx(modem: ModemSpec, sample_rate: float):
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
        ("fake_resampler", {"interp": 3, "decim": 2}),
        ("fake_demod", {}),
        ("fake_demap", {}),
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
        _modem(("fake_demod", {}), ("fake_soft_demap", {}), symbol_rate=1.0), 4.0
    )
    assert pipe.blocks[0].kind == "iq_file_source"
    assert pipe.blocks[-1].kind == "soft_bits_file_sink"


def test_io_params_attach_to_source_and_sink() -> None:
    pipe = _compile_rx(
        _modem(("fake_demod", {}), ("fake_demap", {}), symbol_rate=2.0), 8.0
    )
    assert pipe.blocks[0].params["path"] == "in.iq"
    assert pipe.blocks[-1].params["path"] == "out.bits"


def test_rate_folds_through_resampler_into_demod_sps() -> None:
    modem = _modem(
        ("fake_resampler", {"interp": 3, "decim": 2}),
        ("fake_demod", {}),
        symbol_rate=2.0,
    )
    pipe = _compile_rx(modem, 12.0)
    demod = next(b for b in pipe.blocks if b.kind == "fake_demod")
    assert demod.params["sps"] == 12.0 * 3 / 2 / 2.0  # 9.0


def test_fractional_sps_is_allowed() -> None:
    pipe = _compile_rx(_modem(("fake_demod", {}), symbol_rate=4.0), 10.0)
    demod = next(b for b in pipe.blocks if b.kind == "fake_demod")
    assert demod.params["sps"] == 2.5


def test_pipeline_sample_rate_metadata() -> None:
    pipe = _compile_rx(
        _modem(("fake_demod", {}), ("fake_demap", {}), symbol_rate=2.0), 8.0
    )
    assert pipe.sample_rate == 8.0


def test_unknown_stage_raises_compile_error() -> None:
    with pytest.raises(CompileError):
        _compile_rx(_modem(("nope", {}), symbol_rate=1.0), 4.0)


def test_bad_direction_raises_compile_error() -> None:
    with pytest.raises(CompileError):
        compile_modem(
            _modem(("fake_demod", {}), symbol_rate=1.0),
            fixture_registry(),
            direction="sideways",
            sample_rate=4.0,
            start=IQ,
            source_io={},
            sink_io={},
        )


def test_modemstep_satisfies_specstep_protocol() -> None:
    from marconi.engine.stage import SpecStep

    step: SpecStep = ModemStep(conv="fake_demod")
    assert step.conv == "fake_demod" and dict(step.params) == {}
