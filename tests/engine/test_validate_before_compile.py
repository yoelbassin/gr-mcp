import pytest
from engine._fixtures import FakeSoftFec, RxOnlyDemod, fixture_registry

from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")


def _modem(*steps: tuple[str, dict], symbol_rate: float) -> ModemSpec:
    return ModemSpec(
        symbol_rate=symbol_rate,
        path=[ModemStep(conv=c, params=p) for c, p in steps],
    )


def _compile(modem: ModemSpec, direction: str, registry=None):
    return compile_modem(
        modem,
        registry or fixture_registry(),
        direction=direction,
        sample_rate=8.0,
        start=IQ,
        source_io={"path": "a"},
        sink_io={"path": "b"},
    )


def test_tx_over_rx_only_stage_raises_before_emit() -> None:
    reg = {**fixture_registry(), "rx_only_demod": RxOnlyDemod()}
    m = _modem(("rx_only_demod", {}), ("fake_demap", {}), symbol_rate=2.0)
    with pytest.raises(CompileError) as e:
        _compile(m, "tx", reg)
    assert "direction" in str(e.value)


def test_missing_param_names_the_field() -> None:
    # fake_resampler requires interp/decim; a missing one must name the field in
    # the error, not just "Field required" with no location (issue 07).
    with pytest.raises(CompileError) as e:
        _compile(_modem(("fake_resampler", {"decim": 1}), symbol_rate=2.0), "rx")
    assert "interp" in str(e.value)


def test_non_adjacent_path_raises() -> None:
    # fake_demap is SYMBOLS->BITS; from an IQ start it is not a valid first step.
    with pytest.raises(CompileError):
        _compile(_modem(("fake_demap", {}), symbol_rate=2.0), "rx")


def test_unknown_stage_raises() -> None:
    with pytest.raises(CompileError):
        _compile(_modem(("nope", {}), symbol_rate=2.0), "rx")


def test_valid_rx_path_still_compiles() -> None:
    pipe = _compile(
        _modem(("fake_demod", {}), ("fake_demap", {}), symbol_rate=2.0), "rx"
    )
    assert pipe.blocks[0].kind == "iq_file_source"


def test_hard_bits_into_soft_consumer_raises_naming_both() -> None:
    # fake_demap emits BITS "b"/HARD; fake_soft_fec accepts "f"/SOFT -> ill-typed
    # composition, must fail at compile (not in the worker) naming both stages.
    reg = {**fixture_registry(), "fake_soft_fec": FakeSoftFec()}
    m = _modem(
        ("fake_demod", {}), ("fake_demap", {}), ("fake_soft_fec", {}), symbol_rate=2.0
    )
    with pytest.raises(CompileError) as e:
        _compile(m, "rx", reg)
    msg = str(e.value)
    assert "fake_soft_fec" in msg and "fake_demap" in msg


def test_valid_soft_lane_compiles() -> None:
    reg = {**fixture_registry(), "fake_soft_fec": FakeSoftFec()}
    m = _modem(
        ("fake_demod", {}),
        ("fake_soft_demap", {}),
        ("fake_soft_fec", {}),
        symbol_rate=2.0,
    )
    pipe = _compile(m, "rx", reg)
    assert any(b.kind == "bits_file_sink" for b in pipe.blocks)
