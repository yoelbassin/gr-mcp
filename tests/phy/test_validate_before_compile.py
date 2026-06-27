import pytest
from phy._fixtures import RxOnlyDemod, fixture_registry

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.compiler import CompileError, compile_modem
from marconi.phy.models import ModemSpec, ModemStep

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
