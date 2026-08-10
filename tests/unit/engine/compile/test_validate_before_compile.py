from typing import Any, Literal

import pytest
from helpers._fixtures import (
    FakeDemapStep,
    FakeDemodStep,
    FakeResamplerStep,
    FakeSoftDemapStep,
    FakeSoftFec,
    FakeSoftFecStep,
    RxOnlyDemod,
    RxOnlyDemodStep,
    fixture_registry,
)
from pydantic import ValidationError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.stages.base import Stage
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)


def _modem(*steps: Step, symbol_rate: float) -> Modem:
    return Modem(symbol_rate=symbol_rate, path=list(steps))


def _compile(
    modem: Modem,
    direction: Literal["rx", "tx"],
    registry: dict[str, Stage[CompileContext, Any]] | None = None,
) -> GrPipeline:
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
    m = _modem(RxOnlyDemodStep(), FakeDemapStep(), symbol_rate=2.0)
    with pytest.raises(CompileError) as e:
        _compile(m, "tx", reg)
    assert "direction" in str(e.value)


def test_missing_param_names_the_field() -> None:
    # fake_resampler requires interp/decim; a missing one must name the field in
    # the error. Step construction now validates eagerly (Step migration moved
    # this from a compile-time check to pydantic construction), so the error is
    # a ValidationError raised while building the step, not a CompileError.
    with pytest.raises(ValidationError) as e:
        FakeResamplerStep(decim=1)
    assert "interp" in str(e.value)


def test_non_adjacent_path_raises() -> None:
    # fake_demap is SYMBOLS->BITS; from an IQ start it is not a valid first step.
    with pytest.raises(CompileError):
        _compile(_modem(FakeDemapStep(), symbol_rate=2.0), "rx")


def test_unknown_stage_raises() -> None:
    with pytest.raises(CompileError):
        _compile(_modem(Step(conv="nope"), symbol_rate=2.0), "rx")


def test_valid_rx_path_still_compiles() -> None:
    pipe = _compile(_modem(FakeDemodStep(), FakeDemapStep(), symbol_rate=2.0), "rx")
    assert pipe.blocks[0].kind == "iq_file_source"


def test_hard_bits_into_soft_consumer_raises_naming_both() -> None:
    # fake_demap emits BITS "b"/HARD; fake_soft_fec accepts "f"/SOFT -> ill-typed
    # composition, must fail at compile (not in the worker) naming both stages.
    reg = {**fixture_registry(), "fake_soft_fec": FakeSoftFec()}
    m = _modem(FakeDemodStep(), FakeDemapStep(), FakeSoftFecStep(), symbol_rate=2.0)
    with pytest.raises(CompileError) as e:
        _compile(m, "rx", reg)
    msg = str(e.value)
    assert "fake_soft_fec" in msg and "fake_demap" in msg


def test_valid_soft_lane_compiles() -> None:
    reg = {**fixture_registry(), "fake_soft_fec": FakeSoftFec()}
    m = _modem(
        FakeDemodStep(),
        FakeSoftDemapStep(),
        FakeSoftFecStep(),
        symbol_rate=2.0,
    )
    pipe = _compile(m, "rx", reg)
    assert any(b.kind == "bits_file_sink" for b in pipe.blocks)
