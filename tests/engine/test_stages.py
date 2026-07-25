# tests/core/test_stages.py
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import StrictInt

from marconi.engine.descriptor import Carrier, Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import ValidationIssue
from marconi.engine.params import StageParams
from marconi.engine.stage import (
    DuplexStage,
    RxStage,
    Stage,
    StageDirectionError,
    validate_path,
)


class _FakeBuilder:
    def __init__(self) -> None:
        self.ops: list[str] = []


class Demod(DuplexStage["_FakeBuilder"]):
    name = "demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "fsk"

    class Params(StageParams):
        sps: StrictInt

    params_model = Params

    def emit_rx(self, b: "_FakeBuilder", p: Mapping[str, Any]) -> None:
        b.ops.append("rx")

    def emit_tx(self, b: "_FakeBuilder", p: Mapping[str, Any]) -> None:
        b.ops.append("tx")

    def out_descriptor(self, d: Descriptor, p: Mapping[str, Any]) -> Descriptor:
        return Descriptor(self.to_level, "f", carrier=Carrier.SOFT)


class Resample(DuplexStage["_FakeBuilder"]):
    name = "resample"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "general"

    class Params(StageParams):
        interpolation: StrictInt
        decimation: StrictInt

    params_model = Params

    def emit_rx(self, b: "_FakeBuilder", p: Mapping[str, Any]) -> None: ...
    def emit_tx(self, b: "_FakeBuilder", p: Mapping[str, Any]) -> None: ...

    def rate_factor(self, p: Mapping[str, Any]) -> float:
        return int(p["interpolation"]) / int(p["decimation"])


class Sink(RxStage["_FakeBuilder"]):
    name = "css_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "css"

    class Params(StageParams):
        sf: StrictInt

    params_model = Params

    def emit_rx(self, b: "_FakeBuilder", p: Mapping[str, Any]) -> None: ...


class Fec(DuplexStage["_FakeBuilder"]):
    name = "fec"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"

    class Params(StageParams):
        pass

    params_model = Params

    def emit_rx(self, b: "_FakeBuilder", p: Mapping[str, Any]) -> None: ...
    def emit_tx(self, b: "_FakeBuilder", p: Mapping[str, Any]) -> None: ...


class _Step:
    def __init__(self, conv: str, **params: object) -> None:
        self.conv = conv
        self.params = params


def _registry() -> dict[str, Stage[Any]]:
    return {s.name: s for s in (Demod(), Resample(), Sink())}


def test_stage_declares_descriptor_and_rate_transforms() -> None:
    out = Demod().out_descriptor(Descriptor(Level.IQ, "c"), {"sps": 4})
    assert out == Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
    assert Resample().rate_factor({"interpolation": 3, "decimation": 2}) == 1.5
    assert Demod().rate_factor({"sps": 4}) == 1.0  # default identity


def test_rx_only_stage_rejects_tx() -> None:
    with pytest.raises(StageDirectionError):
        Sink().emit_tx(_FakeBuilder(), {"sf": 7})


def test_valid_path_passes() -> None:
    issues: list[ValidationIssue] = []
    validate_path(
        [_Step("demod", sps=4), _Step("css_demap", sf=7)],
        _registry(),
        Level.IQ,
        "modem",
        issues,
    )
    assert issues == []


def test_nonadjacent_path_is_flagged() -> None:
    issues: list[ValidationIssue] = []
    validate_path([_Step("css_demap", sf=7)], _registry(), Level.IQ, "modem", issues)
    assert any(
        "not adjacent" in i.message or "must start at" in i.message for i in issues
    )


def test_skipped_conversion_between_stages_is_flagged() -> None:
    """demod ends at SYMBOLS, fec starts at BITS: the missing SYMBOLS->BITS
    stage must fail validation. The old rung-successor allowance let this
    compile — soft symbols punning as soft bits decoded garbage."""
    issues: list[ValidationIssue] = []
    validate_path(
        [_Step("demod", sps=4), _Step("fec")],
        {**_registry(), "fec": Fec()},
        Level.IQ,
        "modem",
        issues,
    )
    assert any("does not match" in i.message for i in issues), issues


def test_unknown_converter_is_flagged() -> None:
    issues: list[ValidationIssue] = []
    validate_path([_Step("nope")], _registry(), Level.IQ, "modem", issues)
    assert any("unknown converter" in i.message for i in issues)


def test_bad_params_are_flagged() -> None:
    issues: list[ValidationIssue] = []
    validate_path([_Step("demod", sps="x")], _registry(), Level.IQ, "modem", issues)
    assert any(i.block_id and i.block_id.startswith("demod") for i in issues)
