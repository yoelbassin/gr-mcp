from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage, Stage
from marconi.phy.compile_context import CompileContext


class _ResamplerParams(StageParams):
    interp: int
    decim: int


class _NoParams(StageParams):
    pass


class FakeResampler(DuplexStage[CompileContext]):
    name = "fake_resampler"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "general"
    params_model = _ResamplerParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "fake_resampler", interp=int(params["interp"]), decim=int(params["decim"])
        )

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "fake_resampler", interp=int(params["interp"]), decim=int(params["decim"])
        )

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        return int(params["interp"]) / int(params["decim"])


class FakeDemod(DuplexStage[CompileContext]):
    name = "fake_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "general"
    params_model = _NoParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("fake_demod", sps=b.sps)

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("fake_mod", sps=b.sps)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "s", in_desc.layout, in_desc.carrier)


class FakeDemap(DuplexStage[CompileContext]):
    name = "fake_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "general"
    params_model = _NoParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("fake_demap")

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("fake_map")

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


class FakeSoftDemap(DuplexStage[CompileContext]):
    name = "fake_soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "general"
    params_model = _NoParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("fake_soft_demap")

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("fake_soft_map")

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "f", in_desc.layout, Carrier.SOFT)


def fixture_registry() -> dict[str, Stage[CompileContext]]:
    stages: list[Stage[CompileContext]] = [
        FakeResampler(),
        FakeDemod(),
        FakeDemap(),
        FakeSoftDemap(),
    ]
    return {s.name: s for s in stages}
