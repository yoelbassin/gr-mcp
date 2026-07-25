from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marconi.engine.backends.stub import BlockArity
from marconi.engine.compile_context import CompileContext
from marconi.engine.descriptor import Carrier, Descriptor
from marconi.engine.levels import Level
from marconi.engine.params import StageParams
from marconi.engine.stage import DuplexStage, RxStage, Stage


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


class FakeSoftFec(RxStage[CompileContext]):
    name = "fake_soft_fec"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "general"
    params_model = _NoParams
    accepts_item_type = "f"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("fake_soft_fec")

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


class RxOnlyDemod(RxStage[CompileContext]):
    name = "rx_only_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "general"
    params_model = _NoParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("fake_demod", sps=b.sps)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "s", in_desc.layout, in_desc.carrier)


def fixture_registry() -> dict[str, Stage[CompileContext]]:
    stages: list[Stage[CompileContext]] = [
        FakeResampler(),
        FakeDemod(),
        FakeDemap(),
        FakeSoftDemap(),
    ]
    return {s.name: s for s in stages}


def stub_factories() -> dict[str, BlockArity]:
    return {
        "iq_file_source": BlockArity(0, 1),
        "iq_file_sink": BlockArity(1, 0),
        "bits_file_source": BlockArity(0, 1),
        "bits_file_sink": BlockArity(1, 0),
        "soft_bits_file_sink": BlockArity(1, 0),
        "symbols_file_sink": BlockArity(1, 0),
        "fake_resampler": BlockArity(1, 1),
        "fake_demod": BlockArity(1, 1),
        "fake_mod": BlockArity(1, 1),
        "fake_demap": BlockArity(1, 1),
        "fake_map": BlockArity(1, 1),
        "fake_soft_demap": BlockArity(1, 1),
        "fake_soft_map": BlockArity(1, 1),
        "fake_soft_fec": BlockArity(1, 1),
    }
