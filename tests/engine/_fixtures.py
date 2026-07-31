from __future__ import annotations

from typing import Any, Literal

from marconi.engine.backends.stub import BlockArity
from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, RxStage, Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class FakeResamplerStep(Step):
    conv: Literal["fake_resampler"] = "fake_resampler"
    interp: int
    decim: int


class FakeDemodStep(Step):
    conv: Literal["fake_demod"] = "fake_demod"


class FakeDemapStep(Step):
    conv: Literal["fake_demap"] = "fake_demap"


class FakeSoftDemapStep(Step):
    conv: Literal["fake_soft_demap"] = "fake_soft_demap"


class FakeSoftFecStep(Step):
    conv: Literal["fake_soft_fec"] = "fake_soft_fec"


class RxOnlyDemodStep(Step):
    conv: Literal["rx_only_demod"] = "rx_only_demod"


class FakeResampler(DuplexStage[CompileContext, FakeResamplerStep]):
    name = "fake_resampler"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "general"
    step_model = FakeResamplerStep

    def emit_rx(self, b: CompileContext, step: FakeResamplerStep) -> None:
        b.chain("fake_resampler", interp=step.interp, decim=step.decim)

    def emit_tx(self, b: CompileContext, step: FakeResamplerStep) -> None:
        b.chain("fake_resampler", interp=step.interp, decim=step.decim)

    def rate_factor(self, step: FakeResamplerStep) -> float:
        return step.interp / step.decim


class FakeDemod(DuplexStage[CompileContext, FakeDemodStep]):
    name = "fake_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "general"
    step_model = FakeDemodStep

    def emit_rx(self, b: CompileContext, step: FakeDemodStep) -> None:
        b.chain("fake_demod", sps=b.sps)

    def emit_tx(self, b: CompileContext, step: FakeDemodStep) -> None:
        b.chain("fake_mod", sps=b.sps)

    def out_descriptor(self, in_desc: Descriptor, step: FakeDemodStep) -> Descriptor:
        return Descriptor(Level.SYMBOLS, ItemType.S, in_desc.carrier)


class FakeDemap(DuplexStage[CompileContext, FakeDemapStep]):
    name = "fake_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "general"
    step_model = FakeDemapStep

    def emit_rx(self, b: CompileContext, step: FakeDemapStep) -> None:
        b.chain("fake_demap")

    def emit_tx(self, b: CompileContext, step: FakeDemapStep) -> None:
        b.chain("fake_map")

    def out_descriptor(self, in_desc: Descriptor, step: FakeDemapStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.B, Carrier.HARD)


class FakeSoftDemap(DuplexStage[CompileContext, FakeSoftDemapStep]):
    name = "fake_soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "general"
    step_model = FakeSoftDemapStep

    def emit_rx(self, b: CompileContext, step: FakeSoftDemapStep) -> None:
        b.chain("fake_soft_demap")

    def emit_tx(self, b: CompileContext, step: FakeSoftDemapStep) -> None:
        b.chain("fake_soft_map")

    def out_descriptor(
        self, in_desc: Descriptor, step: FakeSoftDemapStep
    ) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.F, Carrier.SOFT)


class FakeSoftFec(RxStage[CompileContext, FakeSoftFecStep]):
    name = "fake_soft_fec"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "general"
    step_model = FakeSoftFecStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: FakeSoftFecStep) -> None:
        b.chain("fake_soft_fec")

    def out_descriptor(self, in_desc: Descriptor, step: FakeSoftFecStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.B, Carrier.HARD)


class RxOnlyDemod(RxStage[CompileContext, RxOnlyDemodStep]):
    name = "rx_only_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "general"
    step_model = RxOnlyDemodStep

    def emit_rx(self, b: CompileContext, step: RxOnlyDemodStep) -> None:
        b.chain("fake_demod", sps=b.sps)

    def out_descriptor(self, in_desc: Descriptor, step: RxOnlyDemodStep) -> Descriptor:
        return Descriptor(Level.SYMBOLS, ItemType.S, in_desc.carrier)


def fixture_registry() -> dict[str, Stage[CompileContext, Any]]:
    stages: list[Stage[CompileContext, Any]] = [
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
