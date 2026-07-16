from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage, RxStage, Stage
from marconi.phy.compile_context import CompileContext


class _SliceParams(StageParams):
    pass


class Slice(DuplexStage[CompileContext]):
    """Binary symbol<->bit decision. RX: hard-slice soft floats to bits.
    TX: map bits to antipodal +/-1.0 symbols."""

    name = "slice"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "general"
    params_model = _SliceParams
    accepts_item_type = "f"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("binary_slicer")

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("chunks_to_symbols", symbols=[-1.0, 1.0])

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


class _MsliceParams(StageParams):
    thresholds: list[float]
    code: list[int]
    bits: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def _ok(self) -> "_MsliceParams":
        if len(self.code) != len(self.thresholds) + 1:
            raise PydanticCustomError("value_error", "code must have len(thresholds)+1")
        if list(self.thresholds) != sorted(self.thresholds):
            raise PydanticCustomError("value_error", "thresholds must be ascending")
        return self


class Mslice(RxStage[CompileContext]):
    """Generic M-ary soft-slice, SYMBOLS(soft)->BITS(hard). Each soft float lands
    in an amplitude region by `thresholds`; the region's `code` value is emitted
    MSB-first as `bits` bits — a caller-parameterized M-ary demap with an
    integrated symbol map."""

    name = "mslice"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "general"
    params_model = _MsliceParams
    accepts_item_type = "f"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _MsliceParams.model_validate(dict(params))
        b.chain(
            "mslice",
            thresholds=[float(x) for x in p.thresholds],
            code=[int(x) for x in p.code],
            bits=p.bits,
        )

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


GENERAL_STAGES: tuple[type[Stage[CompileContext]], ...] = (Slice, Mslice)
