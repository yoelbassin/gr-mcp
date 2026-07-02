from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import RxStage, Stage
from marconi.phy.compile_context import CompileContext


class _DeinterleaveParams(StageParams):
    perm: list[int]


class Deinterleave(RxStage[CompileContext]):
    """Generic block permute, BITS->BITS (carrier passes through). Stock
    blockinterleaver_ff: out[t]=in[perm[t]] per block of len(perm)."""

    name = "deinterleave"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    params_model = _DeinterleaveParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("blockinterleaver_ff", perm=[int(x) for x in params["perm"]], mode=True)


class _DepunctureParams(StageParams):
    keep_mask: list[int]


class Depuncture(RxStage[CompileContext]):
    """Generic depuncture, BITS->BITS soft. Scatters soft into a wider codeword per
    a keep-mask (0 = erasure). Protocol puncture tables are params."""

    name = "depuncture"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    params_model = _DepunctureParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("depuncture", keep_mask=[int(x) for x in params["keep_mask"]])


class _ConvParams(StageParams):
    scheme: str
    rate_inv: StrictInt = Field(ge=1)
    polys: list[int] = Field(min_length=1)
    frame_bits: StrictInt = Field(ge=1)
    tail: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _ok(self) -> "_ConvParams":
        if self.scheme not in _FEC_EMIT:
            raise PydanticCustomError(
                "value_error",
                "unknown fec scheme '{scheme}'; known: {known}",
                {"scheme": self.scheme, "known": ", ".join(sorted(_FEC_EMIT))},
            )
        return self


def _emit_cc(b: CompileContext, p: Mapping[str, Any]) -> None:
    fb, t = int(p["frame_bits"]), int(p.get("tail", 0))
    b.chain(
        "trellis_viterbi",
        rate_inv=int(p["rate_inv"]),
        polys=[int(x) for x in p["polys"]],
        frame_bits=fb,
        tail=t,
    )
    b.chain("keep_m_in_n_b", m=fb, n=fb + t)


_FEC_EMIT: dict[str, Callable[[CompileContext, Mapping[str, Any]], None]] = {
    "cc": _emit_cc
}


class Fec(RxStage[CompileContext]):
    """Generic FEC decode, BITS soft -> BITS hard. `scheme` selects the decoder via
    a factory map (no if/else); `cc` wires gnuradio.trellis. The soft->hard boundary
    is here."""

    name = "fec"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    params_model = _ConvParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        _FEC_EMIT[str(params["scheme"])](b, params)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


CODING_STAGES: tuple[type[Stage[CompileContext]], ...] = (Deinterleave, Depuncture, Fec)
