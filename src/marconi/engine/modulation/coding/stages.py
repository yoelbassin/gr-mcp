from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile_context import CompileContext
from marconi.engine.descriptor import Carrier, Descriptor
from marconi.engine.levels import Level
from marconi.engine.params import StageParams
from marconi.engine.stage import RxStage, Stage


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
    accepts_item_type = "f"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("blockinterleaver_ff", perm=[int(x) for x in params["perm"]], mode=True)


class _DepunctureParams(StageParams):
    keep_mask: list[int]

    @model_validator(mode="after")
    def _keeps_something(self) -> "_DepunctureParams":
        if not any(self.keep_mask):
            raise ValueError(
                "keep_mask must keep at least one position: an all-erasure "
                "mask consumes no input and the flowgraph never terminates"
            )
        return self


class Depuncture(RxStage[CompileContext]):
    """Generic depuncture, BITS->BITS soft. Scatters soft into a wider codeword per
    a keep-mask (0 = erasure), from STOCK GR blocks: patterned_interleaver pulls
    kept positions from the stream and erasures from a null_source.
    Protocol puncture tables are params."""

    name = "depuncture"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    params_model = _DepunctureParams
    accepts_item_type = "f"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _DepunctureParams.model_validate(dict(params))
        src = b.tail  # incoming soft stream (never None after IO source)
        assert src is not None
        il = b.add(
            "patterned_interleaver_f", pattern=[0 if m else 1 for m in p.keep_mask]
        )
        b.connect(src, il, dst_port=0)
        b.connect(b.add("null_source_f"), il, dst_port=1)
        b.set_tail(il)


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


def _emit_cc(b: CompileContext, p: _ConvParams) -> None:
    fb, t = p.frame_bits, p.tail
    b.chain(
        "trellis_viterbi",
        rate_inv=p.rate_inv,
        polys=[int(x) for x in p.polys],
        frame_bits=fb,
        tail=t,
    )
    b.chain("keep_m_in_n_b", m=fb, n=fb + t)


_FEC_EMIT: dict[str, Callable[[CompileContext, "_ConvParams"], None]] = {"cc": _emit_cc}


class Fec(RxStage[CompileContext]):
    """Generic FEC decode, BITS soft -> BITS hard. `scheme` selects the decoder via
    a factory map (no if/else); `cc` wires gnuradio.trellis. The soft->hard boundary
    is here."""

    name = "fec"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    params_model = _ConvParams
    accepts_item_type = "f"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _ConvParams.model_validate(dict(params))
        _FEC_EMIT[p.scheme](b, p)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


CODING_STAGES: tuple[type[Stage[CompileContext]], ...] = (Deinterleave, Depuncture, Fec)
