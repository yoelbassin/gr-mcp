from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import RxStage, Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.params import StageParams


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

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return in_desc

    def validate_input(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> str | None:
        span = len(list(params["perm"]))
        if in_desc.frame_len is not None and in_desc.frame_len % span:
            return (
                f"permutes blocks of {span} items but the input is framed at "
                f"{in_desc.frame_len}, which {span} does not divide; permute "
                f"blocks would straddle frame boundaries and every frame after "
                f"the first would decode misaligned"
            )
        return None


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

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        p = _DepunctureParams.model_validate(dict(params))
        kept = sum(1 for m in p.keep_mask if m)
        frame = in_desc.frame_len
        if frame is None or frame % kept:
            return replace(in_desc, frame_len=None)
        return replace(in_desc, frame_len=frame // kept * len(p.keep_mask))

    def validate_input(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> str | None:
        p = _DepunctureParams.model_validate(dict(params))
        kept = sum(1 for m in p.keep_mask if m)
        if in_desc.frame_len is not None and in_desc.frame_len % kept:
            return (
                f"keeps {kept} of every {len(p.keep_mask)} codeword positions "
                f"but the input is framed at {in_desc.frame_len}, not a "
                f"multiple of {kept}; the mask would drift against the frames"
            )
        return None


class Harden(RxStage[CompileContext]):
    """Soft-LLR -> hard-bit decision, BITS->BITS. The non-FEC soft->hard bridge
    (fec is the FEC one): this engine's LLR is bit 1 = negative, so negate to the
    slicer's >=0 -> 1 convention then binary_slicer. Lets a soft correlate/gate
    front (sync_align) feed a hard coding tail (permute/block_code)."""

    name = "harden"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"

    class _Params(StageParams):
        pass

    params_model = _Params
    accepts_item_type = "f"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("multiply_const_ff", value=-1.0)
        b.chain("binary_slicer")

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", Carrier.HARD, frame_len=in_desc.frame_len)


class _SyncAlignParams(StageParams):
    access_code: str
    frame_len: StrictInt = Field(ge=1)
    threshold: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _ok(self) -> "_SyncAlignParams":
        if not self.access_code or any(c not in "01" for c in self.access_code):
            raise PydanticCustomError(
                "value_error", "access_code must be a non-empty bit string of 0/1"
            )
        if self.threshold >= len(self.access_code):
            raise PydanticCustomError(
                "value_error", "threshold must be < the access-code length"
            )
        return self


class SyncAlign(RxStage[CompileContext]):
    """Correlating frame align, BITS->BITS soft. Detects an uncoded sync word in
    the soft stream (stock correlate_access_code_tag_ff), then gates the
    fixed-length coded frame after each match (tag_gate), dropping inter-frame
    junk. The output is back-to-back frames that a downstream fec decodes
    contiguously — this is what makes GR-native sync-then-Viterbi expressible.
    The multiply_const_ff pair maps the soft-LLR sign (bit 1 negative here) to
    the correlator's convention (bit 1 positive) and back."""

    name = "sync_align"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    params_model = _SyncAlignParams
    accepts_item_type = "f"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _SyncAlignParams.model_validate(dict(params))
        b.chain("multiply_const_ff", value=-1.0)
        b.chain(
            "correlate_access_code_tag_ff",
            access_code=p.access_code,
            threshold=p.threshold,
            tag_name="frame_sync",
        )
        b.chain("tag_gate", frame_len=p.frame_len, tag_name="frame_sync")
        b.chain("multiply_const_ff", value=-1.0)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return replace(in_desc, frame_len=int(params["frame_len"]))


class _ConvParams(StageParams):
    scheme: str
    rate_inv: StrictInt = Field(ge=1)
    polys: list[int] = Field(min_length=1)
    frame_bits: StrictInt = Field(ge=1)
    tail: StrictInt = Field(default=0, ge=0)
    k: StrictInt = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _ok(self) -> "_ConvParams":
        if self.scheme not in _FEC_EMIT:
            raise PydanticCustomError(
                "value_error",
                "unknown fec scheme '{scheme}'; known: {known}",
                {"scheme": self.scheme, "known": ", ".join(sorted(_FEC_EMIT))},
            )
        if len(self.polys) != self.k * self.rate_inv:
            raise PydanticCustomError(
                "value_error",
                "need k*rate_inv={want} polys, got {have}",
                {"want": self.k * self.rate_inv, "have": len(self.polys)},
            )
        if self.frame_bits % self.k or self.tail % self.k:
            raise PydanticCustomError(
                "value_error", "frame_bits and tail must be multiples of k"
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
        k=p.k,
    )
    b.chain("unpack_k_bits_bb", k=p.k)
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
        return Descriptor(
            Level.BITS, "b", Carrier.HARD, frame_len=int(params["frame_bits"])
        )

    def validate_input(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> str | None:
        p = _ConvParams.model_validate(dict(params))
        need = (p.frame_bits + p.tail) // p.k * p.rate_inv
        if in_desc.frame_len is not None and in_desc.frame_len != need:
            return (
                f"decodes frames of {need} soft bits ((frame_bits "
                f"{p.frame_bits} + tail {p.tail}) / k {p.k} * rate_inv "
                f"{p.rate_inv}) but the input is framed at {in_desc.frame_len}"
                f"; align sync_align.frame_len with the fec frame geometry"
            )
        return None


CODING_STAGES: tuple[type[Stage[CompileContext]], ...] = (
    Deinterleave,
    Depuncture,
    Fec,
    Harden,
    SyncAlign,
)
