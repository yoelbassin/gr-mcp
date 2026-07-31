from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import RxStage, Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class DeinterleaveStep(Step):
    conv: Literal["deinterleave"] = "deinterleave"
    perm: list[int]


class Deinterleave(RxStage[CompileContext, DeinterleaveStep]):
    """Generic block permute, BITS->BITS (carrier passes through). Stock
    blockinterleaver_ff: out[t]=in[perm[t]] per block of len(perm)."""

    name = "deinterleave"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = DeinterleaveStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: DeinterleaveStep) -> None:
        b.chain("blockinterleaver_ff", perm=[int(x) for x in step.perm], mode=True)

    def out_descriptor(self, in_desc: Descriptor, step: DeinterleaveStep) -> Descriptor:
        return in_desc

    def validate_input(self, in_desc: Descriptor, step: DeinterleaveStep) -> str | None:
        span = len(step.perm)
        if in_desc.frame_len is not None and in_desc.frame_len % span:
            return (
                f"permutes blocks of {span} items but the input is framed at "
                f"{in_desc.frame_len}, which {span} does not divide; permute "
                f"blocks would straddle frame boundaries and every frame after "
                f"the first would decode misaligned"
            )
        return None


class DepunctureStep(Step):
    conv: Literal["depuncture"] = "depuncture"
    keep_mask: list[int]

    @model_validator(mode="after")
    def _keeps_something(self) -> "DepunctureStep":
        if not any(self.keep_mask):
            raise ValueError(
                "keep_mask must keep at least one position: an all-erasure "
                "mask consumes no input and the flowgraph never terminates"
            )
        return self


class Depuncture(RxStage[CompileContext, DepunctureStep]):
    """Generic depuncture, BITS->BITS soft. Scatters soft into a wider codeword per
    a keep-mask (0 = erasure), from STOCK GR blocks: patterned_interleaver pulls
    kept positions from the stream and erasures from a null_source.
    Protocol puncture tables are params."""

    name = "depuncture"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = DepunctureStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: DepunctureStep) -> None:
        src = b.tail  # incoming soft stream (never None after IO source)
        assert src is not None
        il = b.add(
            "patterned_interleaver_f",
            pattern=[0 if m else 1 for m in step.keep_mask],
        )
        b.connect(src, il, dst_port=0)
        b.connect(b.add("null_source_f"), il, dst_port=1)
        b.set_tail(il)

    def out_descriptor(self, in_desc: Descriptor, step: DepunctureStep) -> Descriptor:
        kept = sum(1 for m in step.keep_mask if m)
        frame = in_desc.frame_len
        if frame is None or frame % kept:
            return replace(in_desc, frame_len=None)
        return replace(in_desc, frame_len=frame // kept * len(step.keep_mask))

    def validate_input(self, in_desc: Descriptor, step: DepunctureStep) -> str | None:
        kept = sum(1 for m in step.keep_mask if m)
        if in_desc.frame_len is not None and in_desc.frame_len % kept:
            return (
                f"keeps {kept} of every {len(step.keep_mask)} codeword positions "
                f"but the input is framed at {in_desc.frame_len}, not a "
                f"multiple of {kept}; the mask would drift against the frames"
            )
        return None


class HardenStep(Step):
    conv: Literal["harden"] = "harden"


class Harden(RxStage[CompileContext, HardenStep]):
    """Soft-LLR -> hard-bit decision, BITS->BITS. The non-FEC soft->hard bridge
    (fec is the FEC one): this engine's LLR is bit 1 = negative, so negate to the
    slicer's >=0 -> 1 convention then binary_slicer. Lets a soft correlate/gate
    front (sync_align) feed a hard coding tail (permute/block_code)."""

    name = "harden"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = HardenStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: HardenStep) -> None:
        b.chain("multiply_const_ff", value=-1.0)
        b.chain("binary_slicer")

    def out_descriptor(self, in_desc: Descriptor, step: HardenStep) -> Descriptor:
        return Descriptor(
            Level.BITS, ItemType.B, Carrier.HARD, frame_len=in_desc.frame_len
        )


class SyncAlignStep(Step):
    conv: Literal["sync_align"] = "sync_align"
    access_code: str
    frame_len: StrictInt = Field(ge=1)
    threshold: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _ok(self) -> "SyncAlignStep":
        if not self.access_code or any(c not in "01" for c in self.access_code):
            raise PydanticCustomError(
                "value_error", "access_code must be a non-empty bit string of 0/1"
            )
        if self.threshold >= len(self.access_code):
            raise PydanticCustomError(
                "value_error", "threshold must be < the access-code length"
            )
        return self


class SyncAlign(RxStage[CompileContext, SyncAlignStep]):
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
    step_model = SyncAlignStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: SyncAlignStep) -> None:
        b.chain("multiply_const_ff", value=-1.0)
        b.chain(
            "correlate_access_code_tag_ff",
            access_code=step.access_code,
            threshold=step.threshold,
            tag_name="frame_sync",
        )
        b.chain("tag_gate", frame_len=step.frame_len, tag_name="frame_sync")
        b.chain("multiply_const_ff", value=-1.0)

    def out_descriptor(self, in_desc: Descriptor, step: SyncAlignStep) -> Descriptor:
        return replace(in_desc, frame_len=int(step.frame_len))


class FecStep(Step):
    conv: Literal["fec"] = "fec"
    scheme: str
    rate_inv: StrictInt = Field(ge=1)
    polys: list[int] = Field(min_length=1)
    frame_bits: StrictInt = Field(ge=1)
    tail: StrictInt = Field(default=0, ge=0)
    k: StrictInt = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _ok(self) -> "FecStep":
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


def _emit_cc(b: CompileContext, step: FecStep) -> None:
    fb, t = step.frame_bits, step.tail
    b.chain(
        "trellis_viterbi",
        rate_inv=step.rate_inv,
        polys=[int(x) for x in step.polys],
        frame_bits=fb,
        tail=t,
        k=step.k,
    )
    b.chain("unpack_k_bits_bb", k=step.k)
    b.chain("keep_m_in_n_b", m=fb, n=fb + t)


_FEC_EMIT: dict[str, Callable[[CompileContext, FecStep], None]] = {"cc": _emit_cc}


class Fec(RxStage[CompileContext, FecStep]):
    """Generic FEC decode, BITS soft -> BITS hard. `scheme` selects the decoder via
    a factory map (no if/else); `cc` wires gnuradio.trellis. The soft->hard boundary
    is here."""

    name = "fec"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = FecStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: FecStep) -> None:
        _FEC_EMIT[step.scheme](b, step)

    def out_descriptor(self, in_desc: Descriptor, step: FecStep) -> Descriptor:
        return Descriptor(
            Level.BITS, ItemType.B, Carrier.HARD, frame_len=int(step.frame_bits)
        )

    def validate_input(self, in_desc: Descriptor, step: FecStep) -> str | None:
        need = (step.frame_bits + step.tail) // step.k * step.rate_inv
        if in_desc.frame_len is not None and in_desc.frame_len != need:
            return (
                f"decodes frames of {need} soft bits ((frame_bits "
                f"{step.frame_bits} + tail {step.tail}) / k {step.k} * rate_inv "
                f"{step.rate_inv}) but the input is framed at {in_desc.frame_len}"
                f"; align sync_align.frame_len with the fec frame geometry"
            )
        return None


CODING_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (
    Deinterleave,
    Depuncture,
    Fec,
    Harden,
    SyncAlign,
)
