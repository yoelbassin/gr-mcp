from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import Stage
from marconi.engine.types.bounds import (
    MAX_DECODER_ITERATIONS,
    MAX_FRAME_ITEMS,
    MAX_LIST_SIZE,
    MAX_OVERSAMPLE,
    MAX_POLY_BITS,
)
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.perm import check_block_permutation
from marconi.engine.types.step import Step


class DeinterleaveStep(Step):
    conv: Literal["deinterleave"] = "deinterleave"
    perm: list[int]

    @model_validator(mode="after")
    def _permutes(self) -> "DeinterleaveStep":
        check_block_permutation(self.perm, field="perm")
        return self


class Deinterleave(Stage[CompileContext, DeinterleaveStep]):
    """Generic block permute, BITS->BITS (carrier passes through). Stock
    blockinterleaver_ff: out[t]=in[perm[t]] per block of len(perm)."""

    name = "deinterleave"
    description = (
        "Block de-interleave on the soft LLR stream: output item i takes "
        "input perm[i] within each len(perm) block, and perm must be a "
        "strict permutation (every index exactly once). For a gather that "
        "may drop or repeat positions, use permute in the hard-bits lane."
    )
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
        # A mask, so only 0 and 1 mean anything. Read for truthiness alone, any
        # other value was silently "keep" — a spec that looks like it says
        # something the emitter never read.
        if any(m not in (0, 1) for m in self.keep_mask):
            raise PydanticCustomError(
                "value_error",
                "keep_mask entries are 1 (keep) or 0 (erasure); got {bad}",
                {"bad": [m for m in self.keep_mask if m not in (0, 1)][:5]},
            )
        return self


class Depuncture(Stage[CompileContext, DepunctureStep]):
    """Generic depuncture, BITS->BITS soft. Scatters soft into a wider codeword per
    a keep-mask (0 = erasure), from STOCK GR blocks: patterned_interleaver pulls
    kept positions from the stream and erasures from a null_source.
    Protocol puncture tables are params."""

    name = "depuncture"
    description = (
        "Re-insert erased positions (0.0 LLR, keep_mask=) ahead of a "
        "convolutional/LDPC decoder, restoring the mother-code rate. Soft quality "
        "scoring ignores the inserted erasures."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = DepunctureStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: DepunctureStep) -> None:
        src = b.require_tail()
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


class Harden(Stage[CompileContext, HardenStep]):
    """Soft-LLR -> hard-bit decision, BITS->BITS. The non-FEC soft->hard bridge
    (fec is the FEC one): this engine's LLR is bit 1 = negative, so negate to the
    slicer's >=0 -> 1 convention then binary_slicer. Lets a soft correlate/gate
    front (sync_align) feed a hard coding tail (permute/block_code)."""

    name = "harden"
    description = (
        "Soft LLRs to hard bits (bit 1 = negative LLR) - when the downstream "
        "stage needs hard bits and no soft decoder is in the path."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = HardenStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: HardenStep) -> None:
        b.chain_llr_flip()
        b.chain("binary_slicer")

    def out_descriptor(self, in_desc: Descriptor, step: HardenStep) -> Descriptor:
        return Descriptor(
            Level.BITS, ItemType.B, Carrier.HARD, frame_len=in_desc.frame_len
        )


class SyncAlignStep(Step):
    conv: Literal["sync_align"] = "sync_align"
    access_code: str
    frame_len: StrictInt = Field(ge=1, le=MAX_FRAME_ITEMS)
    threshold: StrictInt = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _ok(self) -> "SyncAlignStep":
        if not self.access_code or any(c not in "01" for c in self.access_code):
            raise PydanticCustomError(
                "value_error", "access_code must be a non-empty bit string of 0/1"
            )
        if len(self.access_code) > 64:
            raise PydanticCustomError(
                "value_error",
                "access_code is limited to 64 bits - the stock GR correlator "
                "packs it into one 64-bit register",
            )
        if self.threshold >= len(self.access_code):
            raise PydanticCustomError(
                "value_error", "threshold must be < the access-code length"
            )
        return self


class SyncAlign(Stage[CompileContext, SyncAlignStep]):
    name = "sync_align"
    description = (
        "Correlate an access code on the soft bit stream and GATE: keep "
        "exactly frame_len items after each hit, drop everything else. Output "
        "is a contiguous stream of frame bodies with no windows - follow with "
        "segment to re-tile it for window-scoped coding stages."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = SyncAlignStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: SyncAlignStep) -> None:
        m = len(step.access_code)
        chance = sum(math.comb(m, i) for i in range(step.threshold + 1)) / 2.0**m
        b.chain_llr_flip()
        b.chain(
            "correlate_access_code_tag_ff",
            access_code=step.access_code,
            threshold=step.threshold,
            tag_name="frame_sync",
        )
        b.chain(
            "tag_gate",
            frame_len=step.frame_len,
            tag_name="frame_sync",
            chance_per_item=chance,
            pattern_bits=m,
        )
        b.chain_llr_flip()

    def out_descriptor(self, in_desc: Descriptor, step: SyncAlignStep) -> Descriptor:
        return replace(in_desc, frame_len=int(step.frame_len))


class FecStep(Step):
    conv: Literal["fec"] = "fec"
    scheme: str
    rate_inv: StrictInt = Field(ge=1)
    polys: list[int] = Field(min_length=1)
    frame_bits: StrictInt = Field(ge=1, le=MAX_FRAME_ITEMS)
    tail: StrictInt = Field(default=0, ge=0, le=MAX_FRAME_ITEMS)
    k: StrictInt = Field(default=1, ge=1, le=MAX_OVERSAMPLE)

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
        widest = max(int(p).bit_length() for p in self.polys)
        if widest > MAX_POLY_BITS:
            raise PydanticCustomError(
                "value_error",
                "the widest poly is {widest} bits, so the decoder's trellis "
                "has 2**{states_exp} states; {max} bits is the ceiling "
                "(the K=15 deep-space codes are the widest deployed, and "
                "everything else in use is K<=9)",
                {
                    "widest": widest,
                    "states_exp": widest - 1,
                    "max": MAX_POLY_BITS,
                },
            )
        if any(int(p) <= 0 for p in self.polys):
            raise PydanticCustomError(
                "value_error",
                "every poly must be > 0; a zero generator emits a constant "
                "branch the Viterbi metric cannot distinguish",
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


class Fec(Stage[CompileContext, FecStep]):
    """Generic FEC decode, BITS soft -> BITS hard. `scheme` selects the decoder via
    a factory map (no if/else); `cc` wires gnuradio.trellis. The soft->hard boundary
    is here."""

    name = "fec"
    description = (
        "Convolutional (Viterbi) decode of soft LLRs: scheme='cc' with "
        "polys/rate_inv/k from the protocol datasheet; frame_bits per window. "
        "Depuncture first when the code is punctured."
    )
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


class PolarStep(Step):
    conv: Literal["polar"] = "polar"
    block_size: StrictInt = Field(ge=2, le=MAX_FRAME_ITEMS)
    info_bits: StrictInt = Field(ge=1)
    frozen_positions: list[int]
    frozen_values: list[int]
    # the decoder holds list_size full candidate paths at once
    list_size: StrictInt = Field(default=1, ge=1, le=MAX_LIST_SIZE)

    @model_validator(mode="after")
    def _ok(self) -> "PolarStep":
        n, k = self.block_size, self.info_bits
        if n & (n - 1):
            raise PydanticCustomError(
                "value_error", "block_size must be a power of two, got {n}", {"n": n}
            )
        if k >= n:
            raise PydanticCustomError(
                "value_error",
                "info_bits {k} must be < block_size {n}",
                {"k": k, "n": n},
            )
        nfrozen = len(self.frozen_positions)
        if nfrozen != n - k:
            raise PydanticCustomError(
                "value_error",
                "need block_size-info_bits={want} frozen positions, got {have}",
                {"want": n - k, "have": nfrozen},
            )
        if len(self.frozen_values) != nfrozen:
            raise PydanticCustomError(
                "value_error", "frozen_values must match frozen_positions length"
            )
        if len(set(self.frozen_positions)) != nfrozen:
            raise PydanticCustomError(
                "value_error", "frozen_positions must be distinct"
            )
        if any(not (0 <= p < n) for p in self.frozen_positions):
            raise PydanticCustomError(
                "value_error", "frozen_positions must be in [0, block_size)"
            )
        if any(v not in (0, 1) for v in self.frozen_values):
            raise PydanticCustomError("value_error", "frozen_values must be 0 or 1")
        return self


class Polar(Stage[CompileContext, PolarStep]):
    """Polar decode, BITS soft -> BITS hard, via stock gr-fec
    (fec.polar_decoder_sc[_list] in fec.extended_decoder). list_size=1 is plain
    successive-cancellation; >1 selects the SC-list decoder. block_size (N),
    info_bits (K), and the frozen set (positions + values) are caller params --
    polar code construction is datasheet work, not the engine's. The engine's
    LLR is bit 1 = negative but the SC decoder wants bit 1 = positive, so the
    leading multiply_const_ff(-1) converts the convention (like harden/psk_soft
    _demap do their sign corrections)."""

    name = "polar"
    description = (
        "Polar SC decode of soft LLRs: frozen_positions and codeword size from "
        "the datasheet."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = PolarStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: PolarStep) -> None:
        b.chain_llr_flip()
        b.chain(
            "polar_decode",
            block_size=step.block_size,
            info_bits=step.info_bits,
            frozen_positions=list(step.frozen_positions),
            frozen_values=list(step.frozen_values),
            list_size=step.list_size,
        )

    def out_descriptor(self, in_desc: Descriptor, step: PolarStep) -> Descriptor:
        return Descriptor(
            Level.BITS, ItemType.B, Carrier.HARD, frame_len=int(step.info_bits)
        )

    def validate_input(self, in_desc: Descriptor, step: PolarStep) -> str | None:
        if in_desc.frame_len is not None and in_desc.frame_len != step.block_size:
            return (
                f"decodes {step.block_size}-soft-bit polar frames but the input is "
                f"framed at {in_desc.frame_len}; align the upstream frame geometry "
                f"(sync_align.frame_len) with block_size"
            )
        return None


class LdpcStep(Step):
    conv: Literal["ldpc"] = "ldpc"
    block_size: StrictInt = Field(ge=2, le=MAX_FRAME_ITEMS)
    check_nodes: list[list[int]]
    max_iterations: StrictInt = Field(default=50, ge=1, le=MAX_DECODER_ITERATIONS)

    @model_validator(mode="after")
    def _ok(self) -> "LdpcStep":
        n = self.block_size
        m = len(self.check_nodes)
        if not 1 <= m < n:
            raise PydanticCustomError(
                "value_error",
                "need 1..block_size-1 parity checks, got {m}",
                {"m": m},
            )
        for row in self.check_nodes:
            if not row:
                raise PydanticCustomError(
                    "value_error", "each parity check must connect >= 1 variable"
                )
            if len(set(row)) != len(row):
                raise PydanticCustomError(
                    "value_error", "variable indices within a check must be distinct"
                )
            if any(not 0 <= v < n for v in row):
                raise PydanticCustomError(
                    "value_error", "variable indices must be in [0, block_size)"
                )
        return self

    @property
    def info_bits(self) -> int:
        return self.block_size - len(self.check_nodes)


class Ldpc(Stage[CompileContext, LdpcStep]):
    """LDPC decode, BITS soft -> BITS hard, via stock gr-fec (soft belief-
    propagation fec.ldpc_decoder in fec.extended_decoder). The parity-check
    matrix -- given sparsely as the variable indices each check connects -- plus
    block_size (N) are caller params; the codeword is N bits and the decode emits
    K = N - (parity checks) info bits. LDPC construction is datasheet work, not
    the engine's. The engine's LLR is bit 1 = negative but the decoder wants bit
    1 = positive, so the leading multiply_const_ff(-1) converts the convention
    (like harden/psk_soft_demap do their sign corrections). The ragged matrix
    crosses the IR flattened (check_flat + check_lens) because a param is a flat
    list; the backend rebuilds and serializes it to the alist gr-fec reads."""

    name = "ldpc"
    description = (
        "LDPC belief-propagation decode of soft LLRs against caller-supplied "
        "check nodes. Stock-GR limits apply (systematic-last, no puncturing)."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    step_model = LdpcStep
    accepts_item_type = ItemType.F
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: LdpcStep) -> None:
        b.chain_llr_flip()
        b.chain(
            "ldpc_decode",
            block_size=step.block_size,
            check_flat=[v for row in step.check_nodes for v in row],
            check_lens=[len(row) for row in step.check_nodes],
            max_iterations=step.max_iterations,
        )

    def out_descriptor(self, in_desc: Descriptor, step: LdpcStep) -> Descriptor:
        return Descriptor(
            Level.BITS, ItemType.B, Carrier.HARD, frame_len=step.info_bits
        )

    def validate_input(self, in_desc: Descriptor, step: LdpcStep) -> str | None:
        if in_desc.frame_len is not None and in_desc.frame_len != step.block_size:
            return (
                f"decodes {step.block_size}-soft-bit LDPC codewords but the input "
                f"is framed at {in_desc.frame_len}; align the upstream frame "
                f"geometry (sync_align.frame_len) with block_size"
            )
        return None


CODING_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (
    Deinterleave,
    Depuncture,
    Fec,
    Harden,
    Ldpc,
    Polar,
    SyncAlign,
)
