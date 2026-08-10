from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.coding import ops_bits
from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.primitives import effective_t, syndrome_table
from marconi.engine.stages.base import CodingStage, Stage
from marconi.engine.types.descriptor import Carrier
from marconi.engine.types.enums import DecodeMode, EmitMode, ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step

if TYPE_CHECKING:
    from marconi.engine.coding.stages_symbols import SymbolMapStep


class SyncWordStep(Step):
    conv: Literal["sync_word"] = "sync_word"
    sync: str = Field(
        default="",
        description=(
            "Byte-aligned sync word as an even-length HEX string, e.g. '9162' is "
            "the 16-bit word 0x9162. These are BYTES, not bits: '10' means the byte "
            "0x10, not the bit pair 1,0. Whole-byte patterns only; for any other "
            "length use `bits`. Set exactly one of `sync` or `bits`."
        ),
    )
    bits: str = Field(
        default="",
        description=(
            "Sync word as a raw '0'/'1' BIT string in the demodulated bit domain, "
            "e.g. '0101100011', for non-byte-multiple lengths. This is the exact "
            "bit pattern the correlator matches. If your pattern is in symbols or "
            "dibits (e.g. 4-level FSK symbols 0..3), map each symbol to its bits "
            "first and pass the resulting 0/1 string here. Never pass a raw symbol "
            "string: a digit run like '3133..' is accidentally valid hex, so `sync` "
            "would silently accept it and match nothing. Set exactly one of `sync` "
            "or `bits`."
        ),
    )
    max_errors: StrictInt = Field(
        default=0,
        ge=0,
        description=(
            "Correlation tolerance in bit flips (Hamming distance); 0 = exact. The "
            "sync word is found by correlation, not equality."
        ),
    )

    @model_validator(mode="after")
    def _pattern(self) -> "SyncWordStep":
        if bool(self.sync) == bool(self.bits):
            raise PydanticCustomError(
                "value_error", "exactly one of sync (hex) or bits ('0'/'1') must be set"
            )
        if self.sync:
            try:
                bytes.fromhex(self.sync)
            except ValueError:
                raise PydanticCustomError(
                    "value_error", "sync must be an even-length hex string"
                ) from None
        elif set(self.bits) - {"0", "1"}:
            raise PydanticCustomError(
                "value_error", "bits must contain only '0' and '1'"
            )
        return self


def check_codebook_sizing(
    code_bits: int, data_bits: int, table: list[int], decode: DecodeMode
) -> None:
    if len(table) != (1 << data_bits):
        raise PydanticCustomError(
            "value_error",
            "codebook table needs {want} entries for data_bits={data_bits}, "
            "got {have}",
            {
                "want": 1 << data_bits,
                "data_bits": data_bits,
                "have": len(table),
            },
        )
    if decode == DecodeMode.EXACT and code_bits > 24:
        raise PydanticCustomError(
            "value_error",
            "exact decode builds a 2^code_bits inverse table; beyond 24 "
            "bits use decode='nearest', which never builds it",
        )
    if decode == DecodeMode.NEAREST and code_bits > ops_bits.NEAREST_MAX_CODE_BITS:
        raise PydanticCustomError(
            "value_error",
            "nearest decode packs codewords through int64; {max} bits is the ceiling",
            {"max": ops_bits.NEAREST_MAX_CODE_BITS},
        )
    bad = [x for x in table if x < 0 or x >= (1 << code_bits)]
    if bad:
        raise PydanticCustomError(
            "value_error",
            "codebook entries must lie in [0, 2**code_bits); got {bad}",
            {"bad": bad[:5]},
        )
    if len(set(table)) != len(table):
        raise PydanticCustomError(
            "value_error",
            "codebook entries must be unique; duplicate codewords make "
            "decode ambiguous",
        )


class CodebookStep(Step):
    conv: Literal["codebook"] = "codebook"
    code_bits: StrictInt = Field(
        ge=1,
        description="Width in bits of each input codeword (2 for a "
        "bi-phase/Manchester chip pair, 6 for a 3-of-6 symbol).",
    )
    data_bits: StrictInt = Field(
        ge=1, description="Bits of data each codeword decodes to (1 for Manchester)."
    )
    table: list[int] = Field(
        description="Codeword for each data value, indexed by that value: "
        "table[d] is the code_bits-wide codeword for data d. Manchester is "
        "[1, 2] (data 0 -> 01, data 1 -> 10).",
    )
    decode: DecodeMode = DecodeMode.EXACT

    @model_validator(mode="after")
    def _sized(self) -> "CodebookStep":
        check_codebook_sizing(self.code_bits, self.data_bits, self.table, self.decode)
        return self


def _codebook_kw(step: "CodebookStep | SymbolMapStep") -> dict[str, Any]:
    return {
        "code_bits": int(step.code_bits),
        "data_bits": int(step.data_bits),
        "table": [int(x) for x in step.table],
        "decode": step.decode,
    }


class DescrambleStep(Step):
    conv: Literal["descramble"] = "descramble"
    sequence: str = Field(
        default="",
        description=(
            "Additive de-whitening mask as a HEX string, unpacked MSB-first per "
            "byte and XOR-tiled over the bit stream (restarting at each window). "
            "A whitening sequence is bits: pack them into bytes and hex-encode "
            "(np.packbits(bits).tobytes().hex()); reverse each byte's bits first "
            "for an LSB-first protocol."
        ),
    )
    lfsr_mask: StrictInt | None = Field(
        default=None,
        description="Fibonacci LFSR tap polynomial (bit i set = tap at x^i).",
    )
    lfsr_seed: StrictInt | None = Field(
        default=None, description="Initial LFSR register contents (nonzero)."
    )
    lfsr_len: StrictInt | None = Field(
        default=None,
        ge=2,
        le=64,
        description="LFSR register width in bits; mask and seed must fit it.",
    )

    @model_validator(mode="after")
    def _one_mode(self) -> "DescrambleStep":
        triple = (self.lfsr_mask, self.lfsr_seed, self.lfsr_len)
        has_lfsr = any(v is not None for v in triple)
        if has_lfsr and None in triple:
            raise PydanticCustomError(
                "value_error", "lfsr_mask, lfsr_seed and lfsr_len come together"
            )
        if has_lfsr == bool(self.sequence):
            raise PydanticCustomError(
                "value_error", "exactly one of sequence or lfsr_* must be set"
            )
        if has_lfsr:
            assert self.lfsr_mask is not None and self.lfsr_seed is not None
            assert self.lfsr_len is not None
            top = 1 << self.lfsr_len
            if not 0 < self.lfsr_mask < top or not 0 < self.lfsr_seed < top:
                raise PydanticCustomError(
                    "value_error",
                    "lfsr_mask and lfsr_seed must be nonzero and fit lfsr_len",
                )
        else:
            if len(self.sequence) >= 16 and set(self.sequence) <= {"0", "1"}:
                raise PydanticCustomError(
                    "value_error",
                    "sequence is a HEX mask (unpacked MSB-first per byte), but "
                    "this is a 0/1 bit string; pack the bits into bytes and "
                    "hex-encode them",
                )
            try:
                if not bytes.fromhex(self.sequence):
                    raise ValueError
            except ValueError:
                raise PydanticCustomError(
                    "value_error",
                    "sequence must be a non-empty even-length hex string",
                ) from None
        return self


class SyncWord(CodingStage[SyncWordStep]):
    name = "sync_word"
    description = (
        "Find a bit pattern by correlation and MARK a window at each match "
        "(cursor at payload start); the stream passes through unchanged. "
        "Window-scoped coding stages (codebook/block_code/permute/descramble) "
        "restart their stride at each window. A later window-seeding stage "
        "(segment/mark_frame) OVERWRITES these windows."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    seeds_windows = True
    sync_search = True
    family = "acquisition"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    step_model = SyncWordStep

    def emit_rx(self, b: CodingBuilder, step: SyncWordStep) -> None:
        b.add(
            ops_bits.sync_word_rx,
            sync=step.sync,
            bits=step.bits,
            max_errors=step.max_errors,
        )


class MarkFrameStep(Step):
    conv: Literal["mark_frame"] = "mark_frame"
    offset_bits: StrictInt = Field(default=0, ge=0)


class MarkFrame(CodingStage[MarkFrameStep]):
    name = "mark_frame"
    description = (
        "Seed one window per burst mark at offset_bits - bridges burst-detector "
        "marks into the window model for window-scoped coding stages."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    seeds_windows = True
    family = "acquisition"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    step_model = MarkFrameStep

    def emit_rx(self, b: CodingBuilder, step: MarkFrameStep) -> None:
        b.add(ops_bits.mark_frame_rx, offset_bits=int(step.offset_bits))


class SegmentStep(Step):
    conv: Literal["segment"] = "segment"
    frame_body_len: StrictInt = Field(ge=1)


class Segment(CodingStage[SegmentStep]):
    name = "segment"
    description = (
        "Re-tile the stream into fixed frame_body_len windows from position 0, "
        "DISCARDING any windows a prior stage seeded. Correct after a gating "
        "stage (sync_align); after a marking stage (sync_word) it destroys the "
        "marks - let window-scoped coding stages consume sync_word's windows "
        "directly instead."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    seeds_windows = True
    family = "acquisition"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    step_model = SegmentStep

    def emit_rx(self, b: CodingBuilder, step: SegmentStep) -> None:
        b.add(ops_bits.segment_rx, frame_body_len=int(step.frame_body_len))


class Codebook(CodingStage[CodebookStep]):
    name = "codebook"
    description = (
        "Fixed-width line/block code: map each code_bits-wide input codeword to "
        "a data_bits value via table. Expresses a bi-phase/Manchester pair "
        "decode (code_bits=2, table=[1,2]), 3-of-6, and PPM chip-pairs."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    step_model = CodebookStep

    def emit_rx(self, b: CodingBuilder, step: CodebookStep) -> None:
        b.add(ops_bits.codebook_rx, **_codebook_kw(step))


class BlockCodeStep(Step):
    conv: Literal["block_code"] = "block_code"
    code_bits: StrictInt = Field(ge=2)
    data_bits: StrictInt = Field(ge=1)
    parity_masks: list[int]
    correct_single: bool | None = None
    correct: StrictInt | None = Field(default=None, ge=0, le=3)
    emit: EmitMode = EmitMode.DATA

    @model_validator(mode="after")
    def _shaped(self) -> "BlockCodeStep":
        if self.data_bits >= self.code_bits:
            raise PydanticCustomError("value_error", "data_bits must be < code_bits")
        if len(self.parity_masks) != self.code_bits - self.data_bits:
            raise PydanticCustomError(
                "value_error",
                "need {want} parity masks for ({n},{k}), got {have}",
                {
                    "want": self.code_bits - self.data_bits,
                    "n": self.code_bits,
                    "k": self.data_bits,
                    "have": len(self.parity_masks),
                },
            )
        if self.correct is not None and self.correct_single is not None:
            raise PydanticCustomError(
                "value_error", "set correct or correct_single, not both"
            )
        if self.correct is not None and self.correct >= 2:
            try:
                syndrome_table(
                    [int(m) for m in self.parity_masks],
                    self.code_bits,
                    self.data_bits,
                    self.correct,
                )
            except ValueError as e:
                raise PydanticCustomError(
                    "value_error", "{msg}", {"msg": str(e)}
                ) from None
        t = effective_t(
            self.code_bits - self.data_bits,
            self.data_bits,
            self.correct_single,
            self.correct,
        )
        if t == 1:
            cols = [
                tuple((mask >> j) & 1 for mask in self.parity_masks)
                for j in range(self.data_bits)
            ]
            if any(sum(col) < 2 for col in cols) or len(set(cols)) != len(cols):
                raise PydanticCustomError(
                    "value_error",
                    "correct_single needs pairwise-distinct parity-check "
                    "columns of weight >= 2 - single-error syndromes must "
                    "be unique across data and parity positions",
                )
        return self


class BlockCode(CodingStage[BlockCodeStep]):
    name = "block_code"
    description = (
        "Linear block FEC over a (code_bits, data_bits) code: syndrome-decode "
        "each stride against parity_masks, correcting up to `correct` bit "
        "errors, and tally how many words checked out (signal evidence)."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    validates_words = True
    step_model = BlockCodeStep

    def emit_rx(self, b: CodingBuilder, step: BlockCodeStep) -> None:
        b.add(
            ops_bits.block_code_rx,
            code_bits=step.code_bits,
            data_bits=step.data_bits,
            parity_masks=[int(x) for x in step.parity_masks],
            correct_single=step.correct_single,
            correct=step.correct,
            emit=step.emit,
        )


class PermuteStep(Step):
    conv: Literal["permute"] = "permute"
    perm: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _non_negative(self) -> "PermuteStep":
        if any(i < 0 for i in self.perm):
            raise PydanticCustomError(
                "value_error",
                "perm indices must be >= 0; a negative index would wrap and "
                "understate the input stride",
            )
        return self


class Permute(CodingStage[PermuteStep]):
    name = "permute"
    description = (
        "Block (de)interleaver: reorder each block of bits by the perm index "
        "list (output position i takes input perm[i]); undoes a matrix/"
        "convolutional interleaver when perm is its inverse map."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    step_model = PermuteStep

    def emit_rx(self, b: CodingBuilder, step: PermuteStep) -> None:
        b.add(ops_bits.permute_rx, perm=[int(x) for x in step.perm])


class RealignStep(Step):
    conv: Literal["realign"] = "realign"
    bit_offset: StrictInt = Field(ge=0)


class Realign(CodingStage[RealignStep]):
    name = "realign"
    description = (
        "Drop bit_offset leading bits to re-phase a stream before a "
        "fixed-width block decode — e.g. selecting which of the two pair "
        "phases a bi-phase/Manchester codebook consumes."
    )
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    step_model = RealignStep

    def emit_rx(self, b: CodingBuilder, step: RealignStep) -> None:
        b.add(ops_bits.realign_rx, bit_offset=int(step.bit_offset))


class DifferentialStep(Step):
    conv: Literal["differential"] = "differential"
    invert: bool = False


class Differential(CodingStage[DifferentialStep]):
    name = "differential"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    step_model = DifferentialStep

    def emit_rx(self, b: CodingBuilder, step: DifferentialStep) -> None:
        b.add(ops_bits.differential_rx, invert=step.invert)


class NibbleSwapStep(Step):
    conv: Literal["nibble_swap"] = "nibble_swap"


class NibbleSwap(CodingStage[NibbleSwapStep]):
    name = "nibble_swap"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    step_model = NibbleSwapStep

    def emit_rx(self, b: CodingBuilder, step: NibbleSwapStep) -> None:
        b.add(ops_bits.nibble_swap_rx)


class RsCodeStep(Step):
    conv: Literal["rs_code"] = "rs_code"
    symbol_bits: StrictInt = Field(ge=3, le=16)
    n: StrictInt = Field(ge=3)
    k: StrictInt = Field(ge=1)
    prim_poly: StrictInt
    fcr: StrictInt = Field(default=0, ge=0)
    generator: StrictInt = Field(default=2, ge=1)
    emit: EmitMode = EmitMode.DATA

    @model_validator(mode="after")
    def _shaped(self) -> "RsCodeStep":
        if self.k >= self.n:
            raise PydanticCustomError("value_error", "k must be < n")
        if self.n > (1 << self.symbol_bits) - 1:
            raise PydanticCustomError("value_error", "n must be <= 2^symbol_bits - 1")
        if self.prim_poly.bit_length() != self.symbol_bits + 1:
            raise PydanticCustomError(
                "value_error", "prim_poly must have degree symbol_bits"
            )
        return self


class RsCode(CodingStage[RsCodeStep]):
    name = "rs_code"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    validates_words = True
    step_model = RsCodeStep

    def emit_rx(self, b: CodingBuilder, step: RsCodeStep) -> None:
        b.add(
            ops_bits.rs_code_rx,
            symbol_bits=step.symbol_bits,
            n=step.n,
            k=step.k,
            prim_poly=step.prim_poly,
            fcr=step.fcr,
            generator=step.generator,
            emit=step.emit,
        )


class Descramble(CodingStage[DescrambleStep]):
    name = "descramble"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD
    step_model = DescrambleStep

    def emit_rx(self, b: CodingBuilder, step: DescrambleStep) -> None:
        b.add(
            ops_bits.descramble_rx,
            sequence=step.sequence,
            lfsr_mask=step.lfsr_mask,
            lfsr_seed=step.lfsr_seed,
            lfsr_len=step.lfsr_len,
        )


CODING_BITS_STAGES: tuple[type[Stage[CodingBuilder, Any]], ...] = (
    BlockCode,
    Codebook,
    Descramble,
    Differential,
    MarkFrame,
    NibbleSwap,
    Permute,
    Realign,
    RsCode,
    Segment,
    SyncWord,
)
