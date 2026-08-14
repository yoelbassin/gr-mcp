from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.coding import ops_bits, ops_symbols
from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.stages_bits import _codebook_kw, check_codebook_sizing
from marconi.engine.stages.base import CodingStage, Stage
from marconi.engine.types.bounds import (
    MAX_FRAME_ITEMS,
    MAX_WINDOW_SYMBOLS,
    check_match_tolerance,
)
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import DcMode, DecodeMode, ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step

# The hard-symbol wire's range (ItemType.S is int16), so a stage that labels
# symbols must label them with values the wire can carry.
_INT16_MIN, _INT16_MAX = -(1 << 15), (1 << 15) - 1


class SyncSymbolsStep(Step):
    conv: Literal["sync_symbols"] = "sync_symbols"
    pattern: list[int] = Field(
        description=(
            "Sign pattern matched against the soft symbols: +1 requires a "
            "positive symbol, -1 a negative one, 0 is a wildcard (don't "
            "care). NOT a 0/1 bit string — encode bits as +-1 first, or "
            "every 0 becomes a wildcard and the correlator over-matches."
        )
    )
    max_errors: StrictInt = Field(default=0, ge=0)
    pre_symbols: StrictInt = Field(default=0, ge=0, le=MAX_FRAME_ITEMS)

    @property
    def compared_positions(self) -> int:
        """Entries the correlator actually scores. Wildcards are skipped by
        ops_symbols.sync_symbols_rx (`for j in np.flatnonzero(want)`), so they
        are not positions an error budget may be spent against."""
        return sum(1 for x in self.pattern if x)

    @model_validator(mode="after")
    def _sign_alphabet(self) -> "SyncSymbolsStep":
        if any(x not in (-1, 0, 1) for x in self.pattern):
            raise PydanticCustomError(
                "value_error",
                "pattern entries must be -1 (match negative), 0 (wildcard), "
                "or +1 (match positive)",
            )
        if not any(self.pattern):
            raise PydanticCustomError(
                "value_error",
                "pattern needs at least one non-wildcard (+-1) entry",
            )
        check_match_tolerance(
            self.max_errors, self.compared_positions, field="max_errors"
        )
        return self


class SyncSymbols(CodingStage[SyncSymbolsStep]):
    name = "sync_symbols"
    description = (
        "Correlate a +-1 sign pattern against soft symbols and report each "
        "hit as a MARK (run_rx's 'marks'), REPLACING any marks carried in. "
        "Marks survive the symbols->bits stage (m_slice+symbol_map, or "
        "slice), and mark_frame then turns them into windows THERE, at bits "
        "level - mark_frame directly after this stage does not compile "
        "(levels differ). Unlike sync_word this contributes NO quality "
        "evidence: it has no chance model, so its hits earn the path no "
        "credit and the verdict stays 'uncertain' on its own."
    )
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    step_model = SyncSymbolsStep
    accepts_item_type = ItemType.F
    marks_are_search_hits = True

    def emit_rx(self, b: CodingBuilder, step: SyncSymbolsStep) -> None:
        b.add(
            ops_symbols.sync_symbols_rx,
            pattern=[int(x) for x in step.pattern],
            max_errors=step.max_errors,
            pre_symbols=step.pre_symbols,
        )


class NormalizeStep(Step):
    conv: Literal["normalize"] = "normalize"
    span_symbols: StrictInt = Field(ge=1, le=MAX_WINDOW_SYMBOLS)
    dc: DcMode = DcMode.MEDIAN
    gain_percentile: float | None = Field(default=None, ge=0.0, le=100.0)


class Normalize(CodingStage[NormalizeStep]):
    name = "normalize"
    description = (
        "Per-window DC removal and gain normalization of soft symbols, scoped by "
        "marks (from sync_symbols or burst detection) - flattens per-burst "
        "offset/gain before m_slice thresholds."
    )
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    step_model = NormalizeStep
    accepts_item_type = ItemType.F

    def emit_rx(self, b: CodingBuilder, step: NormalizeStep) -> None:
        b.add(
            ops_symbols.normalize_rx,
            span_symbols=step.span_symbols,
            dc=step.dc,
            gain_percentile=step.gain_percentile,
        )


class MSliceStep(Step):
    conv: Literal["m_slice"] = "m_slice"
    thresholds: list[float]
    levels: list[int]

    @model_validator(mode="after")
    def _shaped(self) -> "MSliceStep":
        if len(self.levels) != len(self.thresholds) + 1:
            raise PydanticCustomError(
                "value_error", "levels must be one longer than thresholds"
            )
        if any(b <= a for a, b in zip(self.thresholds, self.thresholds[1:])):
            raise PydanticCustomError(
                "value_error", "thresholds must be strictly ascending"
            )
        # The symbol wire is int16 (ItemType.S). Unchecked, the cast in
        # ops_symbols.m_slice_rx raised numpy's own "Python integer 40000 out
        # of bounds for int16" from inside the coding lane — naming neither the
        # stage nor the field — on a spec validate_modem had called valid.
        bad = [v for v in self.levels if not _INT16_MIN <= v <= _INT16_MAX]
        if bad:
            raise PydanticCustomError(
                "value_error",
                "levels ride the int16 symbol wire and must lie in "
                "[{lo}, {hi}]; got {bad}",
                {"lo": _INT16_MIN, "hi": _INT16_MAX, "bad": bad[:5]},
            )
        return self


class MSlice(CodingStage[MSliceStep]):
    name = "m_slice"
    description = (
        "Threshold soft symbols into M hard levels: thresholds= the decision "
        "boundaries, levels= the emitted symbol VALUES (label them "
        "0..2^code_bits-1 for symbol_map). The 4-FSK/multi-level entry to "
        "symbol_map."
    )
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "symbols"
    step_model = MSliceStep
    accepts_item_type = ItemType.F

    def emit_rx(self, b: CodingBuilder, step: MSliceStep) -> None:
        b.add(
            ops_symbols.m_slice_rx,
            thresholds=[float(x) for x in step.thresholds],
            levels=[int(x) for x in step.levels],
        )

    def out_descriptor(self, in_desc: Descriptor, step: MSliceStep) -> Descriptor:
        return replace(
            in_desc,
            item_type=ItemType.S,
            carrier=Carrier.HARD,
            order=len(step.levels),
        )

    def required_input_order(self, step: MSliceStep) -> int | None:
        return len(step.levels)


class SymbolMapStep(Step):
    conv: Literal["symbol_map"] = "symbol_map"
    code_bits: StrictInt = Field(ge=1)
    data_bits: StrictInt = Field(ge=1)
    table: list[int]
    decode: DecodeMode = DecodeMode.EXACT

    @model_validator(mode="after")
    def _sized(self) -> "SymbolMapStep":
        check_codebook_sizing(self.code_bits, self.data_bits, self.table, self.decode)
        return self


class SymbolMap(CodingStage[SymbolMapStep]):
    name = "symbol_map"
    description = (
        "Hard M-ary symbols to bits via a codeword table (table[d] = codeword for "
        "data value d): the dibit/tribit mapping stage after m_slice."
    )
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "coding"
    step_model = SymbolMapStep
    accepts_item_type = ItemType.S
    accepts_carrier = Carrier.HARD

    def emit_rx(self, b: CodingBuilder, step: SymbolMapStep) -> None:
        b.add(ops_bits.codebook_rx, **_codebook_kw(step), symbol_input=True)

    def required_input_order(self, step: SymbolMapStep) -> int | None:
        # the one SYMBOLS->BITS demap that declared no alphabet: validate_modem
        # blessed m_slice(levels=[0..3]) -> symbol_map(code_bits=1) and the
        # worker threw "symbol values must lie in [0, 2)" at run time
        return 1 << int(step.code_bits)

    def out_descriptor(self, in_desc: Descriptor, step: SymbolMapStep) -> Descriptor:
        frame = (
            None
            if in_desc.frame_len is None
            else in_desc.frame_len * int(step.data_bits)
        )
        return Descriptor(
            Level.BITS, ItemType.B, Carrier.HARD, Amplitude.UNKNOWN, frame_len=frame
        )


CODING_SYMBOL_STAGES: tuple[type[Stage[CodingBuilder, Any]], ...] = (
    SyncSymbols,
    Normalize,
    MSlice,
    SymbolMap,
)
