from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from marconi.engine.types.bounds import check_sample_rate
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step, steps_from_spec

# The rungs a float32 soft stream can sit on. Level owns the vocabulary; this
# names the subset a Softstream may claim, so the sign convention keyed off it
# (marconi.mcp.payload) has one home and cannot be handed a rung with no rule.
SOFT_LEVELS: frozenset[Level] = frozenset({Level.SYMBOLS, Level.BITS})


class ValidationIssue(BaseModel):
    block_id: str | None = None
    field: str | None = None
    message: str


class Bitstream(BaseModel):
    path: Path
    num_bits: int = Field(ge=0)


class Softstream(BaseModel):
    """The soft float32 stream a decode exposes for soft-decision work, wherever
    it lives: a demod tap sidecar, a soft coding seam, or the terminal stream
    itself. Sign follows level: symbols-level values slice POSITIVE to bit 1,
    bits-level values are LLRs where bit 1 is NEGATIVE."""

    path: Path
    num_items: int = Field(ge=0)
    level: Level

    @model_validator(mode="after")
    def _level_is_soft_bearing(self) -> "Softstream":
        if self.level not in SOFT_LEVELS:
            raise ValueError(
                f"a soft stream sits at {sorted(SOFT_LEVELS)}, not {self.level!r}"
            )
        return self


class Symbolstream(BaseModel):
    """A non-bit output wire: hard int16 symbol indices for "s", soft float32
    values for "f", complex64 constellation points for "c" — one item per
    stream position, plus burst-start marks (item offsets) when acquisition
    tagged them. Sibling to Bitstream, never an optional field on it.

    `level` is the rung those items sit on and is NOT always symbols: a path
    ending at a soft demap emits bits-level LLRs on the same "f" wire, and a
    path ending before any demod emits conditioned IQ on the "c" wire. Carrying
    the rung is what lets num_symbols mean "items" honestly and lets the wire
    say which sign convention an "f" stream follows, instead of the reader
    inferring the rung from a separate soft_stream field."""

    path: Path
    num_symbols: int = Field(ge=0)
    item_type: ItemType = ItemType.S
    level: Level = Level.SYMBOLS
    marks: list[int] = []

    @model_validator(mode="after")
    def _item_type_is_carryable(self) -> "Symbolstream":
        if self.item_type is ItemType.B:
            raise ValueError(
                "a bit wire is a Bitstream, not a Symbolstream with item_type 'b'"
            )
        return self

    @model_validator(mode="after")
    def _marks_are_stream_positions(self) -> "Symbolstream":
        prev = -1
        for m in self.marks:
            if m <= prev or m >= self.num_symbols:
                raise ValueError(
                    "marks must be strictly increasing offsets in "
                    f"[0, num_symbols={self.num_symbols}); got {self.marks}"
                )
            prev = m
        return self


class Modem(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="never")
    name: str = "modem"
    symbol_rate: float = Field(gt=0)
    path: list[SerializeAsAny[Step]]

    @model_validator(mode="after")
    def _symbol_rate_is_finite(self) -> "Modem":
        """gt=0 admits +inf (inf > 0 is True), and json.loads("1e999") is how
        an agent types it. The Step base's finiteness sweep cannot reach this
        one: symbol_rate is spec-level, not a step field."""
        check_sample_rate(self.symbol_rate, field="symbol_rate")
        return self

    def to_spec(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_spec(
        cls, data: Mapping[str, object], step_models: Mapping[str, type[Step]]
    ) -> "Modem":
        # extra="forbid" above only guards the constructor; this reader picked
        # three keys out of the mapping by hand and dropped the rest, so a
        # misspelled 'path' produced an EMPTY path that compiled and reported
        # valid — the one typo the fast iteration loop exists to catch.
        unknown = sorted(set(data) - set(cls.model_fields))
        if unknown:
            raise ValueError(
                f"modem spec has unknown key(s) {unknown}; a spec carries only "
                f"{sorted(cls.model_fields)}. Note sample_rate is an argument "
                "to validate_modem/run_rx, not part of the spec"
            )
        raw = data.get("path")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("modem spec 'path' must be a list of steps")
        rate = data.get("symbol_rate")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            raise ValueError("modem spec requires a numeric 'symbol_rate'")
        steps = steps_from_spec(raw, step_models)
        return cls(
            name=str(data.get("name", "modem")),
            symbol_rate=float(rate),
            path=steps,
        )
