from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from marconi.engine.types.step import Step, steps_from_spec


class ValidationIssue(BaseModel):
    block_id: str | None = None
    field: str | None = None
    message: str


CaptureDtype = Literal["cf32_le", "ci16_le", "cf64_le", "ci8"]


class CaptureRef(BaseModel):
    path: Path
    center_freq: float
    sample_rate: float = Field(gt=0)
    num_samples: int = Field(ge=0)
    datatype: CaptureDtype = "cf32_le"

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate


class Bitstream(BaseModel):
    path: Path
    num_bits: int = Field(ge=0)
    source_capture: Path | None = None
    symbol_rate: float | None = None


class Softstream(BaseModel):
    """The soft float32 stream a decode exposes for soft-decision work, wherever
    it lives: a demod tap sidecar, a soft coding seam, or the terminal stream
    itself. Sign follows level: symbols-level values slice POSITIVE to bit 1,
    bits-level values are LLRs where bit 1 is NEGATIVE."""

    path: Path
    num_items: int = Field(ge=0)
    level: Literal["symbols", "bits"]


class Symbolstream(BaseModel):
    """Values at the SYMBOLS rung: hard int16 symbol indices for "s", soft
    float32 symbol values for "f", complex64 constellation points for "c",
    one per symbol, plus burst-start marks (symbol offsets) when acquisition
    tagged them. Sibling to Bitstream, never an optional field on it."""

    path: Path
    num_symbols: int = Field(ge=0)
    item_type: Literal["s", "f", "c"] = "s"
    marks: list[int] = []
    source_capture: Path | None = None
    symbol_rate: float | None = None

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

    def to_spec(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_spec(
        cls, data: Mapping[str, object], step_models: Mapping[str, type[Step]]
    ) -> "Modem":
        raw = data.get("path", [])
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("modem spec 'path' must be a list of steps")
        rate = data.get("symbol_rate")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            raise ValueError("modem spec requires a numeric 'symbol_rate'")
        steps = steps_from_spec([dict(s) for s in raw], step_models)
        return cls(
            name=str(data.get("name", "modem")),
            symbol_rate=float(rate),
            path=steps,
        )
