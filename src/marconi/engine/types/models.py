from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marconi.engine.types.params import ParamValue


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


class SoftBitstream(BaseModel):
    """Soft bits at the BITS rung: a float32 file of one log-likelihood ratio per
    coded bit. Sibling to Bitstream, never an optional field on it."""

    path: Path
    num_bits: int = Field(ge=0)
    source_capture: Path | None = None
    symbol_rate: float | None = None


class Symbolstream(BaseModel):
    """Values at the SYMBOLS rung: hard int16 symbol indices for "s", soft
    float32 symbol values for "f", one per symbol, plus burst-start marks
    (symbol offsets) when acquisition tagged them. Sibling to Bitstream,
    never an optional field on it."""

    path: Path
    num_symbols: int = Field(ge=0)
    item_type: Literal["s", "f"] = "s"
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


class ModemStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conv: str
    params: dict[str, ParamValue] = {}


class ModemSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "modem"
    symbol_rate: float = Field(gt=0)
    path: list[ModemStep]
