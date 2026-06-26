from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from marconi.core.params import ParamValue


class ModemStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conv: str
    params: dict[str, ParamValue] = {}


class ModemSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "modem"
    symbol_rate: float = Field(gt=0)
    path: list[ModemStep]
