from __future__ import annotations

from pydantic import BaseModel, ConfigDict

ParamValue = float | int | str | bool | list[float | int]


class StageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
