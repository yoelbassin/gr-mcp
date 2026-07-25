from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

ParamValue = float | int | str | bool | list[float | int]

OPEN_LOOP: Final[float] = 0.0


class StageParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
