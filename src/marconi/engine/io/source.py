from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from marconi.engine.types.params import ParamValue


@dataclass(frozen=True)
class SourceSlice:
    """A bounded region of a capture feeding a GR file source. offset/length
    are in stream items; length 0 means to-EOF."""

    path: Path
    offset: int = 0
    length: int = 0

    def to_params(self) -> dict[str, ParamValue]:
        params: dict[str, ParamValue] = {"path": str(self.path)}
        if self.offset:
            params["offset"] = self.offset
        if self.length:
            params["length"] = self.length
        return params

    @classmethod
    def from_params(cls, params: Mapping[str, ParamValue]) -> "SourceSlice":
        path = params.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"source params need a 'path' string, got {path!r}")
        offset = params.get("offset", 0)
        length = params.get("length", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"source offset must be an int >= 0, got {offset!r}")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError(f"source length must be an int >= 0, got {length!r}")
        return cls(path=Path(path), offset=offset, length=length)
