from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import field_validator

from marconi.bits import framing
from marconi.bits.builder import ProgramBuilder
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage


class Crc(DuplexStage[ProgramBuilder]):
    name = "crc"
    from_level = Level.FRAMES
    to_level = Level.FRAMES
    family = "integrity"

    class _Params(StageParams):
        poly: int
        bits: int
        init: int = 0
        reflected: bool = False
        xorout: int = 0
        bit_order: str = "msb"
        fold_tail: int = 0
        checksum_le: bool = False

        @field_validator("bits")
        @classmethod
        def _mult8(cls, v: int) -> int:
            if v % 8 != 0:
                raise ValueError("crc bits must be a multiple of 8")
            return v

    params_model = _Params

    def _args(self, params: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "poly": int(params["poly"]),
            "bits": int(params["bits"]),
            "init": int(params.get("init", 0)),
            "reflected": bool(params.get("reflected", False)),
            "xorout": int(params.get("xorout", 0)),
            "bit_order": str(params.get("bit_order", "msb")),
            "fold_tail": int(params.get("fold_tail", 0)),
            "checksum_le": bool(params.get("checksum_le", False)),
        }

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.crc_rx, **self._args(params))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.crc_tx, **self._args(params))


INTEGRITY_STAGES: tuple[type[DuplexStage[ProgramBuilder]], ...] = (Crc,)
