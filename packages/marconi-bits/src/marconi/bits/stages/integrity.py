from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import model_validator

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
        payload_bits: int | None = None

        @model_validator(mode="after")
        def _widths(self) -> "Crc._Params":
            if self.bits < 1:
                raise ValueError("crc bits must be >= 1")
            if self.bits % 8:
                if self.payload_bits is None:
                    raise ValueError(
                        "sub-byte crc bits requires payload_bits (the frame's "
                        "true bit extent before byte padding)"
                    )
                if (
                    self.reflected
                    or self.fold_tail
                    or self.checksum_le
                    or self.bit_order != "msb"
                ):
                    raise ValueError(
                        "sub-byte crc supports msb bit_order only, without "
                        "reflected/fold_tail/checksum_le"
                    )
            elif self.payload_bits is not None:
                raise ValueError("payload_bits is only for sub-byte crc widths")
            return self

    params_model = _Params

    def _args(self, p: "Crc._Params") -> dict[str, Any]:
        return {
            "poly": p.poly,
            "bits": p.bits,
            "init": p.init,
            "reflected": p.reflected,
            "xorout": p.xorout,
            "bit_order": p.bit_order,
            "fold_tail": p.fold_tail,
            "checksum_le": p.checksum_le,
        }

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(framing.crc_rx, **self._args(p), payload_bits=p.payload_bits)

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        p = self._Params.model_validate(dict(params))
        b.add(framing.crc_tx, **self._args(p))


INTEGRITY_STAGES: tuple[type[DuplexStage[ProgramBuilder]], ...] = (Crc,)
