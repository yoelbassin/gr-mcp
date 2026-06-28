from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage
from marconi.phy.compile_context import CompileContext


class _PreambleSyncParams(StageParams):
    preamble_i: list[float]
    preamble_q: list[float]
    pad_symbols: StrictInt = 192
    threshold: float = 3.0

    @model_validator(mode="after")
    def _ok(self) -> "_PreambleSyncParams":
        if len(self.preamble_i) != len(self.preamble_q):
            raise PydanticCustomError(
                "value_error", "preamble_i and preamble_q must have equal length"
            )
        if not self.preamble_i:
            raise PydanticCustomError("value_error", "preamble must be non-empty")
        if self.pad_symbols < 0:
            raise PydanticCustomError("value_error", "pad_symbols must be >= 0")
        if self.threshold <= 0:
            raise PydanticCustomError("value_error", "threshold must be > 0")
        return self


class PreambleSync(DuplexStage[CompileContext]):
    """Coherent acquisition, SYMBOLS<->SYMBOLS. TX prepends a settle ramp + a
    known sync preamble (a phy training sequence, never seen at BITS). RX
    correlates recovered soft symbols against the preamble to recover timing +
    the M-fold carrier-phase rotation, derotates, and strips to payload-only.
    out_descriptor/rate_factor inherit defaults: level/type/carrier unchanged,
    rate 1.0 (stripping changes symbol count, not rate)."""

    name = "preamble_sync"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "acquisition"
    params_model = _PreambleSyncParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "sym_acquire",
            preamble_i=list(params["preamble_i"]),
            preamble_q=list(params["preamble_q"]),
            pad_symbols=int(params.get("pad_symbols", 192)),
            threshold=float(params.get("threshold", 3.0)),
        )

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "sym_prepend",
            preamble_i=list(params["preamble_i"]),
            preamble_q=list(params["preamble_q"]),
            pad_symbols=int(params.get("pad_symbols", 192)),
        )


ACQUISITION_STAGES: tuple[type[DuplexStage[CompileContext]], ...] = (PreambleSync,)
