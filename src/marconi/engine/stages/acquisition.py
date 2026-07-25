from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, RxStage, Stage
from marconi.engine.types.descriptor import Carrier
from marconi.engine.types.levels import Level
from marconi.engine.types.params import StageParams


class _PreambleSyncParams(StageParams):
    preamble_i: list[float]
    preamble_q: list[float]
    pad_symbols: StrictInt = 192
    threshold: float = 0.9

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
    # corr_est_cc + sym_strip are complex-only: reject real-float symbols (fsk)
    # at compile instead of dying on an itemsize mismatch in the backend.
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _PreambleSyncParams.model_validate(dict(params))
        b.chain(
            "corr_est_cc",
            preamble_i=p.preamble_i,
            preamble_q=p.preamble_q,
            sps=1,
            mark_delay=0,
            threshold=p.threshold,
        )
        b.chain("sym_strip", n_pre=len(p.preamble_i))

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _PreambleSyncParams.model_validate(dict(params))
        b.chain(
            "sym_prepend",
            preamble_i=p.preamble_i,
            preamble_q=p.preamble_q,
            pad_symbols=p.pad_symbols,
        )


class _FllParams(StageParams):
    # Must match the pulse shaping the transmitter used: the band-edge filters
    # are built from the excess bandwidth, so a wrong rolloff mistunes them.
    rolloff: float = 0.35
    filter_size: StrictInt = 44
    loop_bw: float = 0.03

    @model_validator(mode="after")
    def _ok(self) -> "_FllParams":
        if not (0.0 < self.rolloff <= 1.0):
            raise PydanticCustomError("value_error", "rolloff must be in (0, 1]")
        if self.filter_size < 1:
            raise PydanticCustomError("value_error", "filter_size must be >= 1")
        if self.loop_bw <= 0.0:
            raise PydanticCustomError("value_error", "loop_bw must be > 0")
        return self


class Fll(RxStage[CompileContext]):
    """Coarse carrier-frequency acquisition, IQ->IQ (stock fll_band_edge_cc).

    A costas/decision-directed loop only pulls in a fraction of its own loop
    bandwidth, so a real capture's frequency offset -- a 2 ppm TCXO at 900 MHz
    is ~1.8 kHz, 19% of symbol rate on a 9600-baud link -- sits far outside it.
    This band-edge FLL acquires the offset first, from the pulse shaping's
    excess-bandwidth edges rather than from data decisions, so it needs no
    modulation knowledge beyond the rolloff. Runs on the pulse-shaped signal,
    i.e. BEFORE the demod's matched filter. Rotation only, so it preserves
    whatever amplitude statistic it was handed."""

    name = "fll"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "acquisition"
    params_model = _FllParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _FllParams.model_validate(dict(params))
        b.chain(
            "fll_band_edge_cc",
            sps=b.sps,
            rolloff=p.rolloff,
            filter_size=p.filter_size,
            loop_bw=p.loop_bw,
        )


ACQUISITION_STAGES: tuple[type[Stage[CompileContext]], ...] = (PreambleSync, Fll)
