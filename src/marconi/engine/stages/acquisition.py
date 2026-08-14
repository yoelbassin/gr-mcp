from __future__ import annotations

import cmath
import math
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import Stage
from marconi.engine.types.bounds import MAX_FILTER_TAPS, MAX_FRAME_ITEMS
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step

_PREAMBLE_MODULUS_RTOL = 1e-3
_PREAMBLE_GRID_ATOL = 0.05


class PreambleSyncStep(Step):
    conv: Literal["preamble_sync"] = "preamble_sync"
    preamble_i: list[float]
    preamble_q: list[float]
    pad_symbols: StrictInt = Field(default=192, ge=0, le=MAX_FRAME_ITEMS)
    threshold: float = Field(
        default=0.9,
        gt=0,
        le=1.0,
        description=(
            "Detection bar as a FRACTION of the ideal preamble's own "
            "autocorrelation energy (corr_est runs THRESHOLD_ABSOLUTE). The "
            "(0, 1] domain assumes the contract this stage requires: with the "
            "stream normalized upstream (agc) so the preamble arrives near "
            "unit power, a perfect match reaches 1 and cannot exceed it. "
            "Un-normalized it is the GAIN that moves, not the useful "
            "threshold - a 10x hot stream fires on noise at any bar in this "
            "range, so raising the threshold is never the fix for that. "
            "Already lossy AT 1 in noise; 0.9 is the default for that reason."
        ),
    )
    # True: the preamble is drawn from the payload constellation, so the
    # modulus/phase-grid typo check applies. False: a freeform training
    # sequence (CAZAC/Zadoff-Chu, mixed-amplitude) - correlation and phase
    # recovery work the same, only the grid check is skipped.
    constellation_preamble: bool = True

    @model_validator(mode="after")
    def _ok(self) -> "PreambleSyncStep":
        if len(self.preamble_i) != len(self.preamble_q):
            raise PydanticCustomError(
                "value_error", "preamble_i and preamble_q must have equal length"
            )
        if not self.preamble_i:
            raise PydanticCustomError("value_error", "preamble must be non-empty")
        return self


class PreambleSync(Stage[CompileContext, PreambleSyncStep]):
    """Coherent acquisition, SYMBOLS->SYMBOLS. Correlates recovered soft
    symbols against a known preamble (a phy training sequence, never seen at
    BITS) to recover timing + the M-fold carrier-phase rotation, derotates, and
    strips to payload-only.
    out_descriptor/rate_factor inherit defaults: level/type/carrier unchanged,
    rate 1.0 (stripping changes symbol count, not rate)."""

    name = "preamble_sync"
    description = (
        "Correlate a known complex preamble (corr_est) and strip pad+preamble, "
        "derotating each burst by the estimated phase. Detection compares against "
        "the preamble's ABSOLUTE energy: normalize first (agc) so the threshold "
        "means what it says."
    )
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "acquisition"
    step_model = PreambleSyncStep
    # corr_est_cc + sym_strip are complex-only: reject real-float symbols (fsk)
    # at compile instead of dying on an itemsize mismatch in the backend.
    accepts_item_type = ItemType.C
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: PreambleSyncStep) -> None:
        b.chain(
            "corr_est_cc",
            preamble_i=step.preamble_i,
            preamble_q=step.preamble_q,
            sps=1,
            mark_delay=0,
            threshold=step.threshold,
        )
        b.chain("sym_strip", n_pre=len(step.preamble_i))

    def validate_input(self, in_desc: Descriptor, step: PreambleSyncStep) -> str | None:
        if in_desc.order is None or not step.constellation_preamble:
            return None
        m = in_desc.order
        points = [complex(i, q) for i, q in zip(step.preamble_i, step.preamble_q)]
        radii = [abs(z) for z in points]
        if min(radii) == 0.0:
            return (
                "preamble contains a zero point; constellation points "
                "have nonzero modulus"
            )
        spread = (max(radii) - min(radii)) / max(radii)
        if spread > _PREAMBLE_MODULUS_RTOL:
            return (
                f"preamble moduli spread {spread:.3g} exceeds "
                f"{_PREAMBLE_MODULUS_RTOL}; an order-{m} constellation has "
                "one radius"
            )
        ref = points[0].conjugate()
        for k, z in enumerate(points[1:], start=1):
            err = math.remainder(m * cmath.phase(z * ref), 2.0 * math.pi)
            if abs(err) > _PREAMBLE_GRID_ATOL:
                return (
                    f"preamble point {k} sits {abs(err):.3g} rad (x{m}) off "
                    f"the order-{m} phase grid relative to point 0"
                )
        return None


class FllStep(Step):
    conv: Literal["fll"] = "fll"
    rolloff: float = Field(
        default=0.35,
        description=(
            "Excess-bandwidth factor; must match the transmitter's pulse "
            "shaping — the band-edge filters are built from it, so a wrong "
            "rolloff mistunes them."
        ),
    )
    filter_size: StrictInt = Field(default=44, ge=1, le=MAX_FILTER_TAPS)
    loop_bw: float = Field(default=0.03, gt=0.0)

    @model_validator(mode="after")
    def _ok(self) -> "FllStep":
        if not (0.0 < self.rolloff <= 1.0):
            raise PydanticCustomError("value_error", "rolloff must be in (0, 1]")
        return self


class Fll(Stage[CompileContext, FllStep]):
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
    description = (
        "Band-edge frequency-locked loop: pulls a residual carrier offset to ~0 "
        "before a demod whose own loop cannot span it. For offsets beyond a few "
        "percent of the symbol rate, translate first instead."
    )
    from_level = Level.IQ
    to_level = Level.IQ
    family = "acquisition"
    step_model = FllStep

    def emit_rx(self, b: CompileContext, step: FllStep) -> None:
        b.chain(
            "fll_band_edge_cc",
            sps=b.sps,
            rolloff=step.rolloff,
            filter_size=step.filter_size,
            loop_bw=step.loop_bw,
        )


ACQUISITION_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (PreambleSync, Fll)
