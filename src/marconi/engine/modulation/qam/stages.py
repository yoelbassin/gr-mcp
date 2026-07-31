from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import StrictInt

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, Stage
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType, QamOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class QamDemodStep(Step):
    conv: Literal["qam_demod"] = "qam_demod"
    order: QamOrder  # required: an explicit QAM order, no silent default
    alpha: float = 0.35
    loop_bw: float = 0.045  # symbol_sync timing loop
    span: StrictInt = 11


class QamDemod(DuplexStage[CompileContext, QamDemodStep]):
    """QAM demod, IQ<->SYMBOLS. QAM's multi-radius constellation defeats a
    phase-only costas loop, so carrier recovery uses the constellation-aware,
    decision-directed `constellation_receiver_cb` ("costas for an arbitrary
    constellation"). Because it is decision-directed, its output IS the hard
    symbol decision (an index byte) — QAM cannot hand off soft complex symbols
    the way PSK does, so the seam at SYMBOLS is HARD (byte symbol indices). Soft
    QAM is deferred to the soft-seam (bits) design. RX: RRC matched filter +
    Gardner timing + constellation_receiver_cb -> hard symbol indices. TX: map
    indices to constellation points + RRC pulse-shape. rate_factor stays 1.0
    (symbol decimation is internal via ctx.sps; the modulator reads the IQ rate).
    """

    name = "qam_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "qam"
    # MEASURED across a 1e-3..1e3 gain sweep: peak-normalized input is SER
    # 0.91 at every gain and mean-mag is 0.87/0.00/0.92 (invariant only at
    # unity). A fixed-radius constellation needs the RMS statistic.
    accepts_amplitude = frozenset({Amplitude.RMS_UNITY})
    min_input_sps = 2.0
    step_model = QamDemodStep

    def emit_rx(self, b: CompileContext, step: QamDemodStep) -> None:
        b.chain(
            "rrc_filter_ccf",
            interpolation=1,
            rate=b.rate,
            sps=b.sps,
            alpha=step.alpha,
            span=step.span,
        )
        b.chain("symbol_sync_cc", sps=b.sps, loop_bw=step.loop_bw)
        b.chain("constellation_receiver_cb", scheme="qam", order=int(step.order))

    def emit_tx(self, b: CompileContext, step: QamDemodStep) -> None:
        b.chain("chunks_to_symbols_bc", scheme="qam", order=int(step.order))
        b.chain(
            "rrc_filter_ccf",
            interpolation=b.sps_int(),
            rate=b.rate,
            sps=b.sps,
            alpha=step.alpha,
            span=step.span,
        )

    def out_descriptor(self, in_desc: Descriptor, step: QamDemodStep) -> Descriptor:
        return Descriptor(
            Level.SYMBOLS, ItemType.B, Carrier.HARD, order=int(step.order)
        )


class QamDemapStep(Step):
    conv: Literal["qam_demap"] = "qam_demap"
    order: QamOrder  # required


class QamDemap(DuplexStage[CompileContext, QamDemapStep]):
    """Symbol-index <-> bits, SYMBOLS<->BITS. The constellation (Gray map) lives
    in the demod (constellation_receiver_cb decides the index on RX;
    chunks_to_symbols maps it back on TX), so the demap is a pure pack/unpack of
    the k-bit symbol index. RX: unpack index byte to bits. TX: pack bits into an
    index byte. Hard bits at the seam (mirrors the general Slice)."""

    name = "qam_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "qam"
    step_model = QamDemapStep
    accepts_item_type = ItemType.B
    accepts_carrier = Carrier.HARD

    def emit_rx(self, b: CompileContext, step: QamDemapStep) -> None:
        b.chain("unpack_k_bits_bb", k=int(math.log2(int(step.order))))

    def emit_tx(self, b: CompileContext, step: QamDemapStep) -> None:
        b.chain("pack_k_bits_bb", k=int(math.log2(int(step.order))))

    def out_descriptor(self, in_desc: Descriptor, step: QamDemapStep) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.B, Carrier.HARD)

    def required_input_order(self, step: QamDemapStep) -> int | None:
        return int(step.order)


QAM_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (QamDemod, QamDemap)
