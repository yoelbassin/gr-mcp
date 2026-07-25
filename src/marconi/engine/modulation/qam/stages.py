from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.params import StageParams

_QAM_ORDERS = (16, 64)


def _check_order(order: int) -> None:
    if order not in _QAM_ORDERS:
        raise PydanticCustomError(
            "value_error",
            "qam order {o} unsupported (allowed: {allowed})",
            {"o": order, "allowed": list(_QAM_ORDERS), "field": "order"},
        )


class _QamDemodParams(StageParams):
    order: StrictInt  # required: an explicit QAM order, no silent default
    alpha: float = 0.35
    loop_bw: float = 0.045  # symbol_sync timing loop
    span: StrictInt = 11

    @model_validator(mode="after")
    def _ok(self) -> "_QamDemodParams":
        _check_order(self.order)
        return self


class _QamDemapParams(StageParams):
    order: StrictInt  # required

    @model_validator(mode="after")
    def _ok(self) -> "_QamDemapParams":
        _check_order(self.order)
        return self


class QamDemod(DuplexStage[CompileContext]):
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
    params_model = _QamDemodParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _QamDemodParams.model_validate(dict(params))
        b.chain(
            "rrc_filter_ccf",
            interpolation=1,
            rate=b.rate,
            sps=b.sps,
            alpha=p.alpha,
            span=p.span,
        )
        b.chain("symbol_sync_cc", sps=b.sps, loop_bw=p.loop_bw)
        b.chain("constellation_receiver_cb", scheme="qam", order=p.order)

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _QamDemodParams.model_validate(dict(params))
        b.chain("chunks_to_symbols_bc", scheme="qam", order=p.order)
        b.chain(
            "rrc_filter_ccf",
            interpolation=b.sps_int(),
            rate=b.rate,
            sps=b.sps,
            alpha=p.alpha,
            span=p.span,
        )

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "b", Carrier.HARD, order=int(params["order"]))


class QamDemap(DuplexStage[CompileContext]):
    """Symbol-index <-> bits, SYMBOLS<->BITS. The constellation (Gray map) lives
    in the demod (constellation_receiver_cb decides the index on RX;
    chunks_to_symbols maps it back on TX), so the demap is a pure pack/unpack of
    the k-bit symbol index. RX: unpack index byte to bits. TX: pack bits into an
    index byte. Hard bits at the seam (mirrors the general Slice)."""

    name = "qam_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "qam"
    params_model = _QamDemapParams
    accepts_item_type = "b"
    accepts_carrier = Carrier.HARD

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("unpack_k_bits_bb", k=int(math.log2(int(params["order"]))))

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("pack_k_bits_bb", k=int(math.log2(int(params["order"]))))

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", Carrier.HARD)


QAM_STAGES: tuple[type[DuplexStage[CompileContext]], ...] = (QamDemod, QamDemap)
