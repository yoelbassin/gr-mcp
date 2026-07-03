from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage
from marconi.phy.compile_context import CompileContext

_PSK_ORDERS = (2, 4, 8)


def _check_order(order: int) -> None:
    if order not in _PSK_ORDERS:
        raise PydanticCustomError(
            "value_error",
            "psk order {o} unsupported (allowed: {allowed})",
            {"o": order, "allowed": list(_PSK_ORDERS), "field": "order"},
        )


class _PskDemodParams(StageParams):
    order: StrictInt  # required: an explicit PSK order, no silent default modulation
    alpha: float = 0.35
    loop_bw: float = 0.045
    span: StrictInt = 11

    @model_validator(mode="after")
    def _ok(self) -> "_PskDemodParams":
        _check_order(self.order)
        return self


class _PskDemapParams(StageParams):
    order: StrictInt  # required

    @model_validator(mode="after")
    def _ok(self) -> "_PskDemapParams":
        _check_order(self.order)
        return self


class PskDemod(DuplexStage[CompileContext]):
    """Coherent linear demod, IQ<->SYMBOLS. RX: RRC matched filter + Gardner
    symbol timing + costas carrier recovery -> soft complex symbols. TX: RRC
    pulse-shape/upsample. rate_factor stays 1.0 (symbol decimation is internal
    via ctx.sps; the modulator reads the IQ-domain rate)."""

    name = "psk_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "psk"
    params_model = _PskDemodParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        alpha = float(params.get("alpha", 0.35))
        span = int(params.get("span", 11))
        b.chain(
            "rrc_filter_ccf",
            interpolation=1,
            rate=b.rate,
            sps=b.sps,
            alpha=alpha,
            span=span,
        )
        b.chain(
            "symbol_sync_cc", sps=b.sps, loop_bw=float(params.get("loop_bw", 0.045))
        )
        b.chain(
            "costas_loop_cc",
            order=int(params["order"]),
            loop_bw=float(params.get("loop_bw", 0.045)),
        )

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        alpha = float(params.get("alpha", 0.35))
        span = int(params.get("span", 11))
        b.chain(
            "rrc_filter_ccf",
            interpolation=b.sps_int(),
            rate=b.rate,
            sps=b.sps,
            alpha=alpha,
            span=span,
        )

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "c", in_desc.layout, Carrier.SOFT)


class PskDemap(DuplexStage[CompileContext]):
    """Constellation map/demap, SYMBOLS<->BITS. RX: hard min-distance decode +
    unpack to bits. TX: pack bits into symbol indices + map to constellation
    points. Hard bits at the seam (mirrors the general Slice)."""

    name = "psk_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "psk"
    params_model = _PskDemapParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        order = int(params["order"])
        b.chain("constellation_decoder_cb", scheme="psk", order=order)
        b.chain("unpack_k_bits_bb", k=int(math.log2(order)))

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        order = int(params["order"])
        b.chain("pack_k_bits_bb", k=int(math.log2(order)))
        b.chain("chunks_to_symbols_bc", scheme="psk", order=order)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "b", in_desc.layout, Carrier.HARD)


PSK_STAGES: tuple[type[DuplexStage[CompileContext]], ...] = (PskDemod, PskDemap)
