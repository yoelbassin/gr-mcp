from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import DuplexStage, RxStage, Stage
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.params import StageParams

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


class _SymbolSyncParams(StageParams):
    sps: StrictInt  # explicit: rate_factor (1/sps) must be params-derivable
    alpha: float = 0.35
    loop_bw: float = 0.045
    span: StrictInt = 11

    @model_validator(mode="after")
    def _ok(self) -> "_SymbolSyncParams":
        if self.sps < 2:
            raise PydanticCustomError(
                "value_error", "sps must be >= 2 to recover symbol timing"
            )
        if not 0.0 < self.alpha <= 1.0:
            raise PydanticCustomError("value_error", "alpha must be in (0, 1]")
        if self.loop_bw <= 0.0:
            raise PydanticCustomError("value_error", "loop_bw must be > 0")
        if self.span < 1:
            raise PydanticCustomError("value_error", "span must be >= 1")
        return self


class SymbolSync(RxStage[CompileContext]):
    """Symbol-timing recovery, IQ->IQ: RRC matched filter + Gardner symbol_sync,
    decimating an oversampled stream to one sample per symbol. This is
    psk_demod's timing half split out, so a 1-sps seam opens where an equalizer
    (or any per-symbol conditioning) can sit before carrier recovery. Unlike
    psk_demod -- which hides its sps decimation inside the IQ->SYMBOLS level
    change -- this stays at IQ and decimates honestly: rate_factor is 1/sps and
    required_input_rate makes the explicit sps agree with the delivered rate.
    RX-only; output amplitude is UNKNOWN (the filters rescale)."""

    name = "symbol_sync"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "psk"
    accepts_amplitude = frozenset(
        {Amplitude.PEAK_UNITY, Amplitude.MEAN_MAG_UNITY, Amplitude.RMS_UNITY}
    )
    alters_amplitude = True
    params_model = _SymbolSyncParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _SymbolSyncParams.model_validate(dict(params))
        b.chain(
            "rrc_filter_ccf",
            interpolation=1,
            rate=b.rate,
            sps=p.sps,
            alpha=p.alpha,
            span=p.span,
        )
        b.chain("symbol_sync_cc", sps=p.sps, loop_bw=p.loop_bw)

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        return 1.0 / int(params["sps"])

    def required_input_rate(
        self, params: Mapping[str, Any], symbol_rate: float
    ) -> float | None:
        return int(params["sps"]) * symbol_rate


class SampleSymbols(RxStage[CompileContext]):
    """IQ->SYMBOLS at 1 sps with no carrier loop: the timing-only rung.
    Emits no blocks - the level move is the whole stage - so an externally
    recovered or differentially absorbed carrier can reach the symbol seam."""

    name = "sample_symbols"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "psk"

    class _Params(StageParams):
        pass

    params_model = _Params

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        pass

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "c", in_desc.layout, Carrier.SOFT)

    def required_input_rate(
        self, params: Mapping[str, Any], symbol_rate: float
    ) -> float | None:
        return symbol_rate


class _DifferentialDemodParams(StageParams):
    delay: StrictInt = Field(default=1, ge=1)
    rotate: float = 0.0


class DifferentialDemod(RxStage[CompileContext]):
    """SYMBOLS->SYMBOLS continuous differential: z[i] = x[i] * conj(x[i-delay]),
    optionally rotated by exp(j*rotate). With rotate=-pi/4 this lands pi/4-DQPSK
    on the plain QPSK constellation; delay > 1 serves symbol-interleaved
    differential schemes."""

    name = "differential_demod"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "psk"
    params_model = _DifferentialDemodParams
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _DifferentialDemodParams.model_validate(dict(params))
        src = b.tail
        assert src is not None
        dly = b.chain("delay_cc", samples=p.delay)
        mc = b.add("multiply_conjugate_cc")
        b.connect(src, mc, dst_port=0)
        b.connect(dly, mc, dst_port=1)
        b.set_tail(mc)
        if p.rotate:
            b.chain(
                "multiply_const_cc",
                re=math.cos(p.rotate),
                im=math.sin(p.rotate),
            )


class PskDemod(DuplexStage[CompileContext]):
    """Coherent linear demod, IQ<->SYMBOLS. RX: RRC matched filter + Gardner
    symbol timing + costas carrier recovery -> soft complex symbols. TX: RRC
    pulse-shape/upsample. rate_factor stays 1.0 (symbol decimation is internal
    via ctx.sps; the modulator reads the IQ-domain rate)."""

    name = "psk_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "psk"
    # MEASURED across a 1e-3..1e3 gain sweep: BER 0 for every statistic. A
    # constant-modulus constellation has no absolute radius to hit; the
    # requirement is only that SOME normalization bounded the scale.
    accepts_amplitude = frozenset(
        {Amplitude.PEAK_UNITY, Amplitude.MEAN_MAG_UNITY, Amplitude.RMS_UNITY}
    )
    params_model = _PskDemodParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _PskDemodParams.model_validate(dict(params))
        b.chain(
            "rrc_filter_ccf",
            interpolation=1,
            rate=b.rate,
            sps=b.sps,
            alpha=p.alpha,
            span=p.span,
        )
        b.chain("symbol_sync_cc", sps=b.sps, loop_bw=p.loop_bw)
        b.chain("costas_loop_cc", order=p.order, loop_bw=p.loop_bw)

    def emit_tx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _PskDemodParams.model_validate(dict(params))
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
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

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


class PskSoftDemap(RxStage[CompileContext]):
    """Constellation soft demap, SYMBOLS->BITS soft. RX-only: stock
    constellation_soft_decoder, one LLR per bit. The soft counterpart of
    psk_demap, and the single-carrier entry point to the soft coding lane
    (deinterleave/depuncture/fec), which until now only dqpsk_soft_demap could
    feed.

    MEASURED: constellation_soft_decoder emits log P(1)/P(0) -- BPSK point +1
    (hard bit 1) reads +4.0 -- while both soft consumers in this engine use the
    opposite sign (trellis_viterbi's Euclidean table maps encoder bit 0 to +1.0,
    and core.bitfile.harden is llr<0). Without the negation the lane decodes to
    noise on NOISELESS symbols (measured match 0.508); with it, BER 0."""

    name = "psk_soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "psk"
    params_model = _PskDemapParams
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _PskDemapParams.model_validate(dict(params))
        b.chain("constellation_soft_decoder", scheme="psk", order=p.order)
        b.chain("multiply_const_ff", value=-1.0)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "f", in_desc.layout, Carrier.SOFT)


PSK_STAGES: tuple[type[Stage[CompileContext]], ...] = (
    SymbolSync,
    SampleSymbols,
    DifferentialDemod,
    PskDemod,
    PskDemap,
    PskSoftDemap,
)
