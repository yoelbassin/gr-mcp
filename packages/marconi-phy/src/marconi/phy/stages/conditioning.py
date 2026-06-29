from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import RxStage
from marconi.phy.compile_context import CompileContext


class _ChannelizeParams(StageParams):
    decim: (
        StrictInt  # required & explicit: rate_factor (1/decim) must be params-derivable
    )
    bandwidth_hz: float  # required: signal bandwidth -> LPF cutoff/transition
    center_hz: float = 0.0  # offset of the sub-band from the capture centre

    @model_validator(mode="after")
    def _ok(self) -> "_ChannelizeParams":
        if self.decim < 1:
            raise PydanticCustomError("value_error", "decim must be >= 1")
        if self.bandwidth_hz <= 0:
            raise PydanticCustomError("value_error", "bandwidth_hz must be > 0")
        return self


class _InvertParams(StageParams):
    pass


class Channelize(RxStage[CompileContext]):
    """Extract a sub-band: frequency-shift center_hz to baseband, low-pass, and
    decimate by `decim` (one freq_xlating_fir_filter). RX-only conditioning. The
    cutoff/transition are derived from bandwidth_hz; `decim` is explicit because
    rate_factor (1/decim) is computed from params alone. The freq_xlating reads
    the INPUT rate (b.rate); the compiler hands the decimated rate to downstream
    stages via rate_factor.

    NOTE: bandwidth_hz is the filter PASSBAND width (cutoff = bandwidth_hz/2), not
    the signal's occupied bandwidth. A signal that fills its band edge-to-edge --
    notably a CSS chirp sweeping +/-B/2 -- must be given bandwidth_hz >= ~2x its
    bandwidth, or the band edges are clipped (for chirps: a silent, progressive
    decode drift, not an error)."""

    name = "channelize"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    params_model = _ChannelizeParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        bandwidth = float(params["bandwidth_hz"])
        b.chain(
            "freq_xlating_fir_filter_ccf",
            decim=int(params["decim"]),
            center=float(params.get("center_hz", 0.0)),
            cutoff=bandwidth / 2.0,
            transition=bandwidth / 2.0,
            rate=b.rate,
        )

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        return 1.0 / int(params["decim"])


class Invert(RxStage[CompileContext]):
    """Spectral conjugate (un-mirror a flipped capture, e.g. SDR# .wav). RX-only,
    IQ->IQ, rate unchanged."""

    name = "invert"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    params_model = _InvertParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("conjugate_cc")


class _ResampleParams(StageParams):
    interpolation: StrictInt
    decimation: StrictInt

    @model_validator(mode="after")
    def _ok(self) -> "_ResampleParams":
        if self.interpolation < 1:
            raise PydanticCustomError("value_error", "interpolation must be >= 1")
        if self.decimation < 1:
            raise PydanticCustomError("value_error", "decimation must be >= 1")
        return self


class Resample(RxStage[CompileContext]):
    """Rational resample by interpolation/decimation via the polyphase arbitrary
    resampler. The first non-unity rate_factor that is not 1/decim: lands an
    arbitrary capture rate exactly on a target (CSS needs oversample*bandwidth).
    RX-only conditioning."""

    name = "resample"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    params_model = _ResampleParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "pfb_arb_resampler_ccf",
            rate=int(params["interpolation"]) / int(params["decimation"]),
        )

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        return int(params["interpolation"]) / int(params["decimation"])


class _ClockCorrectParams(StageParams):
    ppm: float  # transmitter clock offset in parts-per-million (signed)


class ClockCorrect(RxStage[CompileContext]):
    """Cancel sampling-frequency offset (transmitter clock drift) by resampling
    by 1/(1 + ppm*1e-6) via the polyphase arbitrary resampler. RX-only. SFO is a
    silent no-op below ~1 chip of cumulative drift (ppm * n_samples), so it only
    matters on long frames (e.g. the multi-second IQ_12 LoRa capture)."""

    name = "clock_correct"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    params_model = _ClockCorrectParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("pfb_arb_resampler_ccf", rate=1.0 / (1.0 + float(params["ppm"]) * 1e-6))

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        return 1.0 / (1.0 + float(params["ppm"]) * 1e-6)


CONDITIONING_STAGES: tuple[type[RxStage[CompileContext]], ...] = (
    Channelize,
    Invert,
    Resample,
    ClockCorrect,
)
