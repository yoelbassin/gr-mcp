from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import RxStage
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.params import StageParams


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
    alters_amplitude = True

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _ChannelizeParams.model_validate(dict(params))
        b.chain(
            "freq_xlating_fir_filter_ccf",
            decim=p.decim,
            center=p.center_hz,
            cutoff=p.bandwidth_hz / 2.0,
            transition=p.bandwidth_hz / 2.0,
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
    """Rational resample by interpolation/decimation via rational_resampler_ccf
    (integer ratio, auto-designed anti-imaging taps). The first non-unity
    rate_factor that is not 1/decim: lands an arbitrary capture rate exactly on a
    target (CSS needs oversample*bandwidth). RX-only conditioning."""

    name = "resample"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    params_model = _ResampleParams
    alters_amplitude = True

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "rational_resampler_ccf",
            interpolation=int(params["interpolation"]),
            decimation=int(params["decimation"]),
        )

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        return int(params["interpolation"]) / int(params["decimation"])


class _ClockCorrectParams(StageParams):
    ppm: float  # transmitter clock offset in parts-per-million (signed)

    @model_validator(mode="after")
    def _ok(self) -> "_ClockCorrectParams":
        if self.ppm <= -1e6:
            raise PydanticCustomError("value_error", "ppm must be > -1e6")
        return self


class ClockCorrect(RxStage[CompileContext]):
    """Cancel sampling-frequency offset (transmitter clock drift) by resampling
    by 1/(1 + ppm*1e-6) via the polyphase arbitrary resampler. RX-only. SFO is a
    silent no-op below ~1 chip of cumulative drift (ppm * n_samples), so it only
    matters on long (multi-second) frames."""

    name = "clock_correct"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    params_model = _ClockCorrectParams
    alters_amplitude = True

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("pfb_arb_resampler_ccf", rate=1.0 / (1.0 + float(params["ppm"]) * 1e-6))

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        return 1.0 / (1.0 + float(params["ppm"]) * 1e-6)


class _AgcParams(StageParams):
    mode: Literal["feedforward", "feedback", "power"] = "feedforward"
    reference: float = 1.0
    window_symbols: float = 16.0
    attack_symbols: float = 1.0
    decay_symbols: float = 16.0
    max_gain: float = 0.0

    @model_validator(mode="after")
    def _ok(self) -> "_AgcParams":
        if self.reference <= 0.0:
            raise PydanticCustomError("value_error", "reference must be > 0")
        if self.window_symbols <= 0.0:
            raise PydanticCustomError("value_error", "window_symbols must be > 0")
        if self.attack_symbols <= 0.0 or self.decay_symbols <= 0.0:
            raise PydanticCustomError(
                "value_error", "attack_symbols and decay_symbols must be > 0"
            )
        if self.max_gain < 0.0:
            raise PydanticCustomError("value_error", "max_gain must be >= 0")
        allowed = _MODE_FIELDS[self.mode]
        unused = set().union(*_MODE_FIELDS.values()) - allowed
        supplied = self.model_fields_set & unused
        if supplied:
            raise PydanticCustomError(
                "value_error",
                f"mode '{self.mode}' does not use {sorted(supplied)}",
            )
        return self


_MODE_FIELDS: dict[str, set[str]] = {
    "feedforward": {"window_symbols", "reference"},
    "feedback": {"attack_symbols", "decay_symbols", "max_gain", "reference"},
    "power": {"window_symbols"},
}


def _agc_feedforward(b: CompileContext, p: _AgcParams) -> None:
    b.chain(
        "feedforward_agc_cc",
        nsamples=max(1, round(p.window_symbols * b.sps)),
        reference=p.reference,
    )


def _agc_feedback(b: CompileContext, p: _AgcParams) -> None:
    b.chain(
        "agc2_cc",
        attack_rate=1.0 / (p.attack_symbols * b.sps),
        decay_rate=1.0 / (p.decay_symbols * b.sps),
        reference=p.reference,
        max_gain=p.max_gain,
    )


def _agc_power(b: CompileContext, p: _AgcParams) -> None:
    src = b.tail
    assert src is not None
    alpha = 1.0 / (p.window_symbols * b.sps)
    c2f = b.add("complex_to_float")
    rms = b.add("rms_cf", alpha=alpha)
    b.connect(src, c2f)
    b.connect(src, rms)
    dre = b.add("divide_ff")
    b.connect(c2f, dre, src_port=0, dst_port=0)
    b.connect(rms, dre, src_port=0, dst_port=1)
    dim = b.add("divide_ff")
    b.connect(c2f, dim, src_port=1, dst_port=0)
    b.connect(rms, dim, src_port=0, dst_port=1)
    f2c = b.add("float_to_complex")
    b.connect(dre, f2c, src_port=0, dst_port=0)
    b.connect(dim, f2c, src_port=0, dst_port=1)
    b.set_tail(f2c)


_AGC_MODES: dict[str, Callable[[CompileContext, _AgcParams], None]] = {
    "feedforward": _agc_feedforward,
    "feedback": _agc_feedback,
    "power": _agc_power,
}

# Which statistic each mode drives to `reference`. A consumer declares the
# statistic it needs, so a mode that normalizes the wrong one is a compile
# error instead of a silent mis-decode.
_MODE_AMPLITUDE: dict[str, Amplitude] = {
    "feedforward": Amplitude.PEAK_UNITY,
    "feedback": Amplitude.MEAN_MAG_UNITY,
    "power": Amplitude.RMS_UNITY,
}


class Agc(RxStage[CompileContext]):
    """Normalize IQ amplitude. `mode` selects WHICH statistic is driven to
    `reference`, and that choice is part of the output contract:
    feedforward -> peak_unity, feedback -> mean_mag_unity, power -> rms_unity."""

    name = "agc"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    params_model = _AgcParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _AgcParams.model_validate(dict(params))
        _AGC_MODES[p.mode](b, p)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        p = _AgcParams.model_validate(dict(params))
        return Descriptor(
            Level.IQ,
            in_desc.item_type,
            in_desc.layout,
            in_desc.carrier,
            _MODE_AMPLITUDE[p.mode],
        )


class _SquelchParams(StageParams):
    threshold_db: float
    alpha_symbols: float = 1.0  # power-estimate time constant
    ramp_symbols: float = 0.0  # cosine edge, suppresses the switching transient

    @model_validator(mode="after")
    def _ok(self) -> "_SquelchParams":
        if self.alpha_symbols <= 0.0:
            raise PydanticCustomError("value_error", "alpha_symbols must be > 0")
        if self.ramp_symbols < 0.0:
            raise PydanticCustomError("value_error", "ramp_symbols must be >= 0")
        return self


class Squelch(RxStage[CompileContext]):
    """Mute the stream below a power threshold, IQ->IQ (stock pwr_squelch_cc).

    A clock-recovery loop that free-runs through the gaps between bursts
    integrates noise and arrives at the next burst mistimed; on a real bursty
    capture that turns float-level nondeterminism into whole-symbol drift.
    Muting the gaps zeroes the timing error there, so the loop holds its state
    instead of wandering.

    Samples are ZEROED, never dropped: gating would make the output length
    signal-dependent, which no rate_factor can express, so every downstream
    sps/rate computation would silently disagree with reality.

    `threshold_db` is power relative to the input's scale, so this must run on a
    stream whose scale is known -- hence the amplitude precondition. Put it
    after the agc, and give that agc a window long enough to span the gaps
    (otherwise it lifts the noise floor between bursts to meet the threshold)."""

    name = "squelch"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    accepts_amplitude = frozenset({Amplitude.RMS_UNITY})
    params_model = _SquelchParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _SquelchParams.model_validate(dict(params))
        b.chain(
            "pwr_squelch_cc",
            threshold_db=p.threshold_db,
            alpha=1.0 / (p.alpha_symbols * b.sps),
            ramp=round(p.ramp_symbols * b.sps),
            gate=False,
        )


class _EqualizerParams(StageParams):
    num_taps: StrictInt = 15  # FIR length; must span the channel's delay spread
    step_size: float = 0.01  # CMA adaptation rate (mu)
    modulus: float = 1.0  # target constant modulus the taps drive |y| toward

    @model_validator(mode="after")
    def _ok(self) -> "_EqualizerParams":
        if self.num_taps < 1:
            raise PydanticCustomError("value_error", "num_taps must be >= 1")
        if self.step_size <= 0.0:
            raise PydanticCustomError("value_error", "step_size must be > 0")
        if self.modulus <= 0.0:
            raise PydanticCustomError("value_error", "modulus must be > 0")
        return self


class Equalizer(RxStage[CompileContext]):
    """Blind adaptive channel equalizer: a symbol-spaced FIR whose taps adapt by
    the Constant Modulus Algorithm (stock linear_equalizer at sps=1 +
    adaptive_algorithm_cma). It inverts the inter-symbol interference a
    multipath/frequency-selective channel imposes, with NO training sequence --
    it exploits only that the modulation is (near) constant modulus (PSK, FSK,
    GMSK), so no protocol datasheet enters the product. Run it on a 1-sps stream
    after timing recovery; put an agc before it when the scale is unknown. CMA
    leaves a global sign/phase ambiguity, so a downstream costas / constellation
    receiver still resolves carrier phase. RX-only, IQ->IQ, rate-preserving; the
    output amplitude is left UNKNOWN because CMA's steady state is approximate."""

    name = "equalizer"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    params_model = _EqualizerParams
    alters_amplitude = True

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _EqualizerParams.model_validate(dict(params))
        b.chain(
            "cma_equalizer",
            num_taps=p.num_taps,
            step_size=p.step_size,
            modulus=p.modulus,
        )


class _AmParams(StageParams):
    dc_block_len: StrictInt = 1024

    @model_validator(mode="after")
    def _ok(self) -> "_AmParams":
        if self.dc_block_len < 2:
            raise PydanticCustomError("value_error", "dc_block_len must be >= 2")
        return self


class Am(RxStage[CompileContext]):
    """AM envelope to audio: stock complex_to_mag + dc_blocker. RX-only, IQ->AUDIO,
    rate unchanged."""

    name = "am"
    from_level = Level.IQ
    to_level = Level.AUDIO
    family = "conditioning"
    params_model = _AmParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _AmParams.model_validate(dict(params))
        b.chain("complex_to_mag")
        b.chain("dc_blocker_ff", d=p.dc_block_len)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.AUDIO, "f", in_desc.layout, in_desc.carrier)


class _FmParams(StageParams):
    deviation: float = Field(gt=0)
    dc_block_len: StrictInt = 1024

    @model_validator(mode="after")
    def _ok(self) -> "_FmParams":
        if self.dc_block_len < 2:
            raise PydanticCustomError("value_error", "dc_block_len must be >= 2")
        return self


class FmDemod(RxStage[CompileContext]):
    """FM discriminator to audio: stock quadrature_demod scaled so +-deviation
    maps to +-1.0, then dc_blocker (carrier frequency error rides the
    discriminator output as DC). RX-only, IQ->AUDIO, rate unchanged."""

    name = "fm_demod"
    from_level = Level.IQ
    to_level = Level.AUDIO
    family = "conditioning"
    params_model = _FmParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _FmParams.model_validate(dict(params))
        b.chain("quadrature_demod", gain=b.rate / (2.0 * math.pi * p.deviation))
        b.chain("dc_blocker_ff", d=p.dc_block_len)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.AUDIO, "f", in_desc.layout, in_desc.carrier)


class _AnalyticParams(StageParams):
    ntaps: StrictInt = 65

    @model_validator(mode="after")
    def _ok(self) -> "_AnalyticParams":
        if self.ntaps < 3 or self.ntaps % 2 == 0:
            raise PydanticCustomError("value_error", "ntaps must be odd and >= 3")
        return self


class Analytic(RxStage[CompileContext]):
    """Real audio to its analytic signal (stock hilbert_fc) so IQ demods compose
    downstream. RX-only, AUDIO->IQ, rate unchanged."""

    name = "analytic"
    from_level = Level.AUDIO
    to_level = Level.IQ
    family = "conditioning"
    params_model = _AnalyticParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _AnalyticParams.model_validate(dict(params))
        b.chain("hilbert_fc", ntaps=p.ntaps)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.IQ, "c", in_desc.layout, in_desc.carrier)


CONDITIONING_STAGES: tuple[type[RxStage[CompileContext]], ...] = (
    Channelize,
    Invert,
    Resample,
    ClockCorrect,
    Agc,
    Squelch,
    Equalizer,
    Am,
    FmDemod,
    Analytic,
)
