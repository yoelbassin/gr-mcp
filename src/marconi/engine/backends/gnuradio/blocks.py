from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from marconi.engine.backends.base import BackendError, BackendUnavailable
from marconi.engine.backends.gnuradio.embedded.burst import make_burst_sampler
from marconi.engine.backends.gnuradio.embedded.chirp import (
    dechirp_ref,
    make_chirp_sync,
    make_css_demap,
)
from marconi.engine.backends.gnuradio.embedded.cp_sync import make_cp_symbol_sync
from marconi.engine.backends.gnuradio.embedded.decision import make_peak_decision
from marconi.engine.backends.gnuradio.embedded.framing import make_tag_gate
from marconi.engine.backends.gnuradio.embedded.ldpc import make_ldpc_decoder
from marconi.engine.backends.gnuradio.embedded.msk import make_msk_demod
from marconi.engine.backends.gnuradio.embedded.oerder_meyr import make_oerder_meyr
from marconi.engine.backends.gnuradio.embedded.ofdm import make_ofdm_frame_sync
from marconi.engine.backends.gnuradio.embedded.pilot_lattice import (
    PilotLattice,
    make_pilot_lattice_equalizer,
)
from marconi.engine.backends.gnuradio.embedded.polar import make_polar_decoder
from marconi.engine.backends.gnuradio.embedded.preamble import make_sym_strip
from marconi.engine.backends.gnuradio.embedded.probe import make_burst_probe
from marconi.engine.backends.gnuradio.embedded.trellis_fec import make_trellis_viterbi
from marconi.engine.modulation.fsk.stages import MSK_LOOP_BW_DEFAULT
from marconi.engine.types.params import ParamValue

Params = dict[str, ParamValue]
Factory = Callable[["BlockParams"], Any]

_MISSING = object()

# digital.symbol_sync_*'s positional tail. These are GR 3.10's own defaults,
# passed explicitly so a GR upgrade cannot shift timing behaviour underneath an
# already-validated spec — pybind11 rejects keywords here, so they are named
# rather than commented.
_TED_DAMPING = 1.0
_TED_GAIN = 1.0
_TED_MAX_DEVIATION = 1.5
_SYNC_OUT_SPS = 1
_SYNC_N_FILTERS = 128
_SYNC_TAPS: list[float] = []
# Gardner is non-data-aided: it times off the signal itself and needs no slicer.
_NO_SLICER = None


@dataclass(frozen=True)
class BlockParams:
    """A GR block's parameters, read by name and type. The IR is a serialized
    document, so every read is also a validation: build.py names the block, and
    these name the parameter and what it should have been."""

    values: Params

    def _read(self, name: str, default: object) -> ParamValue:
        if name in self.values:
            return self.values[name]
        if default is _MISSING:
            raise BackendError(f"missing required param {name!r}")
        return cast(ParamValue, default)

    def f(self, name: str, default: object = _MISSING) -> float:
        v = self._read(name, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise BackendError(
                f"param {name!r} must be a real number, got {type(v).__name__}: {v!r}"
            )
        return float(v)

    def i(self, name: str, default: object = _MISSING) -> int:
        v = self._read(name, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise BackendError(
                f"param {name!r} must be an integer, got {type(v).__name__}: {v!r}"
            )
        if isinstance(v, float) and not v.is_integer():
            # the IR-direct dev path skips pydantic, so this is the last line
            # of defense against a silently truncated param (2.7 -> 2)
            raise BackendError(
                f"param {name!r} must be an integer, got non-integral float: {v!r}"
            )
        return int(v)

    def s(self, name: str, default: object = _MISSING) -> str:
        v = self._read(name, default)
        if not isinstance(v, str):
            raise BackendError(
                f"param {name!r} must be a string, got {type(v).__name__}: {v!r}"
            )
        return v

    def b(self, name: str, default: object = _MISSING) -> bool:
        v = self._read(name, default)
        if not isinstance(v, bool):
            raise BackendError(
                f"param {name!r} must be a bool, got {type(v).__name__}: {v!r}"
            )
        return v

    def floats(self, name: str) -> list[float]:
        return [_element_float(name, x) for x in self._sequence(name)]

    def ints(self, name: str) -> list[int]:
        return [_element_int(name, x) for x in self._sequence(name)]

    def _sequence(self, name: str) -> list[float | int]:
        v = self._read(name, _MISSING)
        if not isinstance(v, list):
            raise BackendError(
                f"param {name!r} must be a list, got {type(v).__name__}: {v!r}"
            )
        return v


def _element_float(name: str, v: float | int) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BackendError(
            f"param {name!r} must hold real numbers, got {type(v).__name__}: {v!r}"
        )
    return float(v)


def _element_int(name: str, v: float | int) -> int:
    f = _element_float(name, v)
    if not float(f).is_integer():
        raise BackendError(f"param {name!r} must hold integers, got {v!r}")
    return int(f)


def _complex_syms(i: list[float], q: list[float]) -> list[complex]:
    return [complex(a, b) for a, b in zip(i, q)]


def _unflatten(flat: list[int], lens: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    i = 0
    for n in lens:
        out.append(flat[i : i + n])
        i += n
    if i != len(flat):
        raise BackendError(f"ldpc check_flat length {len(flat)} != sum(check_lens) {i}")
    return out


@dataclass(frozen=True)
class _GrModules:
    gr: Any
    blocks: Any
    analog: Any
    digital: Any
    gr_filter: Any
    firdes: Any
    pfb: Any
    fft: Any
    trellis: Any
    fec: Any


def _modules() -> _GrModules:
    """Single gnuradio import gate. Called at build time only."""
    try:
        from gnuradio import analog, blocks, digital, fec, fft
        from gnuradio import filter as gr_filter
        from gnuradio import gr, trellis
        from gnuradio.filter import firdes, pfb
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise BackendUnavailable(
            "GNU Radio is not importable. Install GNU Radio 3.10+ system-wide "
            "and use a `uv venv --system-site-packages` venv."
        ) from e
    return _GrModules(
        gr=gr,
        blocks=blocks,
        analog=analog,
        digital=digital,
        gr_filter=gr_filter,
        firdes=firdes,
        pfb=pfb,
        fft=fft,
        trellis=trellis,
        fec=fec,
    )


def _const_psk(c: _GrModules, p: BlockParams) -> Any:
    builders = {
        2: c.digital.constellation_bpsk,
        4: c.digital.constellation_qpsk,
        8: c.digital.constellation_8psk,
    }
    order = p.i("order")
    if order not in builders:
        raise BackendError(f"unsupported psk order {order}")
    return builders[order]()


def _const_qam(c: _GrModules, p: BlockParams) -> Any:
    order = p.i("order")
    if order not in (16, 64):
        raise BackendError(f"unsupported qam order {order}")
    con = c.digital.qam.qam_constellation(
        constellation_points=order,
        differential=False,
        mod_code=c.digital.mod_codes.GRAY_CODE,
        large_ampls_to_corners=False,
    )
    con.normalize(c.digital.constellation.POWER_NORMALIZATION)
    return con


def _const_explicit(c: _GrModules, p: BlockParams) -> Any:
    """Arbitrary constellation from caller-supplied points; the bit pattern of a
    point is its index (MSB-first). Covers the 1-D real case (M-PAM / M-ary FSK
    levels, imaginary part zero) as well as any 2-D layout the named schemes
    don't offer. Points are POWER_NORMALIZED, so a consumer must present its
    input at unit RMS."""
    points = _complex_syms(p.floats("points_i"), p.floats("points_q"))
    con = c.digital.constellation_calcdist(points, [], 1, 1)
    con.normalize(c.digital.constellation.POWER_NORMALIZATION)
    return con


_CONSTELLATIONS: dict[str, Callable[[_GrModules, BlockParams], Any]] = {
    "psk": _const_psk,
    "qam": _const_qam,
    "explicit": _const_explicit,
}


def _const(c: _GrModules, p: BlockParams) -> Any:
    scheme = p.s("scheme")
    build = _CONSTELLATIONS.get(scheme)
    if build is None:
        raise BackendError(
            f"unknown constellation scheme {scheme!r}; "
            f"known: {sorted(_CONSTELLATIONS)}"
        )
    return build(c, p)


def _keep_m_in_n_f(c: _GrModules, p: BlockParams) -> Any:
    blk = c.blocks.keep_m_in_n(c.gr.sizeof_float, p.i("m"), p.i("n"), p.i("offset", 0))
    if not bool(p.b("propagate_tags", True)):
        blk.set_tag_propagation_policy(c.gr.TPP_DONT)
    return blk


def _agc2(c: _GrModules, p: BlockParams) -> Any:
    blk = c.analog.agc2_cc(
        p.f("attack_rate"),
        p.f("decay_rate"),
        p.f("reference"),
        1.0,
    )
    max_gain = p.f("max_gain")
    if max_gain > 0.0:
        blk.set_max_gain(max_gain)
    return blk


def _soapy_source(c: _GrModules, p: BlockParams) -> Any:
    from gnuradio import soapy

    src = soapy.source(str(p.s("device", "")), "fc32", 1, "", "", [""], [""])
    src.set_sample_rate(0, p.f("sample_rate"))
    ppm = p.f("ppm", 0.0)
    if ppm:
        src.set_frequency_correction(0, ppm)
    src.set_frequency(0, p.f("center_hz"))
    if bool(p.b("agc", True)):
        src.set_gain_mode(0, True)
    else:
        src.set_gain_mode(0, False)
        src.set_gain(0, p.f("gain_db"))
    return src


# kind -> (ctx, params) -> live GR block. The only GR-aware vocabulary in the
# engine (the engine ⊥ gnuradio invariant in test_invariants).
GR_BLOCKS: dict[str, Callable[[_GrModules, BlockParams], Any]] = {
    # offset/length are in items; length 0 = to EOF (stock file_source
    # semantics) - the streaming way to decode a bounded capture slice
    "iq_file_source": lambda c, p: c.blocks.file_source(
        c.gr.sizeof_gr_complex,
        p.s("path"),
        bool(p.b("repeat", False)),
        p.i("offset", 0),
        p.i("length", 0),
    ),
    "iq_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_gr_complex, p.s("path"), False
    ),
    "soapy_source": _soapy_source,
    "iq_skiphead": lambda c, p: c.blocks.skiphead(
        c.gr.sizeof_gr_complex, p.i("num_items")
    ),
    "iq_head": lambda c, p: c.blocks.head(c.gr.sizeof_gr_complex, p.i("num_items")),
    "bits_file_source": lambda c, p: c.blocks.file_source(
        c.gr.sizeof_char,
        p.s("path"),
        bool(p.b("repeat", False)),
        p.i("offset", 0),
        p.i("length", 0),
    ),
    "bits_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_char, p.s("path"), False
    ),
    "soft_bits_file_source": lambda c, p: c.blocks.file_source(
        c.gr.sizeof_float,
        p.s("path"),
        bool(p.b("repeat", False)),
        p.i("offset", 0),
        p.i("length", 0),
    ),
    "soft_bits_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_float, p.s("path"), False
    ),
    # symbols are hard integer symbol indices (int16) — the wire type of the
    # symbol-terminating verticals (CSS peak_decision, m_slice)
    "symbols_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_short, p.s("path"), False
    ),
    "quadrature_demod": lambda c, p: c.analog.quadrature_demod_cf(p.f("gain")),
    "feedforward_agc_cc": lambda c, p: c.analog.feedforward_agc_cc(
        p.i("nsamples"), p.f("reference")
    ),
    "agc2_cc": _agc2,
    "fll_band_edge_cc": lambda c, p: c.digital.fll_band_edge_cc(
        p.f("sps"),
        p.f("rolloff"),
        p.i("filter_size"),
        p.f("loop_bw"),
    ),
    "pwr_squelch_cc": lambda c, p: c.analog.pwr_squelch_cc(
        p.f("threshold_db"),
        p.f("alpha"),
        p.i("ramp", 0),
        bool(p.b("gate", False)),
    ),
    "msk_demod": lambda c, p: make_msk_demod(
        c.gr,
        sps=p.f("sps"),
        loop_bw=p.f("loop_bw", MSK_LOOP_BW_DEFAULT),
        loop_pole=p.f("loop_pole", 0.52),
        mf_oversample=p.i("mf_oversample", 12),
    ),
    "burst_sampler": lambda c, p: make_burst_sampler(c.gr, sps=p.f("sps")),
    "oerder_meyr_timing": lambda c, p: make_oerder_meyr(
        c.gr,
        sps=p.f("sps"),
        span=p.i("span", 11),
        alpha=p.f("alpha", 0.35),
    ),
    "symbol_sync_ff": lambda c, p: c.digital.symbol_sync_ff(
        c.digital.TED_GARDNER,
        p.f("sps"),
        p.f("loop_bw", 0.045),
        _TED_DAMPING,
        _TED_GAIN,
        _TED_MAX_DEVIATION,
        _SYNC_OUT_SPS,
        c.digital.constellation_bpsk().base(),
        c.digital.IR_MMSE_8TAP,
        _SYNC_N_FILTERS,
        _SYNC_TAPS,
    ),
    "binary_slicer": lambda c, p: c.digital.binary_slicer_fb(),
    "rrc_filter_ccf": lambda c, p: c.gr_filter.interp_fir_filter_ccf(
        p.i("interpolation"),
        c.firdes.root_raised_cosine(
            float(p.i("interpolation")),
            p.f("rate"),
            p.f("rate") / p.f("sps"),
            p.f("alpha", 0.35),
            # tap count is a filter LENGTH, legitimately non-integer-sps: round it
            # so a fractional-sps capture (rate not an integer multiple of baud)
            # builds instead of crashing on an integer read of sps.
            round(p.f("sps") * p.i("span", 11)) + 1,
        ),
    ),
    "freq_xlating_fir_filter_ccf": lambda c, p: c.gr_filter.freq_xlating_fir_filter_ccf(
        p.i("decim"),
        c.firdes.low_pass(
            1.0,
            p.f("rate"),
            p.f("cutoff"),
            p.f("transition"),
        ),
        p.f("center"),
        p.f("rate"),
    ),
    "rotator_cc": lambda c, p: c.blocks.rotator_cc(p.f("phase_inc")),
    # Polyphase arbitrary resampler (rate=interp/decim). Kept over
    # rational_resampler_ccf because the same block also serves clock_correct's
    # irrational 1/(1+ppm) ratio, so one kind covers both.
    "pfb_arb_resampler_ccf": lambda c, p: c.pfb.arb_resampler_ccf(p.f("rate")),
    # Integer-ratio resampler (auto-designed anti-imaging taps). Spectrally clean
    # and bit-perfect on this build; see the note above pfb_arb_resampler_ccf.
    "rational_resampler_ccf": lambda c, p: c.gr_filter.rational_resampler_ccf(
        interpolation=p.i("interpolation"), decimation=p.i("decimation")
    ),
    "conjugate_cc": lambda c, p: c.blocks.conjugate_cc(),
    "symbol_sync_cc": lambda c, p: c.digital.symbol_sync_cc(
        c.digital.TED_GARDNER,
        p.f("sps"),
        p.f("loop_bw", 0.045),
        _TED_DAMPING,
        _TED_GAIN,
        _TED_MAX_DEVIATION,
        _SYNC_OUT_SPS,
        _NO_SLICER,
        c.digital.IR_MMSE_8TAP,
        _SYNC_N_FILTERS,
        _SYNC_TAPS,
    ),
    "costas_loop_cc": lambda c, p: c.digital.costas_loop_cc(
        p.f("loop_bw", 0.045), p.i("order"), False
    ),
    # Blind CMA equalizer: linear_equalizer at sps=1 (symbol-spaced, rate-
    # preserving) driven by the constant-modulus algorithm. The constellation is
    # only a placeholder — CMA is decision-free, so its points never enter the
    # tap update. linear_equalizer holds the algorithm's C++ shared_ptr, so the
    # inline object outlives this expression (same pattern as _const below).
    "cma_equalizer": lambda c, p: c.digital.linear_equalizer(
        p.i("num_taps"),
        1,
        c.digital.adaptive_algorithm_cma(
            c.digital.constellation_bpsk().base(),
            p.f("step_size"),
            p.f("modulus"),
        ),
        True,
        [],
        "",
    ),
    "constellation_receiver_cb": lambda c, p: c.digital.constellation_receiver_cb(
        _const(c, p).base(),
        p.f("loop_bw", 0.04),
        p.f("fmin", -0.5),
        p.f("fmax", 0.5),
    ),
    "constellation_decoder_cb": lambda c, p: c.digital.constellation_decoder_cb(
        _const(c, p).base()
    ),
    "unpack_k_bits_bb": lambda c, p: c.blocks.unpack_k_bits_bb(p.i("k")),
    "complex_to_mag": lambda c, p: c.blocks.complex_to_mag(p.i("vlen", 1)),
    "dc_blocker_ff": lambda c, p: c.gr_filter.dc_blocker_ff(p.i("d"), True),
    "hilbert_fc": lambda c, p: c.gr_filter.hilbert_fc(p.i("ntaps")),
    "uchar_to_float": lambda c, p: c.blocks.uchar_to_float(),
    "float_to_short": lambda c, p: c.blocks.float_to_short(1, p.f("scale")),
    "multiply_const_ff": lambda c, p: c.blocks.multiply_const_ff(p.f("value")),
    "add_const_ff": lambda c, p: c.blocks.add_const_ff(p.f("value")),
    "float_to_complex": lambda c, p: c.blocks.float_to_complex(1),
    "complex_to_float": lambda c, p: c.blocks.complex_to_float(1),
    "rms_cf": lambda c, p: c.blocks.rms_cf(p.f("alpha")),
    "divide_ff": lambda c, p: c.blocks.divide_ff(1),
    "corr_est_cc": lambda c, p: c.digital.corr_est_cc(
        _complex_syms(p.floats("preamble_i"), p.floats("preamble_q")),
        p.i("sps"),
        p.i("mark_delay"),
        p.f("threshold"),
    ),
    "sym_strip": lambda c, p: make_sym_strip(c.gr, n_pre=p.i("n_pre")),
    "correlate_access_code_tag_ff": lambda c, p: c.digital.correlate_access_code_tag_ff(
        p.s("access_code"), p.i("threshold"), p.s("tag_name")
    ),
    "tag_gate": lambda c, p: make_tag_gate(
        c.gr,
        frame_len=p.i("frame_len"),
        tag_name=p.s("tag_name"),
        chance_per_item=p.f("chance_per_item"),
    ),
    "chirp_sync": lambda c, p: make_chirp_sync(
        c.gr,
        p.i("sf"),
        p.i("oversample"),
        p.i("zero_pad"),
        p.i("preamble_len"),
        p.f("bandwidth"),
        p.f("sfd_symbols"),
        p.i("sync_symbols"),
    ),
    "chirp_ref_source": lambda c, p: c.blocks.vector_source_c(
        dechirp_ref(p.i("sf"), p.i("oversample")).tolist(), True
    ),
    "multiply_cc": lambda c, p: c.blocks.multiply_cc(),
    "null_source_c": lambda c, p: c.blocks.null_source(c.gr.sizeof_gr_complex),
    "stream_mux_c": lambda c, p: c.blocks.stream_mux(
        c.gr.sizeof_gr_complex, p.ints("lengths")
    ),
    "vector_to_stream_f": lambda c, p: c.blocks.vector_to_stream(
        c.gr.sizeof_float, p.i("vlen")
    ),
    "stream_to_vector_f": lambda c, p: c.blocks.stream_to_vector(
        c.gr.sizeof_float, p.i("vlen")
    ),
    "keep_m_in_n_f": _keep_m_in_n_f,
    "add_ff": lambda c, p: c.blocks.add_ff(),
    "peak_decision": lambda c, p: make_peak_decision(
        c.gr,
        vlen=p.i("vlen"),
        divisor=p.i("divisor"),
        modulo=p.i("modulo"),
    ),
    "css_demap": lambda c, p: make_css_demap(c.gr, p.i("sf")),
    "ofdm_frame_sync": lambda c, p: make_ofdm_frame_sync(
        c.gr,
        fft_len=p.i("fft_len"),
        cp_len=p.i("cp_len"),
        sym_len=p.i("sym_len"),
        null_len=p.i("null_len"),
        frame_len=p.i("frame_len"),
        data_syms=p.i("data_syms"),
    ),
    "pilot_lattice_equalizer": lambda c, p: make_pilot_lattice_equalizer(
        c.gr,
        fft_len=p.i("fft_len"),
        n_frame_syms=p.i("n_frame_syms"),
        n_carriers=p.i("n_carriers"),
        kmin=p.i("kmin"),
        dc_search=p.i("dc_search"),
        warmup_syms=p.i("warmup_syms"),
        lattice=PilotLattice.from_flat(
            pilot_lens=p.ints("pilot_lens"),
            pilot_carriers=p.ints("pilot_carriers"),
            pilot_i=p.floats("pilot_i"),
            pilot_q=p.floats("pilot_q"),
            fp_carriers=p.ints("fp_carriers"),
            fp_i=p.floats("fp_i"),
            fp_q=p.floats("fp_q"),
        ),
        lock_min_score=p.f("lock_min_score"),
    ),
    "cp_symbol_sync": lambda c, p: make_cp_symbol_sync(
        c.gr,
        fft_len=p.i("fft_len"),
        cp_len=p.i("cp_len"),
        warmup_syms=p.i("warmup_syms"),
        lock_min_ratio=p.f("lock_min_ratio"),
    ),
    "stream_to_vector": lambda c, p: c.blocks.stream_to_vector(
        c.gr.sizeof_gr_complex, p.i("vlen")
    ),
    "vector_to_stream": lambda c, p: c.blocks.vector_to_stream(
        c.gr.sizeof_gr_complex, p.i("vlen")
    ),
    "fft_vcc": lambda c, p: c.fft.fft_vcc(
        p.i("fft_len"),
        bool(p.b("forward", True)),
        c.fft.window.rectangular(p.i("fft_len")),
        bool(p.b("shift", False)),
        1,
    ),
    "blockinterleaver_cc": lambda c, p: c.blocks.blockinterleaver_cc(
        p.ints("perm"), bool(p.b("mode", True)), False
    ),
    "blockinterleaver_ff": lambda c, p: c.blocks.blockinterleaver_ff(
        p.ints("perm"), bool(p.b("mode", True)), False
    ),
    "keep_m_in_n_c": lambda c, p: c.blocks.keep_m_in_n(
        c.gr.sizeof_gr_complex,
        p.i("m"),
        p.i("n"),
        p.i("offset", 0),
    ),
    "delay_cc": lambda c, p: c.blocks.delay(c.gr.sizeof_gr_complex, p.i("samples")),
    "multiply_conjugate_cc": lambda c, p: c.blocks.multiply_conjugate_cc(),
    "multiply_const_cc": lambda c, p: c.blocks.multiply_const_cc(
        complex(p.f("re"), p.f("im"))
    ),
    "constellation_soft_decoder": lambda c, p: c.digital.constellation_soft_decoder_cf(
        _const(c, p).base()
    ),
    "patterned_interleaver_f": lambda c, p: c.blocks.patterned_interleaver(
        c.gr.sizeof_float, p.ints("pattern")
    ),
    "null_source_f": lambda c, p: c.blocks.null_source(c.gr.sizeof_float),
    "trellis_viterbi": lambda c, p: make_trellis_viterbi(
        c,
        rate_inv=p.i("rate_inv"),
        polys=p.ints("polys"),
        frame_bits=p.i("frame_bits"),
        tail=p.i("tail"),
        k=p.i("k"),
    ),
    "keep_m_in_n_b": lambda c, p: c.blocks.keep_m_in_n(
        c.gr.sizeof_char, p.i("m"), p.i("n"), 0
    ),
    "polar_decode": lambda c, p: make_polar_decoder(
        c,
        block_size=p.i("block_size"),
        info_bits=p.i("info_bits"),
        frozen_positions=p.ints("frozen_positions"),
        frozen_values=p.ints("frozen_values"),
        list_size=p.i("list_size"),
    ),
    "ldpc_decode": lambda c, p: make_ldpc_decoder(
        c,
        block_size=p.i("block_size"),
        check_nodes=_unflatten(p.ints("check_flat"), p.ints("check_lens")),
        max_iterations=p.i("max_iterations"),
    ),
    "burst_probe": lambda c, p: make_burst_probe(c.gr),
}


def _bind(fn: Callable[[_GrModules, BlockParams], Any], ctx: _GrModules) -> Factory:
    return lambda p: fn(ctx, p)


def _factories() -> dict[str, Factory]:
    ctx = _modules()
    return {kind: _bind(fn, ctx) for kind, fn in GR_BLOCKS.items()}
