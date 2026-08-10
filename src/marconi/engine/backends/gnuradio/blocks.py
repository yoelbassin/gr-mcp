from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from marconi.engine.backends.base import BackendError, BackendUnavailable
from marconi.engine.backends.gnuradio.embedded.burst import make_burst_sampler
from marconi.engine.backends.gnuradio.embedded.chirp import (
    chirp_prefix,
    dechirp_ref,
    make_chirp_mod,
    make_chirp_sync,
    make_css_demap,
    make_css_map,
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
from marconi.engine.backends.gnuradio.embedded.preamble import (
    make_sym_strip,
    sym_prefix,
)
from marconi.engine.backends.gnuradio.embedded.probe import make_burst_probe
from marconi.engine.backends.gnuradio.embedded.trellis_fec import make_trellis_viterbi
from marconi.engine.types.params import ParamValue

Params = dict[str, ParamValue]
Factory = Callable[[Params], Any]

# vector_insert_c re-inserts each period; graphs stay far below 2^30 items
_INSERT_ONCE = 1 << 30


def _as_float(v: ParamValue) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BackendError(f"expected a real number, got {type(v).__name__}: {v!r}")
    return float(v)


def _as_int(v: ParamValue) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BackendError(f"expected an integer, got {type(v).__name__}: {v!r}")
    if isinstance(v, float) and not v.is_integer():
        # the IR-direct dev path skips pydantic, so this is the last line of
        # defense against a silently truncated param (2.7 -> 2)
        raise BackendError(f"expected an integer, got non-integral float: {v!r}")
    return int(v)


def _as_float_list(v: ParamValue) -> list[float]:
    if not isinstance(v, list):
        raise BackendError(f"expected a list of numbers, got {type(v).__name__}: {v!r}")
    return [_as_float(x) for x in v]


def _as_int_list(v: ParamValue) -> list[int]:
    if not isinstance(v, list):
        raise BackendError(
            f"expected a list of integers, got {type(v).__name__}: {v!r}"
        )
    return [_as_int(x) for x in v]


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


@dataclass(frozen=True)
class _GrCtx:
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
    rate: float


def _make_ctx(rate: float) -> _GrCtx:
    m = _modules()
    return _GrCtx(
        gr=m.gr,
        blocks=m.blocks,
        analog=m.analog,
        digital=m.digital,
        gr_filter=m.gr_filter,
        firdes=m.firdes,
        pfb=m.pfb,
        fft=m.fft,
        trellis=m.trellis,
        fec=m.fec,
        rate=rate,
    )


def _const_psk(c: _GrCtx, p: Params) -> Any:
    builders = {
        2: c.digital.constellation_bpsk,
        4: c.digital.constellation_qpsk,
        8: c.digital.constellation_8psk,
    }
    order = _as_int(p["order"])
    if order not in builders:
        raise BackendError(f"unsupported psk order {order}")
    return builders[order]()


def _const_qam(c: _GrCtx, p: Params) -> Any:
    order = _as_int(p["order"])
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


def _const_explicit(c: _GrCtx, p: Params) -> Any:
    """Arbitrary constellation from caller-supplied points; the bit pattern of a
    point is its index (MSB-first). Covers the 1-D real case (M-PAM / M-ary FSK
    levels, imaginary part zero) as well as any 2-D layout the named schemes
    don't offer. Points are POWER_NORMALIZED, so a consumer must present its
    input at unit RMS."""
    points = _complex_syms(_as_float_list(p["points_i"]), _as_float_list(p["points_q"]))
    con = c.digital.constellation_calcdist(points, [], 1, 1)
    con.normalize(c.digital.constellation.POWER_NORMALIZATION)
    return con


_CONSTELLATIONS: dict[str, Callable[[_GrCtx, Params], Any]] = {
    "psk": _const_psk,
    "qam": _const_qam,
    "explicit": _const_explicit,
}


def _const(c: _GrCtx, p: Params) -> Any:
    scheme = str(p["scheme"])
    build = _CONSTELLATIONS.get(scheme)
    if build is None:
        raise BackendError(
            f"unknown constellation scheme {scheme!r}; "
            f"known: {sorted(_CONSTELLATIONS)}"
        )
    return build(c, p)


def _keep_m_in_n_f(c: _GrCtx, p: Params) -> Any:
    blk = c.blocks.keep_m_in_n(
        c.gr.sizeof_float, _as_int(p["m"]), _as_int(p["n"]), _as_int(p.get("offset", 0))
    )
    if not bool(p.get("propagate_tags", True)):
        blk.set_tag_propagation_policy(c.gr.TPP_DONT)
    return blk


def _agc2(c: _GrCtx, p: Params) -> Any:
    blk = c.analog.agc2_cc(
        _as_float(p["attack_rate"]),
        _as_float(p["decay_rate"]),
        _as_float(p["reference"]),
        1.0,
    )
    max_gain = _as_float(p["max_gain"])
    if max_gain > 0.0:
        blk.set_max_gain(max_gain)
    return blk


def _soapy_source(c: _GrCtx, p: Params) -> Any:
    from gnuradio import soapy

    src = soapy.source(str(p.get("device", "")), "fc32", 1, "", "", [""], [""])
    src.set_sample_rate(0, _as_float(p["sample_rate"]))
    ppm = _as_float(p.get("ppm", 0.0))
    if ppm:
        src.set_frequency_correction(0, ppm)
    src.set_frequency(0, _as_float(p["center_hz"]))
    if bool(p.get("agc", True)):
        src.set_gain_mode(0, True)
    else:
        src.set_gain_mode(0, False)
        src.set_gain(0, _as_float(p["gain_db"]))
    return src


# kind -> (ctx, params) -> live GR block. The ONLY GR-aware vocabulary in phy.
GR_BLOCKS: dict[str, Callable[[_GrCtx, Params], Any]] = {
    # offset/length are in items; length 0 = to EOF (stock file_source
    # semantics) - the streaming way to decode a bounded capture slice
    "iq_file_source": lambda c, p: c.blocks.file_source(
        c.gr.sizeof_gr_complex,
        str(p["path"]),
        bool(p.get("repeat", False)),
        _as_int(p.get("offset", 0)),
        _as_int(p.get("length", 0)),
    ),
    "iq_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_gr_complex, str(p["path"]), False
    ),
    "soapy_source": _soapy_source,
    "iq_skiphead": lambda c, p: c.blocks.skiphead(
        c.gr.sizeof_gr_complex, _as_int(p["num_items"])
    ),
    "iq_head": lambda c, p: c.blocks.head(
        c.gr.sizeof_gr_complex, _as_int(p["num_items"])
    ),
    "bits_file_source": lambda c, p: c.blocks.file_source(
        c.gr.sizeof_char,
        str(p["path"]),
        bool(p.get("repeat", False)),
        _as_int(p.get("offset", 0)),
        _as_int(p.get("length", 0)),
    ),
    "bits_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_char, str(p["path"]), False
    ),
    "soft_bits_file_source": lambda c, p: c.blocks.file_source(
        c.gr.sizeof_float,
        str(p["path"]),
        bool(p.get("repeat", False)),
        _as_int(p.get("offset", 0)),
        _as_int(p.get("length", 0)),
    ),
    "soft_bits_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_float, str(p["path"]), False
    ),
    # symbols are hard integer symbol indices (int16) — the wire type of the
    # symbol-terminating verticals (CSS peak_decision, m_slice)
    "symbols_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_short, str(p["path"]), False
    ),
    "quadrature_demod": lambda c, p: c.analog.quadrature_demod_cf(_as_float(p["gain"])),
    "feedforward_agc_cc": lambda c, p: c.analog.feedforward_agc_cc(
        _as_int(p["nsamples"]), _as_float(p["reference"])
    ),
    "agc2_cc": _agc2,
    "fll_band_edge_cc": lambda c, p: c.digital.fll_band_edge_cc(
        _as_float(p["sps"]),
        _as_float(p["rolloff"]),
        _as_int(p["filter_size"]),
        _as_float(p["loop_bw"]),
    ),
    "pwr_squelch_cc": lambda c, p: c.analog.pwr_squelch_cc(
        _as_float(p["threshold_db"]),
        _as_float(p["alpha"]),
        _as_int(p.get("ramp", 0)),
        bool(p.get("gate", False)),
    ),
    "msk_demod": lambda c, p: make_msk_demod(
        c.gr,
        sps=_as_float(p["sps"]),
        loop_bw=_as_float(p.get("loop_bw", 0.0038)),
        loop_pole=_as_float(p.get("loop_pole", 0.52)),
        mf_oversample=_as_int(p.get("mf_oversample", 12)),
    ),
    "burst_sampler": lambda c, p: make_burst_sampler(c.gr, sps=_as_float(p["sps"])),
    "oerder_meyr_timing": lambda c, p: make_oerder_meyr(
        c.gr,
        sps=_as_float(p["sps"]),
        span=_as_int(p.get("span", 11)),
        alpha=_as_float(p.get("alpha", 0.35)),
    ),
    "symbol_sync_ff": lambda c, p: c.digital.symbol_sync_ff(
        c.digital.TED_GARDNER,
        _as_float(p["sps"]),
        _as_float(p.get("loop_bw", 0.045)),
        1.0,  # damping
        1.0,  # ted_gain
        1.5,  # max_deviation
        1,  # output samples per symbol
        c.digital.constellation_bpsk().base(),
        c.digital.IR_MMSE_8TAP,
        128,  # n_filters
        [],  # taps
    ),
    "binary_slicer": lambda c, p: c.digital.binary_slicer_fb(),
    "chunks_to_symbols": lambda c, p: c.digital.chunks_to_symbols_bf(
        _as_float_list(p["symbols"])
    ),
    "repeat_f": lambda c, p: c.blocks.repeat(c.gr.sizeof_float, _as_int(p["interp"])),
    "frequency_modulator": lambda c, p: c.analog.frequency_modulator_fc(
        _as_float(p["sensitivity"])
    ),
    "rrc_filter_ccf": lambda c, p: c.gr_filter.interp_fir_filter_ccf(
        _as_int(p["interpolation"]),
        c.firdes.root_raised_cosine(
            float(_as_int(p["interpolation"])),
            _as_float(p["rate"]),
            _as_float(p["rate"]) / _as_float(p["sps"]),
            _as_float(p.get("alpha", 0.35)),
            # tap count is a filter LENGTH, legitimately non-integer-sps: round it
            # so a fractional-sps capture (rate not an integer multiple of baud)
            # builds instead of crashing on _as_int(sps).
            round(_as_float(p["sps"]) * _as_int(p.get("span", 11))) + 1,
        ),
    ),
    "freq_xlating_fir_filter_ccf": lambda c, p: c.gr_filter.freq_xlating_fir_filter_ccf(
        _as_int(p["decim"]),
        c.firdes.low_pass(
            1.0,
            _as_float(p["rate"]),
            _as_float(p["cutoff"]),
            _as_float(p["transition"]),
        ),
        _as_float(p["center"]),
        _as_float(p["rate"]),
    ),
    "rotator_cc": lambda c, p: c.blocks.rotator_cc(_as_float(p["phase_inc"])),
    # Polyphase arbitrary resampler (rate=interp/decim). Kept over
    # rational_resampler_ccf because the same block also serves clock_correct's
    # irrational 1/(1+ppm) ratio, so one kind covers both — NOT because rational
    # images. rational_resampler_ccf with no taps is spectrally clean (<-56 dBc)
    # and bit-perfect here; its "BER ~0.46 at 8/7, 8/9" through
    # test_resample_roundtrip is an aligned_ber artifact, not a DSP fault: its
    # group delay lands rx ~2 samples EARLY (negative lag) and aligned_ber only
    # shifts rx forward, so it scores a perfect decode as random. Verified
    # two-sided 2026-07-25; do not "re-fix" this by distrusting rational.
    "pfb_arb_resampler_ccf": lambda c, p: c.pfb.arb_resampler_ccf(_as_float(p["rate"])),
    # Integer-ratio resampler (auto-designed anti-imaging taps). Spectrally clean
    # and bit-perfect on this build; see the note above pfb_arb_resampler_ccf.
    "rational_resampler_ccf": lambda c, p: c.gr_filter.rational_resampler_ccf(
        interpolation=_as_int(p["interpolation"]), decimation=_as_int(p["decimation"])
    ),
    "conjugate_cc": lambda c, p: c.blocks.conjugate_cc(),
    "symbol_sync_cc": lambda c, p: c.digital.symbol_sync_cc(
        c.digital.TED_GARDNER,
        _as_float(p["sps"]),
        _as_float(p.get("loop_bw", 0.045)),
        1.0,  # damping
        1.0,  # ted_gain
        1.5,  # max_deviation
        1,  # output sps
        None,  # non-data-aided (Gardner): no constellation
        c.digital.IR_MMSE_8TAP,
        128,
        [],
    ),
    "costas_loop_cc": lambda c, p: c.digital.costas_loop_cc(
        _as_float(p.get("loop_bw", 0.045)), _as_int(p["order"]), False
    ),
    # Blind CMA equalizer: linear_equalizer at sps=1 (symbol-spaced, rate-
    # preserving) driven by the constant-modulus algorithm. The constellation is
    # only a placeholder — CMA is decision-free, so its points never enter the
    # tap update. linear_equalizer holds the algorithm's C++ shared_ptr, so the
    # inline object outlives this expression (same pattern as _const below).
    "cma_equalizer": lambda c, p: c.digital.linear_equalizer(
        _as_int(p["num_taps"]),
        1,
        c.digital.adaptive_algorithm_cma(
            c.digital.constellation_bpsk().base(),
            _as_float(p["step_size"]),
            _as_float(p["modulus"]),
        ),
        True,
        [],
        "",
    ),
    "constellation_receiver_cb": lambda c, p: c.digital.constellation_receiver_cb(
        _const(c, p).base(),
        _as_float(p.get("loop_bw", 0.04)),
        _as_float(p.get("fmin", -0.5)),
        _as_float(p.get("fmax", 0.5)),
    ),
    "chunks_to_symbols_bc": lambda c, p: c.digital.chunks_to_symbols_bc(
        _const(c, p).points()
    ),
    "constellation_decoder_cb": lambda c, p: c.digital.constellation_decoder_cb(
        _const(c, p).base()
    ),
    "pack_k_bits_bb": lambda c, p: c.blocks.pack_k_bits_bb(_as_int(p["k"])),
    "unpack_k_bits_bb": lambda c, p: c.blocks.unpack_k_bits_bb(_as_int(p["k"])),
    "complex_to_mag": lambda c, p: c.blocks.complex_to_mag(_as_int(p.get("vlen", 1))),
    "dc_blocker_ff": lambda c, p: c.gr_filter.dc_blocker_ff(_as_int(p["d"]), True),
    "hilbert_fc": lambda c, p: c.gr_filter.hilbert_fc(_as_int(p["ntaps"])),
    "uchar_to_float": lambda c, p: c.blocks.uchar_to_float(),
    "float_to_short": lambda c, p: c.blocks.float_to_short(1, _as_float(p["scale"])),
    "multiply_const_ff": lambda c, p: c.blocks.multiply_const_ff(_as_float(p["value"])),
    "add_const_ff": lambda c, p: c.blocks.add_const_ff(_as_float(p["value"])),
    "float_to_complex": lambda c, p: c.blocks.float_to_complex(1),
    "complex_to_float": lambda c, p: c.blocks.complex_to_float(1),
    "rms_cf": lambda c, p: c.blocks.rms_cf(_as_float(p["alpha"])),
    "divide_ff": lambda c, p: c.blocks.divide_ff(1),
    "sym_prepend": lambda c, p: c.blocks.vector_insert_c(
        sym_prefix(
            _as_float_list(p["preamble_i"]),
            _as_float_list(p["preamble_q"]),
            _as_int(p["pad_symbols"]),
        ).tolist(),
        _INSERT_ONCE,
        0,
    ),
    "corr_est_cc": lambda c, p: c.digital.corr_est_cc(
        _complex_syms(_as_float_list(p["preamble_i"]), _as_float_list(p["preamble_q"])),
        _as_int(p["sps"]),
        _as_int(p["mark_delay"]),
        _as_float(p["threshold"]),
    ),
    "sym_strip": lambda c, p: make_sym_strip(c.gr, n_pre=_as_int(p["n_pre"])),
    "correlate_access_code_tag_ff": lambda c, p: c.digital.correlate_access_code_tag_ff(
        str(p["access_code"]), _as_int(p["threshold"]), str(p["tag_name"])
    ),
    "tag_gate": lambda c, p: make_tag_gate(
        c.gr,
        frame_len=_as_int(p["frame_len"]),
        tag_name=str(p["tag_name"]),
        chance_per_item=_as_float(p["chance_per_item"]),
    ),
    "chirp_prepend": lambda c, p: c.blocks.vector_insert_c(
        chirp_prefix(
            _as_int(p["sf"]),
            _as_int(p["oversample"]),
            _as_int(p["preamble_len"]),
            _as_float(p["sfd_symbols"]),
        ).tolist(),
        _INSERT_ONCE,
        0,
    ),
    "chirp_sync": lambda c, p: make_chirp_sync(
        c.gr,
        _as_int(p["sf"]),
        _as_int(p["oversample"]),
        _as_int(p["zero_pad"]),
        _as_int(p["preamble_len"]),
        _as_float(p["bandwidth"]),
        _as_float(p["sfd_symbols"]),
        _as_int(p["sync_symbols"]),
    ),
    "chirp_mod": lambda c, p: make_chirp_mod(
        c.gr, _as_int(p["sf"]), _as_int(p["oversample"])
    ),
    "chirp_ref_source": lambda c, p: c.blocks.vector_source_c(
        dechirp_ref(_as_int(p["sf"]), _as_int(p["oversample"])).tolist(), True
    ),
    "multiply_cc": lambda c, p: c.blocks.multiply_cc(),
    "null_source_c": lambda c, p: c.blocks.null_source(c.gr.sizeof_gr_complex),
    "stream_mux_c": lambda c, p: c.blocks.stream_mux(
        c.gr.sizeof_gr_complex, _as_int_list(p["lengths"])
    ),
    "vector_to_stream_f": lambda c, p: c.blocks.vector_to_stream(
        c.gr.sizeof_float, _as_int(p["vlen"])
    ),
    "stream_to_vector_f": lambda c, p: c.blocks.stream_to_vector(
        c.gr.sizeof_float, _as_int(p["vlen"])
    ),
    "keep_m_in_n_f": _keep_m_in_n_f,
    "add_ff": lambda c, p: c.blocks.add_ff(),
    "peak_decision": lambda c, p: make_peak_decision(
        c.gr,
        vlen=_as_int(p["vlen"]),
        divisor=_as_int(p["divisor"]),
        modulo=_as_int(p["modulo"]),
    ),
    "css_map": lambda c, p: make_css_map(c.gr, _as_int(p["sf"])),
    "css_demap": lambda c, p: make_css_demap(c.gr, _as_int(p["sf"])),
    "ofdm_frame_sync": lambda c, p: make_ofdm_frame_sync(
        c.gr,
        fft_len=_as_int(p["fft_len"]),
        cp_len=_as_int(p["cp_len"]),
        sym_len=_as_int(p["sym_len"]),
        null_len=_as_int(p["null_len"]),
        frame_len=_as_int(p["frame_len"]),
        data_syms=_as_int(p["data_syms"]),
    ),
    "pilot_lattice_equalizer": lambda c, p: make_pilot_lattice_equalizer(
        c.gr,
        fft_len=_as_int(p["fft_len"]),
        n_frame_syms=_as_int(p["n_frame_syms"]),
        n_carriers=_as_int(p["n_carriers"]),
        kmin=_as_int(p["kmin"]),
        dc_search=_as_int(p["dc_search"]),
        warmup_syms=_as_int(p["warmup_syms"]),
        lattice=PilotLattice.from_flat(
            pilot_lens=_as_int_list(p["pilot_lens"]),
            pilot_carriers=_as_int_list(p["pilot_carriers"]),
            pilot_i=_as_float_list(p["pilot_i"]),
            pilot_q=_as_float_list(p["pilot_q"]),
            fp_carriers=_as_int_list(p["fp_carriers"]),
            fp_i=_as_float_list(p["fp_i"]),
            fp_q=_as_float_list(p["fp_q"]),
        ),
        lock_min_score=_as_float(p["lock_min_score"]),
    ),
    "cp_symbol_sync": lambda c, p: make_cp_symbol_sync(
        c.gr,
        fft_len=_as_int(p["fft_len"]),
        cp_len=_as_int(p["cp_len"]),
        warmup_syms=_as_int(p["warmup_syms"]),
        lock_min_ratio=_as_float(p["lock_min_ratio"]),
    ),
    "stream_to_vector": lambda c, p: c.blocks.stream_to_vector(
        c.gr.sizeof_gr_complex, _as_int(p["vlen"])
    ),
    "vector_to_stream": lambda c, p: c.blocks.vector_to_stream(
        c.gr.sizeof_gr_complex, _as_int(p["vlen"])
    ),
    "fft_vcc": lambda c, p: c.fft.fft_vcc(
        _as_int(p["fft_len"]),
        bool(p.get("forward", True)),
        c.fft.window.rectangular(_as_int(p["fft_len"])),
        bool(p.get("shift", False)),
        1,
    ),
    "blockinterleaver_cc": lambda c, p: c.blocks.blockinterleaver_cc(
        _as_int_list(p["perm"]), bool(p.get("mode", True)), False
    ),
    "blockinterleaver_ff": lambda c, p: c.blocks.blockinterleaver_ff(
        _as_int_list(p["perm"]), bool(p.get("mode", True)), False
    ),
    "keep_m_in_n_c": lambda c, p: c.blocks.keep_m_in_n(
        c.gr.sizeof_gr_complex,
        _as_int(p["m"]),
        _as_int(p["n"]),
        _as_int(p.get("offset", 0)),
    ),
    "delay_cc": lambda c, p: c.blocks.delay(
        c.gr.sizeof_gr_complex, _as_int(p["samples"])
    ),
    "multiply_conjugate_cc": lambda c, p: c.blocks.multiply_conjugate_cc(),
    "multiply_const_cc": lambda c, p: c.blocks.multiply_const_cc(
        complex(_as_float(p["re"]), _as_float(p["im"]))
    ),
    "constellation_soft_decoder": lambda c, p: c.digital.constellation_soft_decoder_cf(
        _const(c, p).base()
    ),
    "patterned_interleaver_f": lambda c, p: c.blocks.patterned_interleaver(
        c.gr.sizeof_float, _as_int_list(p["pattern"])
    ),
    "null_source_f": lambda c, p: c.blocks.null_source(c.gr.sizeof_float),
    "trellis_viterbi": lambda c, p: make_trellis_viterbi(
        c,
        rate_inv=_as_int(p["rate_inv"]),
        polys=_as_int_list(p["polys"]),
        frame_bits=_as_int(p["frame_bits"]),
        tail=_as_int(p["tail"]),
        k=_as_int(p["k"]),
    ),
    "keep_m_in_n_b": lambda c, p: c.blocks.keep_m_in_n(
        c.gr.sizeof_char, _as_int(p["m"]), _as_int(p["n"]), 0
    ),
    "polar_decode": lambda c, p: make_polar_decoder(
        c,
        block_size=_as_int(p["block_size"]),
        info_bits=_as_int(p["info_bits"]),
        frozen_positions=_as_int_list(p["frozen_positions"]),
        frozen_values=_as_int_list(p["frozen_values"]),
        list_size=_as_int(p["list_size"]),
    ),
    "ldpc_decode": lambda c, p: make_ldpc_decoder(
        c,
        block_size=_as_int(p["block_size"]),
        check_nodes=_unflatten(
            _as_int_list(p["check_flat"]), _as_int_list(p["check_lens"])
        ),
        max_iterations=_as_int(p["max_iterations"]),
    ),
    "burst_probe": lambda c, p: make_burst_probe(c.gr),
}


def _bind(fn: Callable[[_GrCtx, Params], Any], ctx: _GrCtx) -> Factory:
    return lambda p: fn(ctx, p)


def _factories(rate: float) -> dict[str, Factory]:
    ctx = _make_ctx(rate)
    return {kind: _bind(fn, ctx) for kind, fn in GR_BLOCKS.items()}
