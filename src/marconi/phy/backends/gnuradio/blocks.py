from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from marconi.core.params import ParamValue
from marconi.phy.backends.base import BackendError
from marconi.phy.backends.gnuradio.embedded.chirp import (
    chirp_prefix,
    dechirp_ref,
    make_chirp_mod,
    make_chirp_sync,
    make_css_demap,
    make_css_map,
)
from marconi.phy.backends.gnuradio.embedded.cp_sync import make_cp_symbol_sync
from marconi.phy.backends.gnuradio.embedded.decision import make_peak_decision
from marconi.phy.backends.gnuradio.embedded.depuncture import make_depuncture
from marconi.phy.backends.gnuradio.embedded.msk import make_msk_demod
from marconi.phy.backends.gnuradio.embedded.ofdm import make_ofdm_frame_sync
from marconi.phy.backends.gnuradio.embedded.ofdm_coherent import (
    make_ofdm_coherent_sync,
)
from marconi.phy.backends.gnuradio.embedded.preamble import make_sym_strip, sym_prefix
from marconi.phy.backends.gnuradio.embedded.probe import make_burst_probe
from marconi.phy.backends.gnuradio.embedded.trellis_fec import make_trellis_viterbi

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


def _modules() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Single gnuradio import gate → (gr, blocks, analog, digital,
    gr_filter, firdes, pfb, fft, trellis). Called at build time only."""
    try:
        from gnuradio import analog, blocks, digital, fft
        from gnuradio import filter as gr_filter
        from gnuradio import gr, trellis
        from gnuradio.filter import firdes, pfb
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise BackendError(
            "GNU Radio is not importable. Install GNU Radio 3.10+ system-wide "
            "and use a `uv venv --system-site-packages` venv."
        ) from e
    return gr, blocks, analog, digital, gr_filter, firdes, pfb, fft, trellis


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
    rate: float


def _make_ctx(rate: float) -> _GrCtx:
    gr, blocks, analog, digital, gr_filter, firdes, pfb, fft, trellis = _modules()
    return _GrCtx(
        gr=gr,
        blocks=blocks,
        analog=analog,
        digital=digital,
        gr_filter=gr_filter,
        firdes=firdes,
        pfb=pfb,
        fft=fft,
        trellis=trellis,
        rate=rate,
    )


def _const(c: _GrCtx, scheme: str, order: int) -> Any:
    if scheme == "psk":
        builders = {
            2: c.digital.constellation_bpsk,
            4: c.digital.constellation_qpsk,
            8: c.digital.constellation_8psk,
        }
        if order not in builders:
            raise BackendError(f"unsupported psk order {order}")
        return builders[order]()
    if scheme == "qam":
        if order == 16:
            return c.digital.constellation_16qam()
        if order == 64:
            return c.digital.qam.qam_constellation(
                constellation_points=64,
                differential=False,
                mod_code=c.digital.mod_codes.GRAY_CODE,
                large_ampls_to_corners=False,
            )
        raise BackendError(f"unsupported qam order {order}")
    raise BackendError(f"unknown constellation scheme {scheme!r}")


def _keep_m_in_n_f(c: _GrCtx, p: Params) -> Any:
    blk = c.blocks.keep_m_in_n(
        c.gr.sizeof_float, _as_int(p["m"]), _as_int(p["n"]), _as_int(p.get("offset", 0))
    )
    if not bool(p.get("propagate_tags", True)):
        blk.set_tag_propagation_policy(c.gr.TPP_DONT)
    return blk


# kind -> (ctx, params) -> live GR block. The ONLY GR-aware vocabulary in phy.
GR_BLOCKS: dict[str, Callable[[_GrCtx, Params], Any]] = {
    "iq_file_source": lambda c, p: c.blocks.file_source(
        c.gr.sizeof_gr_complex, str(p["path"]), bool(p.get("repeat", False))
    ),
    "iq_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_gr_complex, str(p["path"]), False
    ),
    "bits_file_source": lambda c, p: c.blocks.file_source(
        c.gr.sizeof_char, str(p["path"]), bool(p.get("repeat", False))
    ),
    "bits_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_char, str(p["path"]), False
    ),
    "soft_bits_file_source": lambda c, p: c.blocks.file_source(
        c.gr.sizeof_float, str(p["path"]), bool(p.get("repeat", False))
    ),
    "soft_bits_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_float, str(p["path"]), False
    ),
    # symbols are hard integer symbol indices (int16); exercised by a later
    # symbol-terminating vertical (CSS). Present here for _IO_BLOCKS coherence.
    "symbols_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_short, str(p["path"]), False
    ),
    "quadrature_demod": lambda c, p: c.analog.quadrature_demod_cf(_as_float(p["gain"])),
    "msk_demod": lambda c, p: make_msk_demod(
        c.gr, sps=_as_float(p["sps"]), loop_bw=_as_float(p.get("loop_bw", 0.0038))
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
            _as_int(p["sps"]) * _as_int(p.get("span", 11)) + 1,
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
    # Rational resample via the polyphase arbitrary resampler (rate=interp/decim).
    # This GR 3.10.12 build's rational_resampler_ccf/ccc with no taps image/alias:
    # BER ~0.46 (random) through the fsk round-trip at 8/7 and 8/9, even though a
    # bare tone's RMS survives (energy preservation != spectral purity). The pfb
    # arbitrary resampler auto-designs proper anti-imaging taps and is the
    # BER-0-proven path (test_resample_roundtrip).
    "pfb_arb_resampler_ccf": lambda c, p: c.pfb.arb_resampler_ccf(_as_float(p["rate"])),
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
    "constellation_receiver_cb": lambda c, p: c.digital.constellation_receiver_cb(
        _const(c, str(p["scheme"]), _as_int(p["order"])).base(),
        _as_float(p.get("loop_bw", 0.04)),
        _as_float(p.get("fmin", -0.5)),
        _as_float(p.get("fmax", 0.5)),
    ),
    "chunks_to_symbols_bc": lambda c, p: c.digital.chunks_to_symbols_bc(
        _const(c, str(p["scheme"]), _as_int(p["order"])).points()
    ),
    "constellation_decoder_cb": lambda c, p: c.digital.constellation_decoder_cb(
        _const(c, str(p["scheme"]), _as_int(p["order"])).base()
    ),
    "pack_k_bits_bb": lambda c, p: c.blocks.pack_k_bits_bb(_as_int(p["k"])),
    "unpack_k_bits_bb": lambda c, p: c.blocks.unpack_k_bits_bb(_as_int(p["k"])),
    "complex_to_mag": lambda c, p: c.blocks.complex_to_mag(_as_int(p.get("vlen", 1))),
    "dc_blocker_ff": lambda c, p: c.gr_filter.dc_blocker_ff(_as_int(p["d"]), True),
    "hilbert_fc": lambda c, p: c.gr_filter.hilbert_fc(_as_int(p["ntaps"])),
    "multiply_const_ff": lambda c, p: c.blocks.multiply_const_ff(_as_float(p["value"])),
    "add_const_ff": lambda c, p: c.blocks.add_const_ff(_as_float(p["value"])),
    "float_to_complex": lambda c, p: c.blocks.float_to_complex(1),
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
    "ofdm_coherent_sync": lambda c, p: make_ofdm_coherent_sync(
        c.gr,
        fft_len=_as_int(p["fft_len"]),
        cp_len=_as_int(p["cp_len"]),
        sym_len=_as_int(p["sym_len"]),
        n_frame_syms=_as_int(p["n_frame_syms"]),
        n_carriers=_as_int(p["n_carriers"]),
        kmin=_as_int(p["kmin"]),
        dc_search=_as_int(p["dc_search"]),
        warmup_syms=_as_int(p["warmup_syms"]),
        pilot_lens=_as_int_list(p["pilot_lens"]),
        pilot_carriers=_as_int_list(p["pilot_carriers"]),
        pilot_i=_as_float_list(p["pilot_i"]),
        pilot_q=_as_float_list(p["pilot_q"]),
        fp_carriers=_as_int_list(p["fp_carriers"]),
        fp_i=_as_float_list(p["fp_i"]),
        fp_q=_as_float_list(p["fp_q"]),
    ),
    "cp_symbol_sync": lambda c, p: make_cp_symbol_sync(
        c.gr,
        fft_len=_as_int(p["fft_len"]),
        cp_len=_as_int(p["cp_len"]),
        warmup_syms=_as_int(p["warmup_syms"]),
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
    "constellation_soft_decoder": lambda c, p: c.digital.constellation_soft_decoder_cf(
        _const(c, str(p["scheme"]), _as_int(p["order"])).base()
    ),
    "depuncture": lambda c, p: make_depuncture(
        c.gr, keep_mask=_as_int_list(p["keep_mask"])
    ),
    "trellis_viterbi": lambda c, p: make_trellis_viterbi(
        c,
        rate_inv=_as_int(p["rate_inv"]),
        polys=_as_int_list(p["polys"]),
        frame_bits=_as_int(p["frame_bits"]),
        tail=_as_int(p["tail"]),
    ),
    "keep_m_in_n_b": lambda c, p: c.blocks.keep_m_in_n(
        c.gr.sizeof_char, _as_int(p["m"]), _as_int(p["n"]), 0
    ),
    "burst_probe": lambda c, p: make_burst_probe(c.gr),
}


def _bind(fn: Callable[[_GrCtx, Params], Any], ctx: _GrCtx) -> Factory:
    return lambda p: fn(ctx, p)


def _factories(rate: float) -> dict[str, Factory]:
    ctx = _make_ctx(rate)
    return {kind: _bind(fn, ctx) for kind, fn in GR_BLOCKS.items()}
