from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from marconi.core.params import ParamValue
from marconi.phy.backends.base import BackendError
from marconi.phy.backends.gnuradio.embedded.preamble import (
    make_sym_acquire,
    make_sym_prepend,
)

Params = dict[str, ParamValue]
Factory = Callable[[Params], Any]


def _as_float(v: ParamValue) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BackendError(f"expected a real number, got {type(v).__name__}: {v!r}")
    return float(v)


def _as_int(v: ParamValue) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise BackendError(f"expected an integer, got {type(v).__name__}: {v!r}")
    return int(v)


def _as_float_list(v: ParamValue) -> list[float]:
    if not isinstance(v, list):
        raise BackendError(f"expected a list of numbers, got {type(v).__name__}: {v!r}")
    return [_as_float(x) for x in v]


def _modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    """The single gnuradio import gate. Returns (gr, blocks, analog, digital,
    gr_filter, firdes). Called only at factory/build time, never at module import."""
    try:
        from gnuradio import analog, blocks, digital
        from gnuradio import filter as gr_filter
        from gnuradio import gr
        from gnuradio.filter import firdes
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise BackendError(
            "GNU Radio is not importable. Install GNU Radio 3.10+ system-wide "
            "and use a `uv venv --system-site-packages` venv."
        ) from e
    return gr, blocks, analog, digital, gr_filter, firdes


@dataclass(frozen=True)
class _GrCtx:
    gr: Any
    blocks: Any
    analog: Any
    digital: Any
    gr_filter: Any
    firdes: Any
    rate: float


def _make_ctx(rate: float) -> _GrCtx:
    gr, blocks, analog, digital, gr_filter, firdes = _modules()
    return _GrCtx(
        gr=gr,
        blocks=blocks,
        analog=analog,
        digital=digital,
        gr_filter=gr_filter,
        firdes=firdes,
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
    "soft_bits_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_float, str(p["path"]), False
    ),
    # symbols are hard integer symbol indices (int16); exercised by a later
    # symbol-terminating vertical (CSS). Present here for _IO_BLOCKS coherence.
    "symbols_file_sink": lambda c, p: c.blocks.file_sink(
        c.gr.sizeof_short, str(p["path"]), False
    ),
    "quadrature_demod": lambda c, p: c.analog.quadrature_demod_cf(_as_float(p["gain"])),
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
    "complex_to_mag": lambda c, p: c.blocks.complex_to_mag(1),
    "multiply_const_ff": lambda c, p: c.blocks.multiply_const_ff(_as_float(p["value"])),
    "add_const_ff": lambda c, p: c.blocks.add_const_ff(_as_float(p["value"])),
    "float_to_complex": lambda c, p: c.blocks.float_to_complex(1),
    "sym_prepend": lambda c, p: make_sym_prepend(
        c.gr,
        _as_float_list(p["preamble_i"]),
        _as_float_list(p["preamble_q"]),
        _as_int(p["pad_symbols"]),
    ),
    "sym_acquire": lambda c, p: make_sym_acquire(
        c.gr,
        _as_float_list(p["preamble_i"]),
        _as_float_list(p["preamble_q"]),
        _as_int(p["pad_symbols"]),
        _as_float(p["threshold"]),
    ),
}


def _bind(fn: Callable[[_GrCtx, Params], Any], ctx: _GrCtx) -> Factory:
    return lambda p: fn(ctx, p)


def _factories(rate: float) -> dict[str, Factory]:
    ctx = _make_ctx(rate)
    return {kind: _bind(fn, ctx) for kind, fn in GR_BLOCKS.items()}
