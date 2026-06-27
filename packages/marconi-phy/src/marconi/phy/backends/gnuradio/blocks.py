from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from marconi.core.params import ParamValue
from marconi.phy.backends.base import BackendError

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
}


def _bind(fn: Callable[[_GrCtx, Params], Any], ctx: _GrCtx) -> Factory:
    return lambda p: fn(ctx, p)


def _factories(rate: float) -> dict[str, Factory]:
    ctx = _make_ctx(rate)
    return {kind: _bind(fn, ctx) for kind, fn in GR_BLOCKS.items()}
