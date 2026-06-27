from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from marconi.core.params import ParamValue
from marconi.phy.backends.base import BackendError

Params = dict[str, ParamValue]
Factory = Callable[[Params], Any]


def _modules() -> tuple[Any, Any, Any, Any, Any]:
    """The single gnuradio import gate. Returns (gr, blocks, analog, digital,
    gr_filter). Called only at factory/build time, never at module import."""
    try:
        from gnuradio import analog, blocks, digital
        from gnuradio import filter as gr_filter
        from gnuradio import gr
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise BackendError(
            "GNU Radio is not importable. Install GNU Radio 3.10+ system-wide "
            "and use a `uv venv --system-site-packages` venv."
        ) from e
    return gr, blocks, analog, digital, gr_filter


@dataclass(frozen=True)
class _GrCtx:
    gr: Any
    blocks: Any
    analog: Any
    digital: Any
    gr_filter: Any
    rate: float


def _make_ctx(rate: float) -> _GrCtx:
    gr, blocks, analog, digital, gr_filter = _modules()
    return _GrCtx(
        gr=gr,
        blocks=blocks,
        analog=analog,
        digital=digital,
        gr_filter=gr_filter,
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
}


def _bind(fn: Callable[[_GrCtx, Params], Any], ctx: _GrCtx) -> Factory:
    return lambda p: fn(ctx, p)


def _factories(rate: float) -> dict[str, Factory]:
    ctx = _make_ctx(rate)
    return {kind: _bind(fn, ctx) for kind, fn in GR_BLOCKS.items()}
