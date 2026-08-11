from __future__ import annotations

import multiprocessing
import threading
from typing import Any

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.gnuradio.blocks import (
    BlockParams,
    _factories,
    _modules,
)
from marconi.engine.backends.gnuradio.embedded.lifecycle import EofProbe
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.io.source import SourceSlice

_FILE_SOURCE_KINDS = frozenset(
    {"iq_file_source", "bits_file_source", "soft_bits_file_source"}
)


def _wire_eof_probe(pipeline: GrPipeline, instances: dict[str, Any]) -> None:
    """Hand every consenting block (declares an ``eof_probe`` attribute) a
    probe for the pipeline's file source, so bounded withheld margins can
    flush at EOF. Exactly-one non-repeating file source is the shape every
    pipeline has today; anything else conservatively gets no probe.

    ``emitted`` is what the source actually feeds the graph — the file item
    count clipped by the source's own offset/length slice, NOT the raw file
    size (a sliced run stops the source early, so file size never proves
    finality). Compiler-built pipelines (terminal_sink marked) get an exact
    ``expected_items = emitted // decim`` for any block whose whole path back
    to the source is single-edged with non-increasing integer-ratio rate tags
    — ratio 1 included, so the plain complex_to_mag -> burst_sampler shape
    flushes its final burst; compiler tags come from the rate model and can
    only overstate a wire's rate, which withholds, never truncates early.
    Hand-built IR (no terminal mark) keeps the conservative last-edge rule:
    decim >= 2 only, since its tags carry no such guarantee. Anything else —
    interpolation in the path, a fork, unknown rates — gets None, which can
    only withhold a tail at EOF, never truncate one early."""
    sources = [
        b
        for b in pipeline.blocks
        if b.kind in _FILE_SOURCE_KINDS and not bool(b.params.get("repeat", False))
    ]
    consenting = {
        bid: inst for bid, inst in instances.items() if hasattr(inst, "eof_probe")
    }
    if len(sources) != 1 or not consenting:
        return
    spec = sources[0]
    src = instances[spec.id]
    item_bytes = int(src.output_signature().sizeof_stream_item(0))
    src_slice = SourceSlice.from_params(spec.params)
    file_count = src_slice.path.stat().st_size // item_bytes
    avail = max(0, file_count - src_slice.offset)
    emitted = avail if src_slice.length == 0 else min(src_slice.length, avail)
    src_rate = pipeline.sample_rate
    for bid, blk in consenting.items():
        into = [c for c in pipeline.connections if c.dst_block == bid]
        expected: int | None = None
        if len(into) == 1:
            if into[0].src_block == spec.id:
                expected = emitted  # source-adjacent: read counter proves finality
            elif pipeline.terminal_sink is not None:
                decim = _integer_decim_path(pipeline, spec.id, bid)
                if decim is not None:
                    expected = emitted // decim
            else:
                # Hand-built IR carries no rate-model guarantee: finality only
                # when the last edge's tag is an exact integer DECIMATION
                # (>=2) of the source — a ratio-1 tag could mislabel an
                # interpolated wire, and an understated expected would flush
                # a withheld margin mid-stream.
                rate = pipeline.block(bid).sample_rate
                if rate is not None and rate > 0:
                    ratio = src_rate / rate
                    decim = round(ratio)
                    if decim >= 2 and abs(ratio - decim) < 1e-6:
                        expected = emitted // decim
        blk.eof_probe = EofProbe(src, emitted, expected)


def _integer_decim_path(pipeline: GrPipeline, src_id: str, bid: str) -> int | None:
    """Net integer decimation from the file source to ``bid`` along a unique
    single-input path whose rate tags never increase and step by integer
    ratios (ratio 1 allowed). None when any hop forks, lacks a tag, or lands
    off-grid — interpolation anywhere upstream disqualifies, because a
    composed interp/decim chain's true delivered count drifts off the
    net-ratio arithmetic (floor composition), and an understated expected
    would flush early."""
    decim = 1
    cur = bid
    while True:
        into = [c for c in pipeline.connections if c.dst_block == cur]
        if len(into) != 1:
            return None
        prev = into[0].src_block
        r_cur = pipeline.block(cur).sample_rate
        if r_cur is None or r_cur <= 0:
            return None
        upstream = (
            pipeline.sample_rate if prev == src_id else pipeline.block(prev).sample_rate
        )
        if upstream is None or upstream <= 0:
            return None
        step = upstream / r_cur
        step_i = round(step)
        if step_i < 1 or abs(step - step_i) > 1e-6:
            return None
        decim *= step_i
        if prev == src_id:
            return decim
        cur = prev


def _guarded_top_block_cls(gr: Any) -> Any:
    class _GuardedTopBlock(gr.top_block):
        _uint8_py_blocks = False

        def run(self, *args: Any, **kwargs: Any) -> None:
            on_main_thread = threading.current_thread() is threading.main_thread()
            in_subprocess_worker = multiprocessing.parent_process() is not None
            if self._uint8_py_blocks and on_main_thread and not in_subprocess_worker:
                raise BackendError(
                    "uint8-output embedded Python blocks segfault under a "
                    "main-thread tb.run(); drive this pipeline through "
                    "run_pipeline (off-main-thread runner)"
                )
            super().run(*args, **kwargs)

    return _GuardedTopBlock


def build_top_block(pipeline: GrPipeline) -> Any:
    """Resolve every block kind via the factories and wire all connections into
    a gr.top_block. Imports gnuradio (lazily) into the calling process."""
    gr = _modules().gr
    factories = _factories()
    tb = _guarded_top_block_cls(gr)(pipeline.name)
    instances: dict[str, Any] = {}
    for b in pipeline.blocks:
        factory = factories.get(b.kind)
        if factory is None:
            raise BackendError(
                f"block '{b.id}': kind '{b.kind}' has no GNU Radio factory"
            )
        try:
            instances[b.id] = factory(BlockParams(dict(b.params)))
        except Exception as e:  # noqa: BLE001
            raise BackendError(
                f"block '{b.id}' ({b.kind}) failed to construct: {e}"
            ) from e
    for c in pipeline.connections:
        try:
            tb.connect(
                (instances[c.src_block], c.src_port),
                (instances[c.dst_block], c.dst_port),
            )
        except KeyError as e:
            raise BackendError(f"connection references unknown block {e}") from e
        except Exception as e:  # noqa: BLE001
            raise BackendError(
                f"connecting {c.src_block}:{c.src_port} -> "
                f"{c.dst_block}:{c.dst_port} failed: {e}"
            ) from e
    _wire_eof_probe(pipeline, instances)
    # Python-defined basic_block subclasses use a pybind11 trampoline whose
    # lifetime is tied to the Python wrapper object.  Without this anchor the
    # GC can collect those wrappers after build_top_block returns, leaving GR's
    # C++ scheduler with a dangling pointer → SIGSEGV on tb.run().
    tb._py_instances = instances
    gw = getattr(gr, "gateway", None)
    base = getattr(gw, "gateway_block", None) if gw is not None else None
    tb._uint8_py_blocks = base is not None and any(
        isinstance(i, base) and i.output_signature().sizeof_stream_item(0) == 1
        for i in instances.values()
    )
    return tb
