from __future__ import annotations

import multiprocessing
import threading
from pathlib import Path
from typing import Any

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.gnuradio.blocks import _factories, _modules
from marconi.engine.backends.gnuradio.embedded.lifecycle import EofProbe
from marconi.engine.compile.ir import GrPipeline

_FILE_SOURCE_KINDS = frozenset(
    {"iq_file_source", "bits_file_source", "soft_bits_file_source"}
)


def _wire_eof_probe(pipeline: GrPipeline, instances: dict[str, Any]) -> None:
    """Hand every consenting block (declares an ``eof_probe`` attribute) a
    probe for the pipeline's file source, so bounded withheld margins can
    flush at EOF. Exactly-one non-repeating file source is the shape every
    pipeline has today; anything else conservatively gets no probe. A block
    fed DIRECTLY by the source gets ``expected_items`` — its own read counter
    then proves finality (see EofProbe)."""
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
    total = Path(str(spec.params["path"])).stat().st_size // item_bytes
    for bid, blk in consenting.items():
        into = [c for c in pipeline.connections if c.dst_block == bid]
        direct = len(into) == 1 and into[0].src_block == spec.id
        blk.eof_probe = EofProbe(src, total, total if direct else None)


def _guarded_top_block_cls(gr: Any) -> Any:
    class _GuardedTopBlock(gr.top_block):  # type: ignore[misc]
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
    gr = _modules()[0]
    factories = _factories(pipeline.sample_rate)
    tb = _guarded_top_block_cls(gr)(pipeline.name)
    instances: dict[str, Any] = {}
    for b in pipeline.blocks:
        factory = factories.get(b.kind)
        if factory is None:
            raise BackendError(
                f"block '{b.id}': kind '{b.kind}' has no GNU Radio factory"
            )
        try:
            instances[b.id] = factory(dict(b.params))
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
