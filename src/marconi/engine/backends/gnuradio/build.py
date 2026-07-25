from __future__ import annotations

from typing import Any

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.gnuradio.blocks import _factories, _modules
from marconi.engine.ir import GrPipeline


def build_top_block(pipeline: GrPipeline) -> Any:
    """Resolve every block kind via the factories and wire all connections into
    a gr.top_block. Imports gnuradio (lazily) into the calling process."""
    gr = _modules()[0]
    factories = _factories(pipeline.sample_rate)
    tb = gr.top_block(pipeline.name)
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
    # Python-defined basic_block subclasses use a pybind11 trampoline whose
    # lifetime is tied to the Python wrapper object.  Without this anchor the
    # GC can collect those wrappers after build_top_block returns, leaving GR's
    # C++ scheduler with a dangling pointer → SIGSEGV on tb.run().
    tb._py_instances = instances
    return tb
