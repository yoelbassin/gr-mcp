"""The MCP tool functions: thin, synchronous marshalling over the marconi ops.

Each function takes JSON-friendly args, calls into the marconi library, and
returns plain dicts/lists. Captures are referenced by their .sigmf-data path
string (CaptureRef.path); _ref() reconstructs the CaptureRef from the sidecar.
Every function is wrapped by tool_error_boundary and registered in TOOLS."""

from __future__ import annotations

from collections.abc import Callable

from marconi import sigmf
from marconi.mcp.errors import tool_error_boundary
from marconi.models import CaptureRef
from marconi.vocabulary import VOCABULARY


def _ref(capture_path: str) -> CaptureRef:
    """Reconstruct a CaptureRef from a capture's .sigmf-data path."""
    return sigmf.read_meta(capture_path)


@tool_error_boundary
def list_blocks() -> dict:
    """The curated block vocabulary for composing pipelines. For each block
    type: input/output dtypes ('c'=complex, 'f'=float) and its parameters
    (name, type, required, default). Compose pipelines ONLY from these types."""
    out: dict = {}
    for name, d in VOCABULARY.items():
        out[name] = {
            "inputs": list(d.inputs),
            "outputs": list(d.outputs),
            "params": [
                {
                    "name": p.name,
                    "type": p.type.__name__,
                    "required": p.required,
                    "default": p.default,
                }
                for p in d.params
            ],
        }
    return out


# Tools are added to this registry as later tasks implement them.
TOOLS: dict[str, Callable] = {
    "list_blocks": list_blocks,
}
