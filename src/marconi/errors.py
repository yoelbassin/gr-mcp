from __future__ import annotations

# Public exception types register their stable agent-facing [code] here at
# import (see register_error call sites in each package). classify_error is
# framework-free: the MCP boundary (marconi/mcp/boundary.py) builds on top of
# it — this root module must not import a server framework.
_REGISTRY: dict[type[Exception], str] = {}

_FALLBACK_CODES: tuple[tuple[type[Exception], str], ...] = (
    (ValueError, "invalid_argument"),
    (TypeError, "invalid_argument"),
    (FileNotFoundError, "not_found"),
    (RuntimeError, "runtime_error"),
)


def register_error(exc_type: type[Exception], code: str) -> None:
    _REGISTRY[exc_type] = code


def classify_error(exc: Exception) -> tuple[str, str]:
    """The most DERIVED registered type wins, resolved by the raised class's own
    MRO. Reducing over the registry with "subclass of the current best" only
    finds the answer when every match is one chain — with a second, unrelated
    match registered first it keeps that one, so the code an exception mapped to
    depended on module import order. These codes are the agent's stable
    contract; they cannot turn on which package imported first."""
    for cls in type(exc).__mro__:
        code = _REGISTRY.get(cls)
        if code is not None:
            return code, str(exc)

    for exc_type, fallback in _FALLBACK_CODES:
        if isinstance(exc, exc_type):
            return fallback, str(exc)
    return "internal_error", f"{type(exc).__name__}: {exc}"
