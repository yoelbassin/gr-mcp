from __future__ import annotations

# Public exception types register their stable agent-facing [code] here at
# import (see register_error call sites in each package). classify_error is
# framework-free: the MCP boundary (marconi/mcp/boundary.py) builds on top of
# it — this root module must not import a server framework.
_REGISTRY: dict[type[Exception], str] = {}

# Codes for the builtins nothing registers explicitly. Resolved by the SAME
# most-derived-wins rule as the registry, not by the order written here: a
# hand-ordered first-match scan is the very import-order dependence
# classify_error exists to remove, one table further down (registering OSError
# above FileNotFoundError would have made "not_found" unreachable).
#
# The environment entries matter as much as the argument ones: "internal_error"
# tells the agent to stop and report a bug, so a full disk, a denied path or an
# oversized allocation must not wear it — those are retryable with a smaller
# slice or a fixed environment, and they were all landing there.
_FALLBACK_CODES: dict[type[Exception], str] = {
    ValueError: "invalid_argument",
    TypeError: "invalid_argument",
    FileNotFoundError: "not_found",
    RuntimeError: "runtime_error",
    MemoryError: "resource_exhausted",
    PermissionError: "failed_precondition",
    OSError: "io_error",
}


def register_error(exc_type: type[Exception], code: str) -> None:
    _REGISTRY[exc_type] = code


def classify_error(exc: Exception) -> tuple[str, str]:
    """The most DERIVED type wins, resolved by the raised class's own MRO across
    the registry and then the builtin fallbacks. Reducing over either table with
    "subclass of the current best" only finds the answer when every match is one
    chain — with a second, unrelated match seen first it keeps that one, so the
    code an exception mapped to depended on module import order (registry) or on
    the order a table was typed (fallbacks). These codes are the agent's stable
    contract; they cannot turn on either."""
    for cls in type(exc).__mro__:
        code = _REGISTRY.get(cls)
        if code is not None:
            return code, str(exc)

    for cls in type(exc).__mro__:
        fallback = _FALLBACK_CODES.get(cls)
        if fallback is not None:
            return fallback, str(exc)
    return "internal_error", f"{type(exc).__name__}: {exc}"
