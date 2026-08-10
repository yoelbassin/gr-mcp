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
    candidates: list[type[Exception]] = [t for t in _REGISTRY if isinstance(exc, t)]
    if candidates:
        best = candidates[0]
        for t in candidates[1:]:
            if issubclass(t, best):
                best = t
        return _REGISTRY[best], str(exc)

    for exc_type, code in _FALLBACK_CODES:
        if isinstance(exc, exc_type):
            return code, str(exc)
    return "internal_error", f"{type(exc).__name__}: {exc}"
