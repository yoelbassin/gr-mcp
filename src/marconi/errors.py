from __future__ import annotations

from pydantic import ValidationError

# Public exception types register their stable agent-facing [code] here at
# import (see register_error call sites in each package). classify_error is
# framework-free: a tool layer, if one is added, builds its own boundary on top
# of it — this root module must not import a server framework.
_REGISTRY: dict[type, str] = {}


def register_error(exc_type: type, code: str) -> None:
    _REGISTRY[exc_type] = code


def classify_error(exc: Exception) -> tuple[str, str]:
    candidates: list[type] = [t for t in _REGISTRY if isinstance(exc, t)]
    if candidates:
        best = candidates[0]
        for t in candidates[1:]:
            if issubclass(t, best):
                best = t
        return _REGISTRY[best], str(exc)

    if isinstance(exc, ValidationError):
        return "invalid_argument", str(exc)
    if isinstance(exc, (ValueError, TypeError)):
        return "invalid_argument", str(exc)
    if isinstance(exc, FileNotFoundError):
        return "not_found", str(exc)
    if isinstance(exc, RuntimeError):
        return "runtime_error", str(exc)
    return "internal_error", f"{type(exc).__name__}: {exc}"
