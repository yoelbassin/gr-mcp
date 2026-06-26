from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import ValidationError

F = TypeVar("F", bound=Callable[..., Any])

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


def to_tool_error(exc: Exception) -> Any:
    from fastmcp.exceptions import ToolError

    code, message = classify_error(exc)
    return ToolError(f"[{code}] {message}")


def tool_error_boundary(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from fastmcp.exceptions import ToolError

        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise to_tool_error(exc) from exc

    return wrapper  # type: ignore[return-value]
