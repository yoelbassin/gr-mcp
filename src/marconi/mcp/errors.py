"""Translate marconi's exception types into structured, agent-actionable tool
errors at the MCP boundary.

The message handed to the agent is always vocabulary-level and carries a stable
[code] prefix the agent can match on. PipelineValidationError already formats its
per-block, per-field issues into its message, so that detail survives."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from fastmcp.exceptions import ToolError

from marconi.backends import BackendError
from marconi.ops.transmit import TransmitNotConfirmedError
from marconi.vocabulary import PipelineValidationError

F = TypeVar("F", bound=Callable[..., Any])


def classify_error(exc: Exception) -> tuple[str, str]:
    """Map an exception to a stable (code, message) pair. Pure and testable."""
    if isinstance(exc, PipelineValidationError):
        return "validation_error", str(exc)
    if isinstance(exc, TransmitNotConfirmedError):
        return "tx_not_confirmed", str(exc)
    if isinstance(exc, PermissionError):
        return "tx_forbidden", str(exc)
    if isinstance(exc, KeyError):
        # KeyError.__str__ wraps the message in repr-quotes; unwrap them.
        return "not_found", str(exc).strip("'\"")
    if isinstance(exc, BackendError):
        return "backend_error", str(exc)
    if isinstance(exc, (ValueError, TypeError)):
        return "invalid_argument", str(exc)
    if isinstance(exc, RuntimeError):
        return "runtime_error", str(exc)
    return "internal_error", f"{type(exc).__name__}: {exc}"


def to_tool_error(exc: Exception) -> ToolError:
    code, message = classify_error(exc)
    return ToolError(f"[{code}] {message}")


def tool_error_boundary(fn: F) -> F:
    """Wrap a tool function so any marconi exception becomes a structured
    ToolError the MCP client (the agent) reliably sees."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001 — the boundary translates everything
            raise to_tool_error(exc) from exc

    return wrapper  # type: ignore[return-value]
