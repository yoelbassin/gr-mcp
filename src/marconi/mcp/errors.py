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
from pydantic import ValidationError

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
        # v1.0: the only PermissionError through this boundary is the transmit
        # can_tx guard; revisit if file-I/O permission errors surface here.
        return "tx_forbidden", str(exc)
    if isinstance(exc, KeyError):
        # KeyError.__str__ wraps the message in repr quotes; strip a single
        # matching outer pair (str.strip would over-strip inner quotes).
        s = str(exc)
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1]
        return "not_found", s
    if isinstance(exc, FileNotFoundError):
        # a bad capture/file path is a common agent mistake
        return "not_found", str(exc)
    if isinstance(exc, BackendError):
        return "backend_error", str(exc)
    if isinstance(exc, ValidationError):
        # pydantic structural errors (malformed pipeline/scene dicts) — str(exc)
        # lists the offending fields, which is actionable for the agent
        return "invalid_argument", str(exc)
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
