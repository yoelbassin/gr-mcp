from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from marconi.errors import classify_error

P = ParamSpec("P")
R = TypeVar("R")


def tool_error_boundary(fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        from fastmcp.exceptions import ToolError

        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            code, message = classify_error(exc)
            raise ToolError(f"[{code}] {message}") from exc

    return wrapper
