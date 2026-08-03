from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from marconi.engine.compile.compiler import CompileError
from marconi.mcp.boundary import tool_error_boundary


def test_registered_exception_gets_stable_code() -> None:
    @tool_error_boundary
    def boom() -> None:
        raise CompileError("rx pipeline ends at IQ")

    with pytest.raises(ToolError, match=r"\[invalid_argument\] rx pipeline"):
        boom()


def test_unknown_exception_is_internal_error() -> None:
    @tool_error_boundary
    def boom() -> None:
        raise ArithmeticError("odd")

    with pytest.raises(ToolError, match=r"\[internal_error\] ArithmeticError"):
        boom()


def test_tool_error_passes_through_untouched() -> None:
    @tool_error_boundary
    def boom() -> None:
        raise ToolError("already shaped")

    with pytest.raises(ToolError, match=r"^already shaped$"):
        boom()


def test_return_value_and_metadata_survive() -> None:
    @tool_error_boundary
    def fine(x: int) -> int:
        """docstring survives"""
        return x + 1

    assert fine(1) == 2
    assert fine.__doc__ == "docstring survives"
