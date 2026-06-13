import pytest
from fastmcp.exceptions import ToolError

from marconi.backends import BackendError
from marconi.mcp.errors import classify_error, tool_error_boundary
from marconi.models import ValidationIssue
from marconi.ops.transmit import TransmitNotConfirmedError
from marconi.vocabulary import PipelineValidationError


def test_classify_validation_error():
    exc = PipelineValidationError(
        [ValidationIssue(block_id="x", field="freq", message="bad")]
    )
    code, message = classify_error(exc)
    assert code == "validation_error"
    assert "bad" in message


def test_classify_tx_not_confirmed():
    assert (
        classify_error(TransmitNotConfirmedError("need confirm"))[0]
        == "tx_not_confirmed"
    )


def test_classify_permission():
    assert classify_error(PermissionError("cannot tx"))[0] == "tx_forbidden"


def test_classify_key_error_unwraps_quotes():
    code, message = classify_error(KeyError("unknown device 'sim0'"))
    assert code == "not_found"
    assert message == "unknown device 'sim0'"


def test_classify_file_not_found():
    code, message = classify_error(FileNotFoundError("no such file: x.sigmf-meta"))
    assert code == "not_found"
    assert "x.sigmf-meta" in message


def test_classify_backend_error():
    assert classify_error(BackendError("engine blew up"))[0] == "backend_error"


def test_classify_value_and_type_errors():
    assert classify_error(ValueError("bad arg"))[0] == "invalid_argument"
    assert classify_error(TypeError("wrong type"))[0] == "invalid_argument"


def test_classify_runtime_error():
    assert classify_error(RuntimeError("render failed"))[0] == "runtime_error"


def test_boundary_translates_and_reraises():
    @tool_error_boundary
    def boom():
        raise ValueError("nope")

    with pytest.raises(ToolError) as ei:
        boom()
    assert "[invalid_argument]" in str(ei.value)
    assert "nope" in str(ei.value)


def test_boundary_passes_through_success():
    @tool_error_boundary
    def ok():
        return {"answer": 42}

    assert ok() == {"answer": 42}


def test_boundary_does_not_double_wrap_tool_error():
    @tool_error_boundary
    def already():
        raise ToolError("[custom] already structured")

    with pytest.raises(ToolError) as ei:
        already()
    assert str(ei.value).count("[custom]") == 1
