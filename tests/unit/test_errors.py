import pytest

from marconi.errors import classify_error, register_error


class _Custom(Exception):
    pass


def test_known_types_get_stable_codes() -> None:
    assert classify_error(ValueError("x"))[0] == "invalid_argument"
    assert classify_error(FileNotFoundError("x"))[0] == "not_found"
    assert classify_error(RuntimeError("x"))[0] == "runtime_error"


def test_registered_type_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    # the registry is process-global; leaving _Custom in it would outlive
    # this test for the rest of the xdist worker
    import marconi.errors as errors

    monkeypatch.setattr(errors, "_REGISTRY", dict(errors._REGISTRY))
    register_error(_Custom, "custom_code")
    assert classify_error(_Custom("x"))[0] == "custom_code"


def test_unknown_type_is_internal_error_with_type_name() -> None:
    code, msg = classify_error(KeyError("k"))
    assert code == "internal_error" and "KeyError" in msg


def test_every_public_exception_classifies_not_internal() -> None:
    # Each package registers its public exceptions at import; a user's bad
    # spec/codec/wiring must never read as an internal bug.
    from marconi.engine.backends.base import BackendError
    from marconi.engine.compile.errors import CompileError
    from marconi.engine.stages.base import SpecValidationError, StageDirectionError
    from marconi.engine.types.models import ValidationIssue

    cases = [
        SpecValidationError([ValidationIssue(message="bad")], "modem"),
        StageDirectionError("psk_demod", "tx", frozenset({"rx"})),
        CompileError("unknown stage"),
        BackendError("unknown block kind"),
    ]
    for exc in cases:
        code, _ = classify_error(exc)
        assert code == "invalid_argument", f"{type(exc).__name__} -> {code}"


def test_no_fastmcp_import_in_core_errors() -> None:
    # core must not import a server framework.
    import inspect

    from marconi import errors

    assert "fastmcp" not in inspect.getsource(errors)
