from marconi.core.errors import classify_error, register_error


class _Custom(Exception):
    pass


def test_known_types_get_stable_codes() -> None:
    assert classify_error(ValueError("x"))[0] == "invalid_argument"
    assert classify_error(FileNotFoundError("x"))[0] == "not_found"
    assert classify_error(RuntimeError("x"))[0] == "runtime_error"


def test_registered_type_wins() -> None:
    register_error(_Custom, "custom_code")
    assert classify_error(_Custom("x"))[0] == "custom_code"


def test_unknown_type_is_internal_error_with_type_name() -> None:
    code, msg = classify_error(KeyError("k"))
    assert code == "internal_error" and "KeyError" in msg


def test_every_public_exception_classifies_not_internal() -> None:
    # Each package registers its public exceptions at import; a user's bad
    # spec/codec/wiring must never read as an internal bug (issue 11).
    from marconi.bits.compiler import CodecCompileError
    from marconi.core.models import ValidationIssue
    from marconi.core.stages import SpecValidationError, StageDirectionError
    from marconi.phy.backends.base import BackendError
    from marconi.phy.compiler import CompileError

    cases = [
        SpecValidationError([ValidationIssue(message="bad")], "modem"),
        StageDirectionError("psk_demod", "tx", frozenset({"rx"})),
        CompileError("unknown stage"),
        BackendError("unknown block kind"),
        CodecCompileError("unknown stage"),
    ]
    for exc in cases:
        code, _ = classify_error(exc)
        assert code == "invalid_argument", f"{type(exc).__name__} -> {code}"


def test_no_fastmcp_import_in_core_errors() -> None:
    # core must not import a server framework (issue 11).
    import inspect

    from marconi.core import errors

    assert "fastmcp" not in inspect.getsource(errors)
