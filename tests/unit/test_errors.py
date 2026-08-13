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


def test_fallback_codes_resolve_most_derived_not_table_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a broader builtin registered "before" FileNotFoundError must not shadow
    # it: the fallbacks walk the raised class's MRO, they are not a scan
    import marconi.errors as errors

    shadowed: dict[type[Exception], str] = {
        OSError: "io_error",
        **errors._FALLBACK_CODES,
    }
    monkeypatch.setattr(errors, "_FALLBACK_CODES", shadowed)
    assert classify_error(FileNotFoundError("x"))[0] == "not_found"
    assert classify_error(OSError("x"))[0] == "io_error"


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


def test_environment_failures_are_not_internal_errors() -> None:
    # "internal_error" tells the agent to stop and report a bug. A full disk, a
    # denied path, or an allocation the machine cannot serve are none of those:
    # they are retryable with a smaller slice or a fixed environment.
    cases: list[tuple[Exception, str]] = [
        (MemoryError("Unable to allocate 6.44 GiB"), "resource_exhausted"),
        (PermissionError(13, "denied"), "failed_precondition"),
        (OSError(28, "No space left on device"), "io_error"),
        (FileNotFoundError(2, "missing"), "not_found"),
    ]
    for exc, want in cases:
        code, _ = classify_error(exc)
        assert code == want, f"{type(exc).__name__} -> {code}, want {want}"
