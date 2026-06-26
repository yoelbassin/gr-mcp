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
