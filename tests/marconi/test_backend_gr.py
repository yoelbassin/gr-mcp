import pytest

from marconi.backends import get_backend
from marconi.backends.base import Backend


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("imaginary")


def test_gnuradio_backend_resolves() -> None:
    b = get_backend("gnuradio")
    assert isinstance(b, Backend)
    assert b.name == "gnuradio"
    assert b.enumerate_devices() == []  # no hardware support in v1.0
