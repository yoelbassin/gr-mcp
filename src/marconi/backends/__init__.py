from marconi.backends.base import Backend, BackendError

__all__ = ["Backend", "BackendError", "get_backend"]


def get_backend(name: str = "gnuradio") -> Backend:
    """Resolve a backend by name; imports lazily so `import marconi` never
    pulls in an engine."""
    if name == "gnuradio":
        from marconi.backends.gnuradio_backend import GnuRadioBackend

        return GnuRadioBackend()
    raise ValueError(f"unknown backend '{name}'; available: gnuradio")
