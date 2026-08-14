from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("marconi")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0+unknown"
