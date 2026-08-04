from __future__ import annotations

from marconi.errors import register_error


class CompileError(Exception):
    pass


register_error(CompileError, "invalid_argument")  # an uncompilable spec
