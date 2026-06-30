from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_BITS_SRC = (
    Path(__file__).resolve().parents[2]
    / "packages"
    / "marconi-bits"
    / "src"
    / "marconi"
    / "bits"
)


def _bits_modules() -> list[str]:
    mods = []
    for p in _BITS_SRC.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        rel = p.relative_to(_BITS_SRC).with_suffix("")
        mods.append("marconi.bits." + ".".join(rel.parts))
    return sorted(mods)


def test_bits_imports_without_gnuradio_or_fastmcp() -> None:
    lines = [
        "import builtins",
        "_real = builtins.__import__",
        "def _blocked(name, *a, **k):",
        "    if name.split('.')[0] in ('gnuradio', 'fastmcp'):",
        "        raise ImportError('blocked ' + name)",
        "    return _real(name, *a, **k)",
        "builtins.__import__ = _blocked",
    ]
    lines += [f"import {m}" for m in _bits_modules()]
    lines.append("print('ok')")
    r = subprocess.run(
        [sys.executable, "-c", "\n".join(lines)], capture_output=True, text=True
    )
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "ok", r.stderr


def test_no_gnuradio_import_anywhere_in_bits() -> None:
    for py in _BITS_SRC.rglob("*.py"):
        mods: list[str] = []
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
        assert not any(m.split(".")[0] == "gnuradio" for m in mods), py
