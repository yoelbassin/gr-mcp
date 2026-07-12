# tests/core/test_invariants.py
import ast
import subprocess
import sys
from pathlib import Path

CORE = Path("src/marconi/core")


def test_core_imports_without_gnuradio_or_fastmcp() -> None:
    # A subprocess with import hooks blocking gnuradio + fastmcp still imports core.
    code = (
        "import builtins; _imp = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] in {'gnuradio', 'fastmcp'}:\n"
        "        raise ImportError(name)\n"
        "    return _imp(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import marconi.core.levels, marconi.core.descriptor, marconi.core.stages\n"
        "import marconi.core.models, marconi.core.params\n"
        "import marconi.core.specs, marconi.core.errors\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_no_gnuradio_import_anywhere_in_core() -> None:
    py_files = list(CORE.rglob("*.py"))
    assert py_files, f"no core source files found under {CORE}; path wrong or empty"
    for py in py_files:
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            assert not any(m.split(".")[0] == "gnuradio" for m in mods), py
