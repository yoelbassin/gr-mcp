import ast
import subprocess
import sys
from pathlib import Path

ENGINE = Path("src/marconi/engine")
BACKENDS = ENGINE / "backends"


def test_engine_imports_without_gnuradio() -> None:
    code = (
        "import builtins; _imp = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] == 'gnuradio':\n"
        "        raise ImportError(name)\n"
        "    return _imp(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import marconi.engine.compile.ir, marconi.engine.types.models\n"
        "import marconi.engine.compile.compile_context\n"
        "import marconi.engine.compile.compiler\n"
        "import marconi.engine.stages\n"
        "import marconi.engine.backends.base, marconi.engine.backends.stub\n"
        "import marconi.engine.backends.gnuradio.blocks\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_no_gnuradio_import_outside_backends() -> None:
    # Spec invariant: every gnuradio import is lazy and under engine/backends/, so
    # gnuradio must not appear in any engine source OUTSIDE backends/. The pure
    # backends are proven gnuradio-free by the runtime guard above.
    py_files = [p for p in ENGINE.rglob("*.py") if BACKENDS not in p.parents]
    assert py_files, f"no engine source files found under {ENGINE}; path wrong or empty"
    for py in py_files:
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            assert not any(m.split(".")[0] == "gnuradio" for m in mods), py
