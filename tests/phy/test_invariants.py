import ast
import subprocess
import sys
from pathlib import Path

PHY = Path("packages/marconi-phy/src/marconi/phy")
BACKENDS = PHY / "backends"


def test_phy_imports_without_gnuradio() -> None:
    code = (
        "import builtins; _imp = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] == 'gnuradio':\n"
        "        raise ImportError(name)\n"
        "    return _imp(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import marconi.phy.ir, marconi.phy.models, marconi.phy.compile_context\n"
        "import marconi.phy.compiler\n"
        "import marconi.phy.backends.base, marconi.phy.backends.stub\n"
        "import marconi.phy.backends.gnuradio.blocks\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_no_gnuradio_import_outside_backends() -> None:
    # Spec invariant: every gnuradio import is lazy and under phy/backends/, so gnuradio
    # must not appear in any phy source OUTSIDE backends/. The pure backends are proven
    # gnuradio-free by the runtime guard above.
    py_files = [p for p in PHY.rglob("*.py") if BACKENDS not in p.parents]
    assert py_files, f"no phy source files found under {PHY}; path wrong or empty"
    for py in py_files:
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            assert not any(m.split(".")[0] == "gnuradio" for m in mods), py
