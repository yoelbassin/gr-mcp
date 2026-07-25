import ast
import subprocess
import sys
from pathlib import Path

VOCAB = [
    Path("src/marconi/errors.py"),
    Path("src/marconi/engine/coding/primitives.py"),
    Path("src/marconi/engine/io/bitfile.py"),
    Path("src/marconi/engine/io/specs.py"),
    Path("src/marconi/engine/stages/base.py"),
    Path("src/marconi/engine/types/descriptor.py"),
    Path("src/marconi/engine/types/levels.py"),
    Path("src/marconi/engine/types/models.py"),
    Path("src/marconi/engine/types/params.py"),
]


def test_vocabulary_imports_without_gnuradio_or_fastmcp() -> None:
    # A subprocess with import hooks blocking gnuradio + fastmcp still imports
    # the vocabulary modules.
    code = (
        "import builtins; _imp = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name.split('.')[0] in {'gnuradio', 'fastmcp'}:\n"
        "        raise ImportError(name)\n"
        "    return _imp(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        "import marconi.engine.types.levels, marconi.engine.types.descriptor\n"
        "import marconi.engine.stages.base\n"
        "import marconi.engine.types.models, marconi.engine.types.params\n"
        "import marconi.engine.io.specs, marconi.errors\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_no_gnuradio_import_in_vocabulary() -> None:
    for py in VOCAB:
        assert py.is_file(), f"vocabulary module missing: {py}"
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            )
            assert not any(m.split(".")[0] == "gnuradio" for m in mods), py
