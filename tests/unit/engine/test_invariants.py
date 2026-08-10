import ast
import subprocess
import sys
from pathlib import Path

from helpers._paths import SRC_MARCONI as _SRC

ENGINE = Path("src/marconi/engine")
BACKENDS = ENGINE / "backends"

VOCAB = [
    Path("src/marconi/errors.py"),
    Path("src/marconi/engine/coding/primitives.py"),
    Path("src/marconi/engine/io/bitfile.py"),
    Path("src/marconi/engine/stages/base.py"),
    Path("src/marconi/engine/types/descriptor.py"),
    Path("src/marconi/engine/types/levels.py"),
    Path("src/marconi/engine/types/models.py"),
    Path("src/marconi/engine/types/params.py"),
]


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
        "import marconi.engine.io.bitfile, marconi.errors\n"
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


def _is_list_annotation(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "list"
    if isinstance(node, ast.Subscript):
        return isinstance(node.value, ast.Name) and node.value.id == "list"
    return False


def _is_allowed_default(value: ast.expr | None) -> bool:
    if value is None:  # required
        return True
    if isinstance(value, ast.List):  # literal list default: only [] is neutral
        return len(value.elts) == 0
    if isinstance(value, ast.Call):  # Field(...) — no default, no positional arg
        return not value.args and not any(
            k.arg in ("default", "default_factory") for k in value.keywords
        )
    return False


def _step_classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
        and any(isinstance(b, ast.Name) and b.id == "Step" for b in n.bases)
    ]


def test_step_tables_are_never_populated_defaults() -> None:
    offenders: list[str] = []
    files = [p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts]
    assert files, f"no source under {_SRC}"
    for path in files:
        tree = ast.parse(path.read_text())
        for cls in _step_classes(tree):
            for field in cls.body:
                if not isinstance(field, ast.AnnAssign):
                    continue
                if _is_list_annotation(field.annotation) and not _is_allowed_default(
                    field.value
                ):
                    name = (
                        field.target.id if isinstance(field.target, ast.Name) else "?"
                    )
                    offenders.append(f"{path.relative_to(_SRC)}:{cls.name}.{name}")
    assert (
        not offenders
    ), "list/table params must be required (or empty-default):\n" + "\n".join(offenders)
