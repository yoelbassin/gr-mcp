import ast
from pathlib import Path

from phy._fixtures import fixture_registry

COMPILER_FILES = [
    Path("packages/marconi-phy/src/marconi/phy/compiler.py"),
    Path("packages/marconi-phy/src/marconi/phy/compile_context.py"),
]


def _string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def test_compiler_has_no_stage_name_literals() -> None:
    stage_names = set(fixture_registry())
    assert stage_names, "fixture registry is empty; the guard would be vacuous"
    for path in COMPILER_FILES:
        assert path.exists(), f"compiler source not found: {path}"
        leaked = _string_literals(path) & stage_names
        assert not leaked, f"{path} hard-codes stage names: {sorted(leaked)}"
