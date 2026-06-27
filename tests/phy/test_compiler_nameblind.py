import ast
from pathlib import Path

from phy._fixtures import fixture_registry

from marconi.phy.stages import stage_registry

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
    names = set(stage_registry()) | set(fixture_registry())
    assert {"fsk", "slice"} <= names, "real registry not represented in the guard"
    for path in COMPILER_FILES:
        assert path.exists(), f"compiler source not found: {path}"
        leaked = _string_literals(path) & names
        assert not leaked, f"{path} hard-codes stage names: {sorted(leaked)}"


def test_compiler_does_not_compare_conv_to_a_literal() -> None:
    # Name-independent: forbid `<expr>.conv == "..."` style coupling, which would
    # evade the literal scan if a future regression keyed on a real stage name.
    for path in COMPILER_FILES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute):
                assert (
                    node.left.attr != "conv"
                ), f"{path} compares a .conv attribute to a literal"
