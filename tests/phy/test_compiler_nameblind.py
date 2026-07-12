import ast
from pathlib import Path

from phy._fixtures import fixture_registry

from marconi.phy.stages import stage_registry

COMPILER_FILES = [
    Path("src/marconi/phy/compiler.py"),
    Path("src/marconi/phy/compile_context.py"),
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
    # Name-independent: forbid any `.conv` comparison (e.g. `step.conv == "fsk"`),
    # which would evade the literal scan if a regression keyed on a real stage
    # name. Walk BOTH sides so a Yoda-style `"fsk" == step.conv` is caught too.
    for path in COMPILER_FILES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Attribute):
                    assert (
                        operand.attr != "conv"
                    ), f"{path} uses a .conv attribute in a comparison"
