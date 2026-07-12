import ast
from pathlib import Path

from marconi.bits.registry import registry

_BITS = Path(__file__).resolve().parents[2] / "src" / "marconi" / "bits"
_ROUTING = [_BITS / "compiler.py", _BITS / "validate.py"]
_SRC = Path(__file__).resolve().parents[2] / "src" / "marconi"


def _bits_files() -> list[Path]:
    files = [p for p in _BITS.rglob("*.py") if "__pycache__" not in p.parts]
    assert files, f"no bits source under {_BITS}"
    return files


def _src_files() -> list[Path]:
    files = [p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts]
    assert files, f"no source under {_SRC}"
    return files


def _string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def test_routing_has_no_stage_name_literals() -> None:
    names = set(registry())
    assert {"crc", "parse"} <= names, "registry not represented in the guard"
    for path in _ROUTING:
        assert path.exists(), path
        leaked = _string_literals(path) & names
        assert not leaked, f"{path} hard-codes stage names: {sorted(leaked)}"


def test_no_conv_compared_to_a_literal() -> None:
    # Forbid `step.conv == "<name>"` anywhere in marconi (Yoda form too). A comparison
    # of .conv against a VARIABLE (e.g. models.py's `step.conv == conv`) is allowed.
    for path in _src_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            operands = [node.left, *node.comparators]
            has_conv = any(
                isinstance(o, ast.Attribute) and o.attr == "conv" for o in operands
            )
            has_str_const = any(
                isinstance(o, ast.Constant) and isinstance(o.value, str)
                for o in operands
            )
            assert not (
                has_conv and has_str_const
            ), f"{path} compares .conv to a string literal"
