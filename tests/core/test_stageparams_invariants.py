import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "marconi"


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
    if isinstance(value, ast.Call):  # Field(...) — allowed iff no default= kwarg
        return not any(k.arg == "default" for k in value.keywords)
    return False


def _stage_params_classes(tree: ast.Module) -> list[ast.ClassDef]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef)
        and any(isinstance(b, ast.Name) and b.id == "StageParams" for b in n.bases)
    ]


def test_stageparams_tables_are_never_populated_defaults() -> None:
    offenders: list[str] = []
    files = [p for p in _SRC.rglob("*.py") if "__pycache__" not in p.parts]
    assert files, f"no source under {_SRC}"
    for path in files:
        tree = ast.parse(path.read_text())
        for cls in _stage_params_classes(tree):
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
