from __future__ import annotations

import ast
from pathlib import Path

from helpers._paths import TESTS_ROOT

# A machine-specific root: /dev/null and the like are fine, a home directory
# is not — it exists only on the machine the test was written on.
_MACHINE_ROOTS = ("/Users/", "/home/", "/root/")


def _test_modules() -> list[Path]:
    here = Path(__file__).resolve()
    return [p for p in sorted(TESTS_ROOT.rglob("test_*.py")) if p.resolve() != here]


def test_no_gate_hardcodes_an_absolute_path() -> None:
    """An absolute asset path turns a moved or renamed checkout into a
    permanent skip whose reason still reads like a benign missing asset —
    which is exactly how one off-air CSS gate came to run on no machine at
    all, taking with it the no-SFO control its clock_correct sibling is
    measured against. Asset roots live in helpers._paths."""
    offenders: list[str] = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.startswith(_MACHINE_ROOTS):
                    rel = path.relative_to(TESTS_ROOT)
                    offenders.append(f"{rel}:{node.lineno} {node.value!r}")
    assert not offenders, "absolute paths in tests:\n" + "\n".join(offenders)
