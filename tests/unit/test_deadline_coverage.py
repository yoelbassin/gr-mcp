from __future__ import annotations

import ast

from helpers._paths import SRC_MARCONI

# A loop that pulls from a file is the shape that turns a 2.0s timeout into a
# 62.6s one, and it has recurred in four consecutive reviews: reactive bounding
# fixes the loop that was measured and never its siblings. So the rule is
# structural rather than another audit — a loop whose body reads a file must
# poll the deadline inside that body, or draw from an iterator that already
# does.
_READS = ("fromfile", "read", "readinto", "recv")
_POLLS = ("check_deadline", "bounded", "iter_iq", "iter_probes")


def _calls(node: ast.AST) -> set[str]:
    """Names called at or under `node`, not descending into nested function
    bodies — a helper defined inside the loop is not the loop's own polling."""
    found: set[str] = set()
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name):
            found.add(fn.id)
        elif isinstance(fn, ast.Attribute):
            found.add(fn.attr)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        found |= _calls(child)
    return found


def _loop_body(loop: ast.For | ast.While) -> ast.Module:
    return ast.Module(body=list(loop.body), type_ignores=[])


def test_every_streaming_loop_polls_the_deadline() -> None:
    offenders: list[str] = []
    for path in sorted(SRC_MARCONI.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While)):
                continue
            body = _loop_body(node)
            names = _calls(body)
            if not names & set(_READS):
                continue
            iter_names = _calls(node.iter) if isinstance(node, ast.For) else set()
            if names & set(_POLLS) or iter_names & set(_POLLS):
                continue
            rel = path.relative_to(SRC_MARCONI.parent.parent)
            offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "streaming loops with no deadline poll:\n  "
        + "\n  ".join(offenders)
        + f"\ncall check_deadline() in the body, or iterate {_POLLS[1]}(...)"
    )
