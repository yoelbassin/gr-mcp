"""The block -> verdict channel, pinned from BOTH ends.

DiagnosticKey's own docstring records the failure it was built after: the
producer side is an embedded block's plain dict crossing a process boundary,
types cannot reach it, and a judged key that stops being produced degrades its
verdict to "uncertain" while the run still reports ok — which is how the
equalizer's lock statistic went unread. Documenting a trap is not enforcing it.

The producers are read from source rather than instantiated: every one of them
is a closure over a GR block class with its own geometry, so constructing them
here would restate nine signatures that have nothing to do with the contract
under test — and would skip the whole file wherever GNU Radio is absent.
"""

from __future__ import annotations

import ast
from pathlib import Path

import marconi.engine.backends.gnuradio.embedded as embedded
from marconi.engine.backends.base import DiagnosticKey

_EMBEDDED_DIR = Path(embedded.__file__).parent
_JUDGED = {k.value for k in DiagnosticKey}


def _diagnostic_key_names(tree: ast.AST) -> set[str]:
    """Judged keys this module names as DiagnosticKey.MEMBER."""
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "DiagnosticKey"
    }


def _diagnostics_string_keys(tree: ast.AST) -> set[str]:
    """Keys a module writes as bare strings: a seed dict entry, a subscript
    assignment, or the key argument to bump()."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            found |= {
                k.value
                for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
        elif isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                found.add(sl.value)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "bump":
            for arg in node.args[1:2]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value)
    return found


def _producer_modules() -> dict[str, ast.AST]:
    return {
        p.name: ast.parse(p.read_text())
        for p in sorted(_EMBEDDED_DIR.glob("*.py"))
        if p.name != "__init__.py"
    }


def test_every_judged_key_has_a_live_producer() -> None:
    """A judged key nothing emits is a verdict that can never fire, and the run
    still reports ok — the quality layer reads it as "untestable", not absent."""
    produced: set[str] = set()
    for tree in _producer_modules().values():
        produced |= {
            name
            for name in _diagnostic_key_names(tree)
            if getattr(DiagnosticKey, name, None) is not None
        }
    orphans = sorted(k.value for k in DiagnosticKey if k.name not in produced)
    assert not orphans, (
        f"DiagnosticKey members no embedded block produces: {orphans}. "
        "Either a producer was renamed/removed, or the key should leave the "
        "vocabulary — a judged key with no producer silently degrades its "
        "verdict while the run still reports ok."
    )


def test_producers_name_judged_keys_through_the_enum_never_as_strings() -> None:
    """The rename trap itself. A judged key written as a bare string on the
    producer side type-checks, ships, and is read by nothing: the consumers all
    take DiagnosticKey, so the lookup comes back empty and the verdict quietly
    weakens. Observability-only counters may be strings — they are judged by
    nobody — but a judged name must come from the enum."""
    offenders: list[str] = []
    for name, tree in _producer_modules().items():
        for key in _diagnostics_string_keys(tree) & _JUDGED:
            offenders.append(f"{name}: {key!r}")
    assert not offenders, (
        "judged diagnostic keys written as bare strings (use DiagnosticKey): "
        f"{sorted(offenders)}"
    )


def test_no_producer_emits_a_near_miss_of_a_judged_key() -> None:
    """A string key differing from a judged one only by case, separator or
    plural is almost certainly meant to BE it, and reads as nothing."""

    def normal(key: str) -> str:
        return key.lower().replace("_", "").rstrip("s")

    judged = {normal(v): v for v in _JUDGED}
    offenders: list[str] = []
    for name, tree in _producer_modules().items():
        for key in _diagnostics_string_keys(tree) - _JUDGED:
            match = judged.get(normal(key))
            if match is not None:
                offenders.append(f"{name}: {key!r} vs judged {match!r}")
    assert not offenders, (
        f"near-miss diagnostic keys the quality layer will never read: "
        f"{sorted(offenders)}"
    )
