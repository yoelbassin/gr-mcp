import ast
from pathlib import Path

from marconi.engine.stages.general import GENERAL_STAGES
from marconi.engine.stages.registry import stage_registry

_FAMILIES = {"fsk", "psk", "qam", "ask", "ook", "ppm", "css"}
_GENERAL_SRC = Path("src/marconi/engine/stages/general.py")


def test_general_stages_declare_general_family() -> None:
    for cls in GENERAL_STAGES:
        assert cls.family == "general", f"{cls.__name__}: family {cls.family!r}"


def test_general_source_has_no_family_name_literal() -> None:
    assert _GENERAL_SRC.exists(), _GENERAL_SRC
    tree = ast.parse(_GENERAL_SRC.read_text())
    lits = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    leaked = lits & _FAMILIES
    assert not leaked, f"general.py references families: {sorted(leaked)}"


def test_registry_exposes_fsk_and_slice() -> None:
    reg = stage_registry()
    assert {"fsk", "slice"} <= set(reg)
    assert reg["slice"].family == "general"
    assert reg["fsk"].family == "fsk"
