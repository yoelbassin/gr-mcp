"""Architectural invariants for the coding engine: no gnuradio import (the
old bits-package guarantee, re-homed to phy/coding, its replacement as the
pure-python coding layer), plus a registry-driven sweep of every
CodingStage against the rules the compiler and codec-shape
validators lean on.

The old package's frame/window-seeding machinery and its extra level rung
above BITS are gone entirely, not just unused: the coding engine never
leaves BITS/SYMBOLS, and seeds_windows is the one flag a window-establishing
stage declares now.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from typing import Any

import pytest
from helpers._paths import SRC_MARCONI

from marconi.engine.coding.stages_bits import SyncWordStep
from marconi.engine.coding.stages_symbols import SymbolMapStep
from marconi.engine.compile.compiler import compile_pipeline
from marconi.engine.compile.errors import CompileError
from marconi.engine.stages.base import CodingStage
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

_CODING_SRC = SRC_MARCONI / "engine" / "coding"


def _coding_modules() -> list[str]:
    mods = []
    for p in _CODING_SRC.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        rel = p.relative_to(_CODING_SRC).with_suffix("")
        mods.append("marconi.engine.coding." + ".".join(rel.parts))
    return sorted(mods)


def test_coding_imports_without_gnuradio_or_fastmcp() -> None:
    lines = [
        "import builtins",
        "_real = builtins.__import__",
        "def _blocked(name, *a, **k):",
        "    if name.split('.')[0] in ('gnuradio', 'fastmcp'):",
        "        raise ImportError('blocked ' + name)",
        "    return _real(name, *a, **k)",
        "builtins.__import__ = _blocked",
    ]
    lines += [f"import {m}" for m in _coding_modules()]
    lines.append("print('ok')")
    r = subprocess.run(
        [sys.executable, "-c", "\n".join(lines)], capture_output=True, text=True
    )
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "ok", r.stderr


def test_no_gnuradio_import_anywhere_in_coding() -> None:
    for py in _CODING_SRC.rglob("*.py"):
        mods: list[str] = []
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
        assert not any(m.split(".")[0] == "gnuradio" for m in mods), py


def _coding_stages() -> dict[str, CodingStage[Any]]:
    return {
        name: stage
        for name, stage in stage_registry().items()
        if isinstance(stage, CodingStage)
    }


def test_coding_engine_has_stages() -> None:
    assert _coding_stages(), "expected at least one CodingStage in the registry"


def test_no_coding_stage_touches_the_frames_level() -> None:
    coding_levels = {Level.BITS, Level.SYMBOLS}
    for name, stage in _coding_stages().items():
        assert stage.from_level in coding_levels, (name, stage.from_level)
        assert stage.to_level in coding_levels, (name, stage.to_level)


def test_bits_entry_coding_stages_declare_hard_bit_input() -> None:
    for name, stage in _coding_stages().items():
        if stage.from_level is Level.BITS:
            assert stage.accepts_item_type == "b", name


def test_soft_bits_into_sync_word_is_a_compile_error() -> None:
    spec = Modem(
        symbol_rate=1.0,
        path=[SyncWordStep(sync="7e")],
    )
    with pytest.raises(CompileError, match="item_type"):
        compile_pipeline(
            spec,
            stage_registry(),
            sample_rate=1.0,
            start=Descriptor(Level.BITS, ItemType.F, carrier=Carrier.SOFT),
            source_io={},
            sink_io={},
        )


def test_soft_symbols_into_a_hard_symbol_stage_is_a_compile_error() -> None:
    spec = Modem(
        symbol_rate=1.0,
        path=[SymbolMapStep(code_bits=1, data_bits=1, table=[0, 1])],
    )
    with pytest.raises(CompileError, match="item_type"):
        compile_pipeline(
            spec,
            stage_registry(),
            sample_rate=1.0,
            start=Descriptor(Level.SYMBOLS, ItemType.F, carrier=Carrier.SOFT),
            source_io={},
            sink_io={},
        )
