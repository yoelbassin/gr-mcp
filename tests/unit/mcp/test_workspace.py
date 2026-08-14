from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from marconi.mcp.tools import TOOLS
from marconi.mcp.workspace import new_run_dir, workspace_root


def test_root_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    assert workspace_root() == tmp_path


def test_run_dirs_are_fresh_and_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    a, b = new_run_dir("rx"), new_run_dir("rx")
    assert a != b
    assert a.is_dir() and b.is_dir()
    assert a.parent == tmp_path / "marconi-runs"
    assert a.name.startswith("rx-")


def test_docstrings_name_the_directory_the_writing_tools_actually_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_rx and capture told the agent their output was in ./marconi-runs/ —
    the CWD default this module deliberately abandoned, while read_stream in
    the same surface named the real one. The claim is checked against the
    directory new_run_dir returns under both roots, not against itself."""
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    assert new_run_dir("rx").parent == tmp_path / "marconi-runs"
    monkeypatch.delenv("MARCONI_WORKSPACE")
    assert workspace_root() == Path("~/.cache/marconi").expanduser()

    docs = {
        name: " ".join((inspect.getdoc(fn) or "").split()) for name, fn in TOOLS.items()
    }
    naming = {name for name, doc in docs.items() if "marconi-runs" in doc}
    assert {"run_rx", "capture", "read_stream"} <= naming
    for name in sorted(naming):
        doc = docs[name]
        assert "./marconi-runs" not in doc, name
        assert "$MARCONI_WORKSPACE" in doc, name
        assert "~/.cache/marconi" in doc, name


def test_a_run_dir_holding_only_empty_subdirs_is_discarded(tmp_path: Path) -> None:
    """ "Nothing written" means no FILE at any depth, not an empty top level: a
    trace=True run mkdirs run_dir/trace before the graph starts, so a plain
    rmdir failed on the very case this exists for."""
    from marconi.mcp.workspace import discarded_if_unused

    run_dir = tmp_path / "rx-abc"
    run_dir.mkdir()
    with discarded_if_unused(run_dir) as d:
        (d / "trace").mkdir()
    assert not run_dir.exists()


def test_a_run_dir_holding_a_nested_file_is_kept(tmp_path: Path) -> None:
    from marconi.mcp.workspace import discarded_if_unused

    run_dir = tmp_path / "rx-def"
    run_dir.mkdir()
    with discarded_if_unused(run_dir) as d:
        (d / "trace").mkdir()
        (d / "trace" / "stage_0.f32").write_bytes(b"\x00\x00\x00\x00")
    assert run_dir.exists(), "partial output is evidence"


def test_eviction_leaves_room_for_the_entry_about_to_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "make room BEFORE writing, so the budget bounds the peak" was
    aspirational: the incoming entry was not counted, so the cache settled at
    budget + one slice, and a slice can be gigabytes."""
    from marconi.mcp.workspace import conversion_cache_dir, evict_conversion_cache

    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    cache = conversion_cache_dir()
    for i in range(4):
        (cache / f"e{i}.cf32").write_bytes(b"\x00" * 1000)

    def _held() -> int:
        return sum(p.stat().st_size for p in cache.iterdir() if p.is_file())

    evict_conversion_cache(budget=4000)
    assert _held() == 4000, "already fits: nothing to do"
    evict_conversion_cache(budget=4000, incoming=2000)
    assert _held() <= 2000, "must leave room for the 2000 bytes about to land"
