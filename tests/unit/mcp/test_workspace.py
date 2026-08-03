from __future__ import annotations

from pathlib import Path

import pytest

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
