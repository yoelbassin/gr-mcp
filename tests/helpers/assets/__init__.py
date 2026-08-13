from __future__ import annotations

import os
from pathlib import Path

import pytest
from helpers.assets.manifest import Asset
from helpers.assets.resolve import ASSET_ROOT

SKIPPED: set[str] = set()

_FETCH_HINT = "PYTHONPATH=tests uv run python -m helpers.assets fetch"


def asset_root() -> Path:
    override = os.environ.get("MARCONI_ASSET_ROOT")
    return Path(override) if override else ASSET_ROOT


def asset_path(name: str) -> Path:
    return asset_root() / name


def require_asset(name: str) -> Path:
    path = asset_path(name)
    if path.exists():
        return path
    SKIPPED.add(name)
    pytest.skip(f"asset {name!r} absent; fetch with: {_FETCH_HINT}")


def strict_failures(index: dict[str, Asset], skipped: set[str]) -> list[str]:
    return sorted(n for n in skipped if n in index and index[n].ci_required)
