from __future__ import annotations

from pathlib import Path

import pytest
from helpers.assets.manifest import Asset
from helpers.assets.root import asset_root

_FETCH_HINT = "PYTHONPATH=tests uv run python -m helpers.assets fetch"


def asset_path(name: str) -> Path:
    return asset_root() / name


def require_asset(name: str) -> Path:
    path = asset_path(name)
    if path.exists():
        return path
    pytest.skip(f"asset {name!r} absent; fetch with: {_FETCH_HINT}")


def asset_fixture(name: str) -> object:
    """`object`, not the concrete pytest type: the previous annotation
    imported the private _pytest.fixtures.FixtureFunctionDefinition, which
    did not exist before pytest 8.4 - any resolver landing on an older
    pytest collection-errored the whole suite through this one import."""

    @pytest.fixture(scope="module")
    def _resolved() -> Path:
        return require_asset(name)

    return _resolved


# The strict verdict reads the filesystem, never the skips a run happened to
# collect: a deselected gate, an xdist worker's private state or a typo'd
# asset name must not be able to turn a missing capture into a green run.
def missing_required(index: dict[str, Asset], root: Path) -> list[str]:
    return sorted(
        name
        for name, asset in index.items()
        if asset.ci_required and not (root / name).exists()
    )


__all__ = [
    "asset_fixture",
    "asset_path",
    "asset_root",
    "missing_required",
    "require_asset",
]
