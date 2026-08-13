from __future__ import annotations

import os
from pathlib import Path

from helpers._paths import REPO_ARTIFACTS

_DEFAULT_ROOT = REPO_ARTIFACTS / "assets"


def asset_root() -> Path:
    override = os.environ.get("MARCONI_ASSET_ROOT")
    return Path(override) if override else _DEFAULT_ROOT
