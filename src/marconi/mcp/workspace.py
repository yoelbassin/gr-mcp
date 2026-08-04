from __future__ import annotations

import os
import uuid
from pathlib import Path


def workspace_root() -> Path:
    return Path(os.environ.get("MARCONI_WORKSPACE", ".")).resolve()


def new_run_dir(label: str) -> Path:
    d = workspace_root() / "marconi-runs" / f"{label}-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def conversion_cache_dir() -> Path:
    d = workspace_root() / "marconi-runs" / "conversions"
    d.mkdir(parents=True, exist_ok=True)
    return d
