from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)
logger = logging.getLogger(__name__)


def write_yaml(data: object, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def read_yaml(path: Path | str) -> object:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_yaml_spec(spec: BaseModel, path: Path | str) -> Path:
    return write_yaml(spec.model_dump(mode="json"), path)


def load_yaml_spec(cls: type[M], path: Path | str) -> M:
    return cls.model_validate(read_yaml(path))


def list_specs(
    directory: Path, spec_name: Callable[[Path], str]
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if directory.is_dir():
        for p in sorted(directory.glob("*.yaml")):
            try:
                out.append({"name": spec_name(p), "path": str(p)})
            except Exception:
                logger.warning("skipping unreadable spec file %s", p, exc_info=True)
    return out
