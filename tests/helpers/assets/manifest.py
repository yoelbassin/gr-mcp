from __future__ import annotations

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from helpers._paths import TESTS_ROOT
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

MANIFEST_PATH = TESTS_ROOT / "assets.toml"


class ManifestError(Exception):
    pass


class Strategy(StrEnum):
    PLAIN = "plain"
    BROWSER = "browser"
    ZIP_MEMBER = "zip_member"
    DRIVE_ZIP_MEMBER = "drive_zip_member"


_ZIP_STRATEGIES = frozenset({Strategy.ZIP_MEMBER, Strategy.DRIVE_ZIP_MEMBER})


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str | None = None
    bytes: int | None = None
    ci_required: bool = False
    note: str = ""


class FetchedAsset(_Base):
    kind: Literal["fetched"]
    url: str
    strategy: Strategy = Strategy.PLAIN
    member: str | None = None
    discard_after_derive: bool = False


class DerivedAsset(_Base):
    kind: Literal["derived"]
    derive_from: str
    derive: str


class LocalAsset(_Base):
    kind: Literal["local"]


Asset = Annotated[FetchedAsset | DerivedAsset | LocalAsset, Field(discriminator="kind")]

_ADAPTER: TypeAdapter[list[Asset]] = TypeAdapter(list[Asset])


def _check_members(assets: list[Asset]) -> None:
    for a in assets:
        if isinstance(a, FetchedAsset) and a.strategy in _ZIP_STRATEGIES:
            if not a.member:
                raise ManifestError(
                    f"{a.path}: strategy {a.strategy.value!r} needs a 'member'"
                )


def _index(assets: list[Asset]) -> dict[str, Asset]:
    out: dict[str, Asset] = {}
    for a in assets:
        if a.path in out:
            raise ManifestError(f"duplicate asset path {a.path!r}")
        out[a.path] = a
    return out


def _check_graph(index: dict[str, Asset]) -> None:
    for name, a in index.items():
        if isinstance(a, DerivedAsset) and a.derive_from not in index:
            raise ManifestError(f"{name}: derive_from {a.derive_from!r} names no asset")
    for name in index:
        seen: set[str] = set()
        cur = name
        while True:
            node = index[cur]
            if not isinstance(node, DerivedAsset):
                break
            if cur in seen:
                raise ManifestError(f"derive cycle through {cur!r}")
            seen.add(cur)
            cur = node.derive_from


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Asset]:
    raw: dict[str, Any] = tomllib.loads(path.read_text())
    try:
        assets = _ADAPTER.validate_python(raw.get("asset", []))
    except ValidationError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    _check_members(assets)
    index = _index(assets)
    _check_graph(index)
    return index
