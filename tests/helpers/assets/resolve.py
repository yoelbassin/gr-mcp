from __future__ import annotations

from pathlib import Path

from helpers._paths import REPO_ARTIFACTS
from helpers.assets.derive import DERIVERS
from helpers.assets.fetch import FetchError, fetch
from helpers.assets.manifest import Asset, FetchedAsset, LocalAsset

ASSET_ROOT = REPO_ARTIFACTS / "assets"


class ResolveError(Exception):
    pass


def _local_missing(asset: LocalAsset, dest: Path) -> ResolveError:
    hint = asset.note or "no upstream is recorded for this asset"
    return ResolveError(f"{asset.path}: absent at {dest} and {hint}")


def resolve(name: str, index: dict[str, Asset], root: Path = ASSET_ROOT) -> Path:
    asset = index.get(name)
    if asset is None:
        raise ResolveError(f"{name!r} is not in the manifest")
    dest = root / name
    if dest.exists():
        return dest
    if isinstance(asset, LocalAsset):
        raise _local_missing(asset, dest)
    if isinstance(asset, FetchedAsset):
        try:
            fetch(asset, dest)
        except FetchError as exc:
            raise ResolveError(str(exc)) from exc
        return dest
    deriver = DERIVERS.get(asset.derive)
    if deriver is None:
        raise ResolveError(
            f"{name}: deriver {asset.derive!r} is not registered; "
            f"known: {sorted(DERIVERS)}"
        )
    src = resolve(asset.derive_from, index, root=root)
    deriver(src, dest)
    if not dest.exists():
        raise ResolveError(f"{name}: deriver {asset.derive!r} wrote nothing")
    parent = index[asset.derive_from]
    if isinstance(parent, FetchedAsset) and parent.discard_after_derive:
        src.unlink(missing_ok=True)
    return dest


def resolve_all(index: dict[str, Asset], root: Path = ASSET_ROOT) -> list[str]:
    unresolved: list[str] = []
    for name, asset in index.items():
        if isinstance(asset, FetchedAsset) and asset.discard_after_derive:
            continue
        if isinstance(asset, LocalAsset):
            continue
        try:
            resolve(name, index, root=root)
        except ResolveError:
            unresolved.append(name)
    return unresolved


def absent_local_assets(index: dict[str, Asset], root: Path = ASSET_ROOT) -> list[str]:
    return [
        name
        for name, asset in index.items()
        if isinstance(asset, LocalAsset) and not (root / name).exists()
    ]
