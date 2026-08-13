from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

from helpers.assets.derive import DERIVERS, DeriveError, Deriver
from helpers.assets.fetch import FetchError, fetch
from helpers.assets.manifest import (
    Asset,
    DerivedAsset,
    FetchedAsset,
    LocalAsset,
    is_consumed,
)
from helpers.assets.store import IntegrityError, check, publish


class ResolveError(Exception):
    pass


@dataclass(frozen=True)
class Unresolved:
    name: str
    cause: str


# A changed upstream escapes the resolver as whatever the reader chokes on:
# DeriveError for an unexpected format, zlib.error for a truncated member,
# TypeError for a response with no Content-Length. Each is a named unresolved
# asset, never a traceback out of the CLI.
_FAILURES: tuple[type[Exception], ...] = (
    ResolveError,
    FetchError,
    DeriveError,
    zlib.error,
    TypeError,
)


def _local_missing(asset: LocalAsset, dest: Path) -> ResolveError:
    hint = asset.note or "no upstream is recorded for this asset"
    return ResolveError(f"{asset.path}: absent at {dest} and {hint}")


def _derive_into(asset: DerivedAsset, deriver: Deriver, src: Path, dest: Path) -> None:
    def _write(tmp: Path) -> None:
        deriver(src, tmp)
        if tmp.stat().st_size == 0:
            raise ResolveError(f"{asset.path}: deriver {asset.derive!r} wrote nothing")
        try:
            check(asset, tmp)
        except IntegrityError as exc:
            raise ResolveError(f"{asset.path}: {exc}") from exc

    publish(dest, _write)


def resolve(name: str, index: dict[str, Asset], root: Path) -> Path:
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
    src = resolve(asset.derive_from, index, root)
    _derive_into(asset, deriver, src, dest)
    if is_consumed(index[asset.derive_from]):
        src.unlink(missing_ok=True)
    return dest


def resolve_all(index: dict[str, Asset], root: Path) -> list[Unresolved]:
    unresolved: list[Unresolved] = []
    for name, asset in index.items():
        if is_consumed(asset) or isinstance(asset, LocalAsset):
            continue
        try:
            resolve(name, index, root)
        except _FAILURES as exc:
            unresolved.append(Unresolved(name, f"{type(exc).__name__}: {exc}"))
    return unresolved


def absent_local_assets(index: dict[str, Asset], root: Path) -> list[str]:
    return [
        name
        for name, asset in index.items()
        if isinstance(asset, LocalAsset) and not (root / name).exists()
    ]
