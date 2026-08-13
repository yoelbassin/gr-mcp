from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from helpers.assets.manifest import MANIFEST_PATH, Asset, is_consumed, load_manifest
from helpers.assets.resolve import absent_local_assets, resolve_all
from helpers.assets.root import asset_root
from helpers.assets.store import IntegrityError, check


@dataclass(frozen=True)
class _Args:
    command: str
    missing: bool
    manifest: Path
    root: Path


@dataclass(frozen=True)
class _Verdict:
    label: str
    ok: bool
    detail: str = ""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="helpers.assets")
    p.add_argument("command", choices=["fetch", "list", "verify"])
    p.add_argument("--missing", action="store_true")
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--root", type=Path, default=None)
    return p


def _parse_args(argv: list[str] | None) -> _Args:
    ns = _parser().parse_args(argv)
    root = cast("Path | None", ns.root)
    return _Args(
        command=cast(str, ns.command),
        missing=cast(bool, ns.missing),
        manifest=cast(Path, ns.manifest),
        root=asset_root() if root is None else root,
    )


def _run_list(index: dict[str, Asset], args: _Args) -> int:
    names = sorted(index)
    if args.missing:
        names = [
            name
            for name in names
            if not (args.root / name).exists() and not is_consumed(index[name])
        ]
    for name in names:
        print(name)
    return 0


def _verify_one(asset: Asset, path: Path) -> _Verdict:
    if not path.exists():
        if is_consumed(asset):
            return _Verdict("used", True, "absent by design once derived")
        return _Verdict("MISS", not asset.ci_required)
    try:
        check(asset, path)
    except IntegrityError as exc:
        return _Verdict("BAD ", False, str(exc))
    return _Verdict("ok  ", True)


def _run_verify(index: dict[str, Asset], args: _Args) -> int:
    failed = False
    for name in sorted(index):
        verdict = _verify_one(index[name], args.root / name)
        failed = failed or not verdict.ok
        detail = f": {verdict.detail}" if verdict.detail else ""
        print(f"{verdict.label} {name}{detail}")
    return 1 if failed else 0


def _run_fetch(index: dict[str, Asset], args: _Args) -> int:
    unresolved = resolve_all(index, args.root)
    for item in sorted(unresolved, key=lambda u: u.name):
        print(f"unresolved: {item.name}: {item.cause}")
    for name in sorted(absent_local_assets(index, args.root)):
        print(f"local (no upstream): {name}")
    return 1 if unresolved else 0


_COMMANDS: dict[str, Callable[[dict[str, Asset], _Args], int]] = {
    "fetch": _run_fetch,
    "list": _run_list,
    "verify": _run_verify,
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    index = load_manifest(args.manifest)
    return _COMMANDS[args.command](index, args)


if __name__ == "__main__":
    sys.exit(main())
