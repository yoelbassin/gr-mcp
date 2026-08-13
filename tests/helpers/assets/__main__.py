from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from helpers.assets.manifest import MANIFEST_PATH, Asset, load_manifest
from helpers.assets.resolve import ASSET_ROOT, resolve_all


@dataclass(frozen=True)
class _Args:
    command: str
    missing: bool
    manifest: Path
    root: Path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="helpers.assets")
    p.add_argument("command", choices=["fetch", "list", "verify"])
    p.add_argument("--missing", action="store_true")
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--root", type=Path, default=ASSET_ROOT)
    return p


def _parse_args(argv: list[str] | None) -> _Args:
    ns = _parser().parse_args(argv)
    return _Args(
        command=cast(str, ns.command),
        missing=cast(bool, ns.missing),
        manifest=cast(Path, ns.manifest),
        root=cast(Path, ns.root),
    )


def _run_list(index: dict[str, Asset], args: _Args) -> int:
    names = sorted(index)
    if args.missing:
        names = [name for name in names if not (args.root / name).exists()]
    for name in names:
        print(name)
    return 0


def _run_verify(index: dict[str, Asset], args: _Args) -> int:
    for name in sorted(index):
        path = args.root / name
        print(f"{'ok  ' if path.exists() else 'MISS'} {name}")
    return 0


def _run_fetch(index: dict[str, Asset], args: _Args) -> int:
    unresolved = resolve_all(index, root=args.root)
    for name in sorted(unresolved):
        print(f"unresolved: {name}")
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
