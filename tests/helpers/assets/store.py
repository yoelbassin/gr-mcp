from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from helpers.assets.manifest import Asset

_CHUNK = 1 << 20


class IntegrityError(Exception):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def check(asset: Asset, path: Path) -> None:
    size = path.stat().st_size
    if asset.bytes is not None and size != asset.bytes:
        raise IntegrityError(f"expected {asset.bytes} bytes, got {size}")
    if asset.sha256 is None:
        return
    got = sha256_of(path)
    if got != asset.sha256:
        raise IntegrityError(f"sha256 {got} != manifest {asset.sha256}")


def publish(dest: Path, write: Callable[[Path], None]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp = Path(name)
    try:
        write(tmp)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
