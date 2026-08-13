from __future__ import annotations

import hashlib
import http.client
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import cast

from helpers.assets.manifest import FetchedAsset, Strategy

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_TIMEOUT = 120.0
_CHUNK = 1 << 20


class FetchError(Exception):
    pass


def _open(
    url: str, ua: str | None, extra: dict[str, str] | None = None
) -> http.client.HTTPResponse:
    headers = dict(extra or {})
    if ua:
        headers["User-Agent"] = ua
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=_TIMEOUT)  # noqa: S310
        return cast(http.client.HTTPResponse, resp)
    except urllib.error.URLError as exc:
        raise FetchError(f"{url}: {exc}") from exc


def head_prefix(url: str, n: int, *, ua: str | None = None) -> bytes:
    with _open(url, ua, {"Range": f"bytes=0-{n - 1}"}) as r:
        return r.read(n)


def _verify(asset: FetchedAsset, tmp: Path) -> None:
    size = tmp.stat().st_size
    if asset.bytes is not None and size != asset.bytes:
        raise FetchError(f"{asset.path}: expected {asset.bytes} bytes, got {size}")
    if asset.sha256 is None:
        return
    h = hashlib.sha256()
    with tmp.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    if h.hexdigest() != asset.sha256:
        raise FetchError(
            f"{asset.path}: sha256 {h.hexdigest()} != manifest {asset.sha256}"
        )


def _stream_to(url: str, ua: str | None, tmp: Path) -> None:
    with _open(url, ua) as r, tmp.open("wb") as out:
        while chunk := r.read(_CHUNK):
            out.write(chunk)


def _fetch_plain(asset: FetchedAsset, tmp: Path) -> None:
    _stream_to(asset.url, None, tmp)


def _fetch_browser(asset: FetchedAsset, tmp: Path) -> None:
    _stream_to(asset.url, BROWSER_UA, tmp)


_STRATEGIES: dict[Strategy, Callable[[FetchedAsset, Path], None]] = {
    Strategy.PLAIN: _fetch_plain,
    Strategy.BROWSER: _fetch_browser,
}


def fetch(asset: FetchedAsset, dest: Path) -> None:
    handler = _STRATEGIES.get(asset.strategy)
    if handler is None:
        raise FetchError(f"{asset.path}: no handler for {asset.strategy.value!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    tmp = Path(name)
    try:
        handler(asset, tmp)
        _verify(asset, tmp)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
