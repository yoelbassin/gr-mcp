from __future__ import annotations

import hashlib
import http.client
import os
import struct
import tempfile
import urllib.error
import urllib.request
import zlib
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


_EOCD_SIG = b"PK\x05\x06"
_EOCD_SCAN = 65536


def _content_length(url: str, ua: str | None) -> int:
    req = urllib.request.Request(url, method="HEAD")
    if ua:
        req.add_header("User-Agent", ua)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:  # noqa: S310
            return int(r.headers["Content-Length"])
    except urllib.error.URLError as exc:
        raise FetchError(f"{url}: {exc}") from exc


def _range(url: str, ua: str | None, start: int, end: int) -> bytes:
    with _open(url, ua, {"Range": f"bytes={start}-{end}"}) as r:
        data: bytes = r.read()
    return data


def _central_directory(url: str, ua: str | None) -> list[tuple[str, int, int, int]]:
    total = _content_length(url, ua)
    tail = _range(url, ua, max(0, total - _EOCD_SCAN), total - 1)
    i = tail.rfind(_EOCD_SIG)
    if i < 0:
        raise FetchError(f"{url}: no zip end-of-central-directory found")
    count = struct.unpack("<H", tail[i + 10 : i + 12])[0]
    size = struct.unpack("<I", tail[i + 12 : i + 16])[0]
    offset = struct.unpack("<I", tail[i + 16 : i + 20])[0]
    cd = _range(url, ua, offset, offset + size - 1)
    out: list[tuple[str, int, int, int]] = []
    p = 0
    for _ in range(count):
        method = struct.unpack("<H", cd[p + 10 : p + 12])[0]
        csize = struct.unpack("<I", cd[p + 20 : p + 24])[0]
        usize = struct.unpack("<I", cd[p + 24 : p + 28])[0]
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28 : p + 34])
        local = struct.unpack("<I", cd[p + 42 : p + 46])[0]
        name = cd[p + 46 : p + 46 + nlen].decode("utf-8", "replace")
        out.append((name, usize, local, (method << 32) | csize))
        p += 46 + nlen + elen + clen
    return out


def zip_members(url: str, *, ua: str | None = None) -> dict[str, int]:
    return {name: usize for name, usize, _, _ in _central_directory(url, ua)}


def _fetch_zip_member(asset: FetchedAsset, tmp: Path) -> None:
    ua = BROWSER_UA if asset.strategy is Strategy.ZIP_MEMBER else None
    entries = _central_directory(asset.url, ua)
    row = next((e for e in entries if e[0] == asset.member), None)
    if row is None:
        raise FetchError(
            f"{asset.path}: member {asset.member!r} not in {asset.url}; "
            f"present: {sorted(n for n, _, _, _ in entries)[:8]}"
        )
    _name, _usize, local_off, packed = row
    method, csize = packed >> 32, packed & 0xFFFFFFFF
    if method not in (0, 8):
        raise FetchError(
            f"{asset.path}: member {asset.member!r} uses unsupported "
            f"zip compression method {method}"
        )
    if csize == 0:
        tmp.write_bytes(b"")
        return
    header = _range(asset.url, ua, local_off, local_off + 29)
    nlen, elen = struct.unpack("<HH", header[26:30])
    start = local_off + 30 + nlen + elen
    raw = _range(asset.url, ua, start, start + csize - 1)
    data = zlib.decompress(raw, -zlib.MAX_WBITS) if method == 8 else raw
    tmp.write_bytes(data)


_STRATEGIES[Strategy.ZIP_MEMBER] = _fetch_zip_member
