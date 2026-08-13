from __future__ import annotations

import http.server
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    if not header.startswith("bytes=") or "," in header:
        return None
    start_s, _, end_s = header[len("bytes=") :].partition("-")
    try:
        if start_s:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        elif end_s:
            start = max(size - int(end_s), 0)
            end = size - 1
        else:
            return None
    except ValueError:
        return None
    end = min(end, size - 1)
    if start < 0 or start > end:
        return None
    return start, end


class _Handler(http.server.SimpleHTTPRequestHandler):
    require_browser_ua = False
    last_status: int | None = None
    last_range: tuple[int, int] | None = None
    served_bytes: int = 0

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _authorized(self) -> bool:
        if self.require_browser_ua and "Mozilla" not in self.headers.get(
            "User-Agent", ""
        ):
            self.send_error(403, "Forbidden")
            return False
        return True

    def do_GET(self) -> None:
        if not self._authorized():
            return
        range_header = self.headers.get("Range")
        if range_header is None:
            _Handler.last_status = 200
            _Handler.last_range = None
            path = self.translate_path(self.path)
            if os.path.isfile(path):
                _Handler.served_bytes += os.path.getsize(path)
            super().do_GET()
            return
        self._serve_range(range_header)

    def do_HEAD(self) -> None:
        if not self._authorized():
            return
        super().do_HEAD()

    def _serve_range(self, range_header: str) -> None:
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return
        size = os.path.getsize(path)
        span = _parse_range(range_header, size)
        if span is None:
            _Handler.last_status = 416
            _Handler.last_range = None
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        start, end = span
        with open(path, "rb") as f:
            f.seek(start)
            body = f.read(end - start + 1)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        _Handler.served_bytes += len(body)
        _Handler.last_status = 206
        _Handler.last_range = (start, end)


@pytest.fixture
def local_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    root = tmp_path / "www"
    root.mkdir()
    _Handler.last_status = None
    _Handler.last_range = None
    _Handler.served_bytes = 0

    def factory(*args: object, **kw: object) -> _Handler:
        return _Handler(*args, directory=str(root), **kw)  # type: ignore[arg-type]

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), factory)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}", root
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


@pytest.fixture
def browser_only() -> Iterator[None]:
    _Handler.require_browser_ua = True
    try:
        yield
    finally:
        _Handler.require_browser_ua = False
