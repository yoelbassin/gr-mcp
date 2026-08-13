from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest


class _Handler(http.server.SimpleHTTPRequestHandler):
    require_browser_ua = False

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.require_browser_ua and "Mozilla" not in self.headers.get(
            "User-Agent", ""
        ):
            self.send_error(403, "Forbidden")
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self.require_browser_ua and "Mozilla" not in self.headers.get(
            "User-Agent", ""
        ):
            self.send_error(403, "Forbidden")
            return
        super().do_HEAD()


@pytest.fixture
def local_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    root = tmp_path / "www"
    root.mkdir()

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
