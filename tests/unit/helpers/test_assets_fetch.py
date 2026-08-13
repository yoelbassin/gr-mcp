from __future__ import annotations

from pathlib import Path

import pytest
from helpers.assets.fetch import FetchError, fetch, head_prefix
from helpers.assets.manifest import FetchedAsset, Strategy
from unit.helpers.conftest import _Handler

PAYLOAD = bytes(range(256)) * 64


def _asset(url: str, **kw: object) -> FetchedAsset:
    return FetchedAsset.model_validate(
        {"kind": "fetched", "path": "A/x.bin", "url": url, **kw}
    )


def test_plain_fetch_writes_the_bytes(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, root = local_server
    (root / "x.bin").write_bytes(PAYLOAD)
    dest = tmp_path / "out.bin"
    fetch(_asset(f"{base}/x.bin"), dest)
    assert dest.read_bytes() == PAYLOAD


def test_checksum_mismatch_is_refused_and_leaves_no_file(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, root = local_server
    (root / "x.bin").write_bytes(PAYLOAD)
    dest = tmp_path / "out.bin"
    with pytest.raises(FetchError, match="sha256"):
        fetch(_asset(f"{base}/x.bin", sha256="00" * 32), dest)
    assert not dest.exists()


def test_size_mismatch_is_refused(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, root = local_server
    (root / "x.bin").write_bytes(PAYLOAD)
    with pytest.raises(FetchError, match="bytes"):
        fetch(_asset(f"{base}/x.bin", bytes=1), tmp_path / "out.bin")


def test_default_ua_is_rejected_but_browser_ua_succeeds(
    local_server: tuple[str, Path], browser_only: None, tmp_path: Path
) -> None:
    base, root = local_server
    (root / "x.bin").write_bytes(PAYLOAD)
    with pytest.raises(FetchError):
        fetch(_asset(f"{base}/x.bin"), tmp_path / "a.bin")
    dest = tmp_path / "b.bin"
    fetch(_asset(f"{base}/x.bin", strategy=Strategy.BROWSER), dest)
    assert dest.read_bytes() == PAYLOAD


def test_head_prefix_reads_only_the_first_n_bytes(
    local_server: tuple[str, Path],
) -> None:
    base, root = local_server
    (root / "x.bin").write_bytes(PAYLOAD)
    assert head_prefix(f"{base}/x.bin", 16) == PAYLOAD[:16]
    assert _Handler.last_status == 206
    assert _Handler.last_range == (0, 15)
