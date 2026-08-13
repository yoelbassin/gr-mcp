from __future__ import annotations

import random
import zipfile
from pathlib import Path

import pytest
from helpers.assets.fetch import FetchError, fetch, zip_members
from helpers.assets.manifest import FetchedAsset, Strategy

WANTED = bytes(range(256)) * 512
# Incompressible and larger than fetch._EOCD_SCAN: the skipped member must
# outweigh the fixed tail-scan request, or served_bytes < archive can never
# hold no matter how small the wanted member is.
OTHER = random.Random(1234).randbytes(120_000)


def _zip(root: Path) -> None:
    with zipfile.ZipFile(root / "a.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pack/other.bin", OTHER)
        z.writestr("pack/wanted.bin", WANTED)


def _asset(url: str, member: str) -> FetchedAsset:
    return FetchedAsset.model_validate(
        {
            "kind": "fetched",
            "path": "A/wanted.bin",
            "url": url,
            "strategy": Strategy.ZIP_MEMBER,
            "member": member,
        }
    )


def test_lists_members_without_downloading_the_archive(
    local_server: tuple[str, Path],
) -> None:
    base, root = local_server
    _zip(root)
    listing = zip_members(f"{base}/a.zip")
    assert listing == {"pack/other.bin": len(OTHER), "pack/wanted.bin": len(WANTED)}


def test_extracts_only_the_requested_member(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, root = local_server
    _zip(root)
    dest = tmp_path / "out.bin"
    fetch(_asset(f"{base}/a.zip", "pack/wanted.bin"), dest)
    assert dest.read_bytes() == WANTED


def test_only_the_members_bytes_cross_the_wire(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    # the entire premise of zip_member: during the provenance hunt this listed a
    # 539 MB archive from two range requests. Asserting the extracted bytes alone
    # would pass just as well if the whole archive had been downloaded.
    from unit.helpers.conftest import _Handler

    base, root = local_server
    _zip(root)
    archive = (root / "a.zip").stat().st_size
    _Handler.served_bytes = 0
    fetch(_asset(f"{base}/a.zip", "pack/wanted.bin"), tmp_path / "out.bin")
    assert 0 < _Handler.served_bytes < archive


def test_a_malformed_range_value_is_rejected_not_crashed(
    local_server: tuple[str, Path],
) -> None:
    import urllib.error
    import urllib.request

    base, root = local_server
    (root / "x.bin").write_bytes(WANTED)
    req = urllib.request.Request(f"{base}/x.bin", headers={"Range": "bytes=abc-5"})
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 416


@pytest.mark.parametrize(
    "header, size, expected",
    [
        ("bytes=0-9", 100, (0, 9)),
        ("bytes=10-", 100, (10, 99)),
        ("bytes=-20", 100, (80, 99)),
        ("bytes=95-500", 100, (95, 99)),
        ("bytes=200-300", 100, None),
        ("bytes=abc-5", 100, None),
        ("items=0-9", 100, None),
    ],
)
def test_parse_range_covers_every_form_it_claims(
    header: str, size: int, expected: tuple[int, int] | None
) -> None:
    from unit.helpers.conftest import _parse_range

    assert _parse_range(header, size) == expected


def test_missing_member_is_named_in_the_error(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, root = local_server
    _zip(root)
    with pytest.raises(FetchError, match="pack/absent.bin"):
        fetch(_asset(f"{base}/a.zip", "pack/absent.bin"), tmp_path / "o.bin")


def test_unsupported_compression_method_is_named_in_the_error(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, root = local_server
    with zipfile.ZipFile(root / "b.zip", "w") as z:
        z.writestr("odd.bin", b"hello world" * 100, compress_type=zipfile.ZIP_BZIP2)
    with pytest.raises(FetchError, match="12"):
        fetch(_asset(f"{base}/b.zip", "odd.bin"), tmp_path / "o.bin")


def test_zero_byte_member_extracts_as_an_empty_file(
    local_server: tuple[str, Path], tmp_path: Path
) -> None:
    base, root = local_server
    with zipfile.ZipFile(root / "c.zip", "w", zipfile.ZIP_STORED) as z:
        z.writestr("empty.bin", b"")
    dest = tmp_path / "o.bin"
    fetch(_asset(f"{base}/c.zip", "empty.bin"), dest)
    assert dest.exists()
    assert dest.read_bytes() == b""
