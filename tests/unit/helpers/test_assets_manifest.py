from __future__ import annotations

from pathlib import Path

import pytest
from helpers.assets.manifest import (
    DerivedAsset,
    FetchedAsset,
    LocalAsset,
    ManifestError,
    Strategy,
    load_manifest,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "assets.toml"
    p.write_text(body)
    return p


def test_loads_each_node_kind(tmp_path: Path) -> None:
    m = load_manifest(
        _write(
            tmp_path,
            """
[[asset]]
kind = "fetched"
path = "POCSAG/POCSAG_IQ.zip"
url = "https://example.invalid/POCSAG_IQ.zip"
strategy = "browser"
discard_after_derive = true

[[asset]]
kind = "derived"
path = "POCSAG/pocsag.cf32"
derive_from = "POCSAG/POCSAG_IQ.zip"
derive = "pocsag_slice"
ci_required = true

[[asset]]
kind = "local"
path = "ADS-B/adsb.wav"
note = "upstream blocks automated requests"
""",
        )
    )
    assert set(m) == {
        "POCSAG/POCSAG_IQ.zip",
        "POCSAG/pocsag.cf32",
        "ADS-B/adsb.wav",
    }
    root = m["POCSAG/POCSAG_IQ.zip"]
    assert isinstance(root, FetchedAsset)
    assert root.strategy is Strategy.BROWSER
    assert root.discard_after_derive is True
    leaf = m["POCSAG/pocsag.cf32"]
    assert isinstance(leaf, DerivedAsset)
    assert leaf.derive_from == "POCSAG/POCSAG_IQ.zip"
    assert leaf.ci_required is True
    assert isinstance(m["ADS-B/adsb.wav"], LocalAsset)


def test_duplicate_paths_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(
            _write(
                tmp_path,
                """
[[asset]]
kind = "local"
path = "A/x.cf32"

[[asset]]
kind = "local"
path = "A/x.cf32"
""",
            )
        )


def test_unresolvable_derive_from_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="derive_from"):
        load_manifest(
            _write(
                tmp_path,
                """
[[asset]]
kind = "derived"
path = "A/x.cf32"
derive_from = "A/missing.zip"
derive = "slice"
""",
            )
        )


def test_cycles_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="cycle"):
        load_manifest(
            _write(
                tmp_path,
                """
[[asset]]
kind = "derived"
path = "A/x.cf32"
derive_from = "A/y.cf32"
derive = "s"

[[asset]]
kind = "derived"
path = "A/y.cf32"
derive_from = "A/x.cf32"
derive = "s"
""",
            )
        )


def test_zip_member_strategy_requires_a_member(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="member"):
        load_manifest(
            _write(
                tmp_path,
                """
[[asset]]
kind = "fetched"
path = "A/x.wav"
url = "https://example.invalid/a.zip"
strategy = "zip_member"
""",
            )
        )


def test_unknown_field_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ManifestError):
        load_manifest(
            _write(
                tmp_path,
                """
[[asset]]
kind = "local"
path = "A/x.cf32"
typo_field = 1
""",
            )
        )
