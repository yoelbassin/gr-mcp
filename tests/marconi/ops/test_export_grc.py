import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from marconi.models import BlockSpec, ConnectionSpec, PipelineSpec
from marconi.ops.export_grc import export_grc, export_grc_to_workspace
from marconi.vocabulary import PipelineValidationError
from marconi.workspace import Workspace

GRCC = shutil.which("grcc") or "/opt/homebrew/bin/grcc"


def _rx_pipeline(tmp_path: Path) -> PipelineSpec:
    return PipelineSpec(
        name="nbfm_receiver",
        sample_rate=2e6,
        blocks=[
            BlockSpec(
                id="src", type="file_source", params={"path": str(tmp_path / "in.cf32")}
            ),
            BlockSpec(
                id="chan",
                type="freq_xlating_lowpass",
                params={
                    "decimation": 20,
                    "center_offset": 300e3,
                    "cutoff": 8e3,
                    "transition": 4e3,
                },
            ),
            BlockSpec(
                id="demod",
                type="nbfm_rx",
                params={"audio_rate": 25000, "quad_rate": 100000},
            ),
            BlockSpec(
                id="audio",
                type="wav_sink",
                params={"path": str(tmp_path / "out.wav"), "sample_rate": 25000},
            ),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="chan"),
            ConnectionSpec(src_block="chan", dst_block="demod"),
            ConnectionSpec(src_block="demod", dst_block="audio"),
        ],
    )


def test_export_structure(tmp_path: Path) -> None:
    out = export_grc(_rx_pipeline(tmp_path), tmp_path / "rx.grc")
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["options"]["parameters"]["id"] == "nbfm_receiver"
    ids = [b["id"] for b in doc["blocks"]]
    assert "analog_nbfm_rx" in ids and "blocks_file_source" in ids
    assert ["src", "0", "chan", "0"] in [list(c) for c in doc["connections"]]


def test_export_validates_before_mapping(tmp_path: Path) -> None:
    """A missing required param must raise PipelineValidationError, not a bare
    KeyError (which the MCP boundary would mislabel [not_found])."""
    spec = PipelineSpec(
        name="bad",
        sample_rate=1e6,
        blocks=[BlockSpec(id="src", type="tone_source", params={})],  # missing freq
        connections=[],
    )
    with pytest.raises(PipelineValidationError, match="freq"):
        export_grc(spec, tmp_path / "bad.grc")


def test_export_to_workspace_dedupes(tmp_path: Path) -> None:
    """Re-exporting the same name never overwrites a hand-tweaked .grc."""
    ws = Workspace(tmp_path)
    spec = _rx_pipeline(tmp_path)
    p1 = export_grc_to_workspace(spec, ws)
    p2 = export_grc_to_workspace(spec, ws)
    assert p1 != p2
    assert p1.exists() and p2.exists()
    assert p1.suffix == ".grc" and p2.suffix == ".grc"


def test_exported_grc_compiles(tmp_path: Path) -> None:
    if not Path(GRCC).exists():
        pytest.skip("grcc not available")
    out = export_grc(_rx_pipeline(tmp_path), tmp_path / "rx.grc")
    proc = subprocess.run(
        [GRCC, "-o", str(tmp_path), str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"grcc failed:\n{proc.stdout}\n{proc.stderr}"
    assert (tmp_path / "nbfm_receiver.py").exists()
