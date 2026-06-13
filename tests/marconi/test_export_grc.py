import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from marconi.models import BlockSpec, ConnectionSpec, PipelineSpec
from marconi.ops.export_grc import export_grc

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
