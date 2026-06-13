from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from marconi.mcp import tools as T


def _tone_to_file_pipeline(out_path: str) -> dict:
    # tone_source -> head -> file_sink ; minimal runnable graph
    return {
        "name": "tone_dump",
        "sample_rate": 1e6,
        "blocks": [
            {
                "id": "src",
                "type": "tone_source",
                "params": {"freq": 1e5, "amplitude": 0.5},
            },
            {"id": "head", "type": "head", "params": {"num_samples": 4096}},
            {"id": "sink", "type": "file_sink", "params": {"path": out_path}},
        ],
        "connections": [
            {"src_block": "src", "dst_block": "head"},
            {"src_block": "head", "dst_block": "sink"},
        ],
    }


def test_validate_pipeline_ok(server_state, tmp_path):
    issues = T.validate_pipeline(_tone_to_file_pipeline(str(tmp_path / "o.bin")))
    assert issues == []


def test_validate_pipeline_reports_issues(server_state):
    bad = {
        "name": "bad",
        "sample_rate": 1e6,
        "blocks": [{"id": "x", "type": "nope", "params": {}}],
        "connections": [],
    }
    issues = T.validate_pipeline(bad)
    assert any("unknown block type" in i["message"] for i in issues)


def test_run_pipeline_invalid_raises_tool_error(server_state):
    bad = {
        "name": "bad",
        "sample_rate": 1e6,
        "blocks": [{"id": "x", "type": "nope", "params": {}}],
        "connections": [],
    }
    with pytest.raises(ToolError) as ei:
        T.run_pipeline(bad)
    assert "[validation_error]" in str(ei.value)


@pytest.mark.gnuradio
def test_run_pipeline_records_history(server_state, tmp_path):
    out = str(tmp_path / "o.bin")
    result = T.run_pipeline(_tone_to_file_pipeline(out))
    assert result["status"] == "ok"
    assert result["run_id"] == "run-1"
    assert Path(out).exists()
    runs = T.list_runs()
    assert runs[-1]["run_id"] == "run-1"


def test_save_pipeline_and_export_grc(server_state, tmp_path):
    spec = _tone_to_file_pipeline(str(tmp_path / "o.bin"))
    saved = T.save_pipeline(spec)
    grc = T.export_grc(spec)
    assert Path(saved["path"]).exists() and saved["path"].endswith(".yaml")
    assert Path(grc["path"]).exists() and grc["path"].endswith(".grc")
