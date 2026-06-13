import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_plugin_json_valid():
    data = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert data["name"] == "marconi"
    assert "version" in data and "description" in data


def test_marketplace_json_lists_marconi():
    data = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    names = [p["name"] for p in data["plugins"]]
    assert "marconi" in names


def test_mcp_json_declares_server():
    data = json.loads((ROOT / ".mcp.json").read_text())
    server = data["mcpServers"]["marconi"]
    assert server["command"] == "uv"
    assert "marconi-mcp" in server["args"]
    assert "${CLAUDE_PLUGIN_ROOT}" in server["args"]
    # fastmcp is an optional extra; the plugin must launch with it
    assert "--extra" in server["args"] and "mcp" in server["args"]
