def test_list_blocks_tool_returns_vocabulary(server_state):
    from marconi.mcp.tools import list_blocks

    blocks = list_blocks()
    assert "nbfm_rx" in blocks
    assert blocks["nbfm_rx"]["inputs"] == ["c"]
    assert blocks["nbfm_rx"]["outputs"] == ["f"]
    names = {p["name"] for p in blocks["nbfm_rx"]["params"]}
    assert {"audio_rate", "quad_rate"} <= names


def test_tools_registry_count():
    from marconi.mcp.tools import TOOLS

    assert len(TOOLS) >= 1
    assert "list_blocks" in TOOLS


def test_build_server_registers_all_tools():
    from marconi.mcp.server import build_server
    from marconi.mcp.tools import TOOLS

    mcp = build_server()
    assert mcp is not None
    assert len(TOOLS) >= 1


async def test_tools_listed_through_mcp_client(server_state):
    from fastmcp import Client

    from marconi.mcp.server import build_server
    from marconi.mcp.tools import TOOLS

    mcp = build_server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
    names = {t.name for t in tools}
    assert set(TOOLS) <= names and "list_blocks" in names


def test_main_initializes_state_from_env(tmp_path, monkeypatch):
    import marconi.mcp.server as server
    from marconi.mcp.state import get_state, reset_state

    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    monkeypatch.setattr("fastmcp.FastMCP.run", lambda self: None)
    server.main()
    assert get_state().workspace.root == tmp_path
    reset_state()


def test_base_import_does_not_load_mcp_or_fastmcp():
    import subprocess
    import sys

    # Run in a fresh interpreter so other test modules' top-level imports of
    # fastmcp/marconi.mcp can't pollute this check.
    code = (
        "import marconi, sys; "
        "assert 'fastmcp' not in sys.modules, 'fastmcp leaked into base import'; "
        "assert 'gnuradio' not in sys.modules, 'gnuradio leaked into base import'; "
        "assert 'marconi.mcp.server' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mcp_subpackage_importable():
    import marconi.mcp  # noqa: F401
    import marconi.mcp.errors  # noqa: F401


def test_entry_point_target_exists():
    from marconi.mcp.server import main  # noqa: F401
