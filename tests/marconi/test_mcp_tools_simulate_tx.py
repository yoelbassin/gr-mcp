from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from marconi.mcp import tools as T


def test_render_scene_adhoc(server_state):
    ref = T.render_scene(
        [
            {"kind": "tone", "freq": 100.02e6, "amplitude": 0.5},
            {"kind": "noise", "amplitude": 0.005},
        ],
        center_freq=100e6,
        sample_rate=1e6,
        duration=0.05,
    )
    assert Path(ref["path"]).exists()
    assert ref["sample_rate"] == 1e6


def test_transmit_requires_confirmation(server_state):
    T.simulate_scene("sim0", [{"kind": "noise", "amplitude": 0.005}])
    payload = T.render_scene(
        [{"kind": "tone", "freq": 100e6, "amplitude": 0.3}],
        center_freq=100e6,
        sample_rate=1e6,
        duration=0.05,
    )
    with pytest.raises(ToolError) as ei:
        T.transmit_capture("sim0", payload["path"], freq=100e6)
    assert "[tx_not_confirmed]" in str(ei.value)


def test_transmit_confirmed_appends_element(server_state):
    T.simulate_scene("sim0", [{"kind": "noise", "amplitude": 0.005}])
    payload = T.render_scene(
        [{"kind": "tone", "freq": 100e6, "amplitude": 0.3}],
        center_freq=100e6,
        sample_rate=1e6,
        duration=0.05,
    )
    el = T.transmit_capture("sim0", payload["path"], freq=100e6, confirmed=True)
    assert el["kind"] == "iq_file"
    assert el["freq"] == 100e6
