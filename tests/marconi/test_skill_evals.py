from marconi.mcp import tools as T


def test_simulate_then_survey_sequence(server_state):
    # simulate-scene → survey-spectrum documented sequence
    T.simulate_scene(
        "band",
        [
            {"kind": "tone", "freq": 100.2e6, "amplitude": 0.6},
            {"kind": "tone", "freq": 99.7e6, "amplitude": 0.5},
            {"kind": "noise", "amplitude": 0.005},
        ],
    )
    cap = T.capture("band", center_freq=100e6, sample_rate=1e6, duration=0.1)
    sigs = T.find_signals(cap["path"])
    img = T.spectrogram(cap["path"])
    assert len(sigs) >= 2
    assert img["path"].endswith(".png")


def test_build_receiver_anti_overfit_cw_carrier(server_state):
    """build-receiver's CW worked example on a target the FM demo never uses:
    channelize the 99.5 MHz carrier to baseband and confirm it via measure."""
    T.simulate_scene(
        "sim0",
        [
            {
                "kind": "fm_tone",
                "freq": 100.3e6,
                "amplitude": 1.0,
                "params": {"mod_freq": 1e3},
            },
            {"kind": "tone", "freq": 99.5e6, "amplitude": 0.5},
            {"kind": "noise", "amplitude": 0.005},
        ],
    )
    cap = T.capture("sim0", center_freq=100e6, sample_rate=2e6, duration=0.1)
    # Channelize the CW carrier (offset = 99.5 - 100 = -0.5 MHz) down to baseband.
    chan_path = str(server_state.workspace.root / "captures" / "cw_chan.sigmf-data")
    pipeline = {
        "name": "cw_rx",
        "sample_rate": 2e6,
        "blocks": [
            {"id": "src", "type": "file_source", "params": {"path": cap["path"]}},
            {
                "id": "chan",
                "type": "freq_xlating_lowpass",
                "params": {
                    "decimation": 40,
                    "center_offset": -0.5e6,
                    "cutoff": 10e3,
                    "transition": 5e3,
                },
            },
            {"id": "head", "type": "head", "params": {"num_samples": 5000}},
            {"id": "sink", "type": "file_sink", "params": {"path": chan_path}},
        ],
        "connections": [
            {"src_block": "src", "dst_block": "chan"},
            {"src_block": "chan", "dst_block": "head"},
            {"src_block": "head", "dst_block": "sink"},
        ],
    }
    assert T.validate_pipeline(pipeline) == []
    assert T.run_pipeline(pipeline)["status"] == "ok"
    # The channelized capture is at 2e6/40 = 50 kHz, centered on the carrier.
    from marconi import sigmf

    sigmf.write_meta_for(chan_path, center_freq=99.5e6, sample_rate=50e3)
    m = T.measure(chan_path, center_freq=99.5e6, search_bandwidth=40e3)
    assert m["snr_db"] > 8  # the carrier is clearly present at baseband
