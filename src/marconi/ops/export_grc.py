"""Export a PipelineSpec as a GNU Radio Companion .grc file.

.grc is the format for handing work to humans: open in GRC, tweak, keep.
Parameter values are strings because GRC evaluates them as Python.
"""

import math
from pathlib import Path

import yaml

from marconi.models import BlockSpec, PipelineSpec
from marconi.vocabulary import PipelineValidationError, validate_pipeline
from marconi.workspace import Workspace


def _block_states(i: int) -> dict:
    return {
        "bus_sink": False,
        "bus_source": False,
        "bus_structure": None,
        "coordinate": [200 + 250 * (i % 4), 100 + 150 * (i // 4)],
        "rotation": 0,
        "state": "enabled",
    }


def _map_block(b: BlockSpec, rate: float) -> tuple[str, dict[str, str]]:
    p = b.params
    r = float(p.get("sample_rate", rate))
    if b.type == "tone_source":
        return "analog_sig_source_x", {
            "type": "complex",
            "samp_rate": str(r),
            "waveform": "analog.GR_COS_WAVE",
            "freq": str(p["freq"]),
            "amp": str(p.get("amplitude", 1.0)),
            "offset": "0",
            "phase": "0",
            "showports": "False",
        }
    if b.type == "audio_tone_source":
        return "analog_sig_source_x", {
            "type": "float",
            "samp_rate": str(r),
            "waveform": "analog.GR_COS_WAVE",
            "freq": str(p["freq"]),
            "amp": str(p.get("amplitude", 0.5)),
            "offset": "0",
            "phase": "0",
            "showports": "False",
        }
    if b.type == "noise_source":
        return "analog_noise_source_x", {
            "type": "complex",
            "noise_type": "analog.GR_GAUSSIAN",
            "amp": str(p["amplitude"]),
            "seed": str(p.get("seed", 0)),
        }
    if b.type == "file_source":
        return "blocks_file_source", {
            "file": str(p["path"]),
            "type": "complex",
            "repeat": str(bool(p.get("repeat", False))),
            "vlen": "1",
            "begin_tag": "pmt.PMT_NIL",
            "offset": "0",
            "length": "0",
        }
    if b.type == "head":
        return "blocks_head", {
            "type": "complex",
            "num_items": str(int(p["num_samples"])),
            "vlen": "1",
        }
    if b.type == "add":
        return "blocks_add_xx", {"type": "complex", "num_inputs": "2", "vlen": "1"}
    if b.type == "multiply_const":
        return "blocks_multiply_const_vxx", {
            "type": "complex",
            "const": str(p["value"]),
            "vlen": "1",
        }
    if b.type == "freq_shift":
        return "blocks_rotator_cc", {
            "phase_inc": str(2.0 * math.pi * float(p["offset"]) / r),
            "tag_inc_update": "False",
        }
    if b.type == "freq_xlating_lowpass":
        taps = f"firdes.low_pass(1.0, {r}, {p['cutoff']}, {p['transition']})"
        return "freq_xlating_fir_filter_xxx", {
            "type": "ccf",
            "decim": str(int(p["decimation"])),
            "taps": taps,
            "center_freq": str(p["center_offset"]),
            "samp_rate": str(r),
        }
    if b.type == "quadrature_demod":
        return "analog_quadrature_demod_cf", {"gain": str(p.get("gain", 1.0))}
    if b.type in ("rational_resampler_f", "rational_resampler_c"):
        return "rational_resampler_xxx", {
            "type": "fff" if b.type.endswith("_f") else "ccc",
            "interp": str(int(p["interpolation"])),
            "decim": str(int(p["decimation"])),
            "taps": "[]",
            "fbw": "0",
        }
    if b.type == "fm_deemphasis":
        return "analog_fm_deemph", {
            "samp_rate": str(r),
            "tau": str(p.get("tau", 75e-6)),
        }
    if b.type == "nbfm_rx":
        return "analog_nbfm_rx", {
            "audio_rate": str(int(p["audio_rate"])),
            "quad_rate": str(int(p["quad_rate"])),
            "tau": str(p.get("tau", 75e-6)),
            "max_dev": str(p.get("max_dev", 5e3)),
        }
    if b.type == "nbfm_tx":
        return "analog_nbfm_tx", {
            "audio_rate": str(int(p["audio_rate"])),
            "quad_rate": str(int(p["quad_rate"])),
            "tau": str(p.get("tau", 75e-6)),
            "max_dev": str(p.get("max_dev", 5e3)),
            "fh": "-1.0",
        }
    if b.type == "file_sink":
        return "blocks_file_sink", {
            "file": str(p["path"]),
            "type": "complex",
            "unbuffered": "False",
            "append": "False",
        }
    if b.type == "wav_sink":
        return "blocks_wavfile_sink", {
            "file": str(p["path"]),
            "nchan": "1",
            "samp_rate": str(int(p["sample_rate"])),
            "format": "FORMAT_WAV",
            "bits_per_sample1": "FORMAT_PCM_16",
            "append": "False",
        }
    raise ValueError(f"no .grc mapping for block type '{b.type}'")


def export_grc(spec: PipelineSpec, path: Path | str) -> Path:
    # Validate first (mirrors run_pipeline) so a missing required param surfaces
    # as a PipelineValidationError, not a bare KeyError from the _map_block
    # lookups below (which the MCP boundary would mislabel [not_found]).
    issues = validate_pipeline(spec)
    if issues:
        raise PipelineValidationError(issues)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    safe_id = spec.name.replace(" ", "_").replace("-", "_")

    blocks = []
    for i, b in enumerate(spec.blocks):
        grc_id, params = _map_block(b, spec.sample_rate)
        blocks.append(
            {
                "name": b.id,
                "id": grc_id,
                "parameters": params,
                "states": _block_states(i),
            }
        )

    doc = {
        "options": {
            "parameters": {
                "author": "",
                "catch_exceptions": "True",
                "category": "Custom",
                "cmake_opt": "",
                "comment": "",
                "copyright": "",
                "description": "",
                "gen_cmake": "On",
                "gen_linking": "dynamic",
                "generate_options": "no_gui",
                "hier_block_src_path": ".:",
                "id": safe_id,
                "max_nouts": "0",
                "output_language": "python",
                "placement": "(0,0)",
                "qt_qss_theme": "",
                "realtime_scheduling": "",
                "run": "True",
                "run_command": "{python} -u {filename}",
                "run_options": "run",
                "sizing_mode": "fixed",
                "thread_safe_setters": "",
                "title": spec.name,
                "window_size": "(1000,1000)",
            },
            "states": {
                "bus_sink": False,
                "bus_source": False,
                "bus_structure": None,
                "coordinate": [8, 8],
                "rotation": 0,
                "state": "enabled",
            },
        },
        "blocks": blocks,
        "connections": [
            [c.src_block, str(c.src_port), c.dst_block, str(c.dst_port)]
            for c in spec.connections
        ],
        "metadata": {"file_format": 1, "grc_version": "3.10.12.0"},
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def export_grc_to_workspace(
    spec: PipelineSpec, workspace: Workspace, name: str | None = None
) -> Path:
    """Export `spec` to a collision-free .grc under the workspace's pipelines/.
    The op layer owns path construction (the MCP tool stays thin marshalling)."""
    return export_grc(spec, workspace.new_grc_path(name or spec.name))
