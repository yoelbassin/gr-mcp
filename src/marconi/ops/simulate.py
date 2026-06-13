from pathlib import Path

from marconi import sigmf
from marconi.models import (
    BlockSpec,
    CaptureRef,
    ConnectionSpec,
    PipelineSpec,
    SceneSpec,
)
from marconi.ops.pipeline import run_pipeline
from marconi.workspace import Workspace

_FM_AUDIO_RATE = 25_000
_FM_QUAD_RATE = 100_000


def scene_to_pipeline(
    scene: SceneSpec,
    center_freq: float,
    sample_rate: float,
    duration: float,
    out_path: Path | str,
) -> PipelineSpec:
    """Generate the pipeline that renders `scene` as seen by a receiver at
    center_freq/sample_rate for `duration` seconds, writing raw cf32 to
    out_path. Out-of-band elements (beyond ±45% of the rate) are skipped."""
    blocks: list[BlockSpec] = []
    connections: list[ConnectionSpec] = []
    outputs: list[str] = []

    for i, el in enumerate(scene.elements):
        offset = el.freq - center_freq
        if el.kind != "noise" and abs(offset) > sample_rate * 0.45:
            continue

        if el.kind == "tone":
            bid = f"tone{i}"
            blocks.append(
                BlockSpec(
                    id=bid,
                    type="tone_source",
                    params={"freq": offset, "amplitude": el.amplitude},
                )
            )
            outputs.append(bid)

        elif el.kind == "noise":
            bid = f"noise{i}"
            blocks.append(
                BlockSpec(
                    id=bid,
                    type="noise_source",
                    params={
                        "amplitude": el.amplitude,
                        "seed": int(el.params.get("seed", 0)),
                    },
                )
            )
            outputs.append(bid)

        elif el.kind == "fm_tone":
            if "mod_freq" not in el.params:
                raise ValueError(
                    f"fm_tone element '{el.kind}' at {el.freq} Hz requires "
                    "params.mod_freq (the audio modulation frequency in Hz)"
                )
            if sample_rate % _FM_QUAD_RATE != 0:
                raise ValueError(
                    f"fm_tone requires sample_rate to be a multiple of "
                    f"{_FM_QUAD_RATE}, got {sample_rate}"
                )
            interp = int(sample_rate // _FM_QUAD_RATE)
            blocks += [
                BlockSpec(
                    id=f"fmaudio{i}",
                    type="audio_tone_source",
                    params={
                        "freq": float(el.params["mod_freq"]),
                        "sample_rate": float(_FM_AUDIO_RATE),
                    },
                ),
                BlockSpec(
                    id=f"fmtx{i}",
                    type="nbfm_tx",
                    params={"audio_rate": _FM_AUDIO_RATE, "quad_rate": _FM_QUAD_RATE},
                ),
                BlockSpec(
                    id=f"fmrr{i}",
                    type="rational_resampler_c",
                    params={"interpolation": interp, "decimation": 1},
                ),
                BlockSpec(
                    id=f"fmamp{i}",
                    type="multiply_const",
                    params={"value": el.amplitude},
                ),
                BlockSpec(
                    id=f"fmshift{i}", type="freq_shift", params={"offset": offset}
                ),
            ]
            connections += [
                ConnectionSpec(src_block=f"fmaudio{i}", dst_block=f"fmtx{i}"),
                ConnectionSpec(src_block=f"fmtx{i}", dst_block=f"fmrr{i}"),
                ConnectionSpec(src_block=f"fmrr{i}", dst_block=f"fmamp{i}"),
                ConnectionSpec(src_block=f"fmamp{i}", dst_block=f"fmshift{i}"),
            ]
            outputs.append(f"fmshift{i}")

        elif el.kind == "iq_file":
            missing = {"path", "sample_rate"} - el.params.keys()
            if missing:
                raise ValueError(
                    f"iq_file element requires params {sorted(missing)} "
                    "(the source file path and its sample_rate in Hz)"
                )
            file_rate = float(el.params["sample_rate"])
            if file_rate != sample_rate:
                raise ValueError(
                    f"transmit capture sample rate {file_rate} does not match "
                    f"scene render rate {sample_rate} (v1.0 requires equal rates)"
                )
            blocks += [
                BlockSpec(
                    id=f"iq{i}",
                    type="file_source",
                    params={"path": str(el.params["path"]), "repeat": True},
                ),
                BlockSpec(
                    id=f"iqamp{i}",
                    type="multiply_const",
                    params={"value": el.amplitude},
                ),
                BlockSpec(
                    id=f"iqshift{i}", type="freq_shift", params={"offset": offset}
                ),
            ]
            connections += [
                ConnectionSpec(src_block=f"iq{i}", dst_block=f"iqamp{i}"),
                ConnectionSpec(src_block=f"iqamp{i}", dst_block=f"iqshift{i}"),
            ]
            outputs.append(f"iqshift{i}")

        else:
            raise ValueError(f"unknown scene element kind '{el.kind}'")

    if not outputs:
        raise ValueError("scene has no elements within the rendered band")

    current = outputs[0]
    for j, other in enumerate(outputs[1:]):
        adder = f"sum{j}"
        blocks.append(BlockSpec(id=adder, type="add", params={}))
        connections += [
            ConnectionSpec(src_block=current, dst_block=adder, dst_port=0),
            ConnectionSpec(src_block=other, dst_block=adder, dst_port=1),
        ]
        current = adder

    blocks += [
        BlockSpec(
            id="head", type="head", params={"num_samples": int(duration * sample_rate)}
        ),
        BlockSpec(id="sink", type="file_sink", params={"path": str(out_path)}),
    ]
    connections += [
        ConnectionSpec(src_block=current, dst_block="head"),
        ConnectionSpec(src_block="head", dst_block="sink"),
    ]

    return PipelineSpec(
        name=f"render_{scene.name}",
        sample_rate=sample_rate,
        blocks=blocks,
        connections=connections,
    )


def render_scene(
    scene: SceneSpec,
    center_freq: float,
    sample_rate: float,
    duration: float,
    workspace: Workspace,
    name: str | None = None,
    timeout: float = 60.0,
) -> CaptureRef:
    """Render a scene to a SigMF capture in the workspace."""
    base = workspace.new_capture_path(name or scene.name)
    data_path = base.with_name(base.name + ".sigmf-data")
    spec = scene_to_pipeline(scene, center_freq, sample_rate, duration, data_path)
    result = run_pipeline(spec, timeout=timeout)
    if result.status != "ok":
        raise RuntimeError(f"scene render failed ({result.status}): {result.error}")
    return sigmf.write_meta_for(data_path, center_freq, sample_rate)
